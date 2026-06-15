from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import traci
except Exception:  # pragma: no cover
    traci = None

from aco_routing.aco_router import ACORouter
from aco_routing.congestion_costs import DEFAULT_COST_WEIGHTS, normalize_predictions
from aco_routing.pheromone_manager import PheromoneManager
from aco_routing.route_optimizer import RouteOptimizer
from emergency_system.accident_manager import AccidentManager
from emergency_system.aggressive_emergency_handler import AggressiveEmergencyHandler
from emergency_system.emergency_detector import EmergencyVehicleDetector
from emergency_system.emergency_eta_predictor import EmergencyETAPredictor
from emergency_system.emergency_priority import EmergencyPriorityManager
from emergency_system.emergency_router import EmergencyRouter
from emergency_system.green_corridor_manager import GreenCorridorManager
from emergency_system.hospital_router import HospitalRouter
from emergency_system.multi_emergency_manager import MultiEmergencyManager
from emergency_system.traffic_clearing import TrafficClearingSystem
from emergency_system.clearing_monitor import get_monitor
from integration.hybrid_smart_city_controller import HybridSmartCityController
from integration.rl_pinn_aco_integration import build_adjacency_and_edgeinfo, load_pinn, update_episode_metrics
from environment.sumo_environment import SumoEnvironment
import logging

logger = logging.getLogger(__name__)


class HybridEmergencyController:
    def __init__(self, config_file: str, pinn_model_path: Optional[str] = None, duration: int = 2000, use_gui: bool = False, lookahead_steps: int = 10, reroute_interval: int = 30, decision_interval: int = 5, pinn_interval: int = 30, emergency_hold_steps: int = 20, hospital_edges: Optional[Sequence[str]] = None, metrics_output_dir: str = 'results/emergency_metrics', run_name: str = 'full_emergency', enable_corridor: bool = True, seed: Optional[int] = None, debug_log_path: Optional[str] = None):
        self.repo_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        self.config_file = config_file
        self.pinn_model_path = pinn_model_path
        self.duration = int(duration)
        self.use_gui = bool(use_gui)
        self.lookahead_steps = int(lookahead_steps)
        self.reroute_interval = int(reroute_interval)
        self.decision_interval = int(decision_interval)
        self.pinn_interval = int(pinn_interval)
        self.emergency_hold_steps = int(emergency_hold_steps)
        self.hospital_edges = list(hospital_edges or [])
        self.metrics_output_dir = Path(metrics_output_dir)
        self.run_name = run_name
        self.enable_corridor = bool(enable_corridor)
        self.seed = None if seed is None else int(seed)
        self.debug_log_path = Path(debug_log_path) if debug_log_path else (self.metrics_output_dir / f'{self.run_name}_debug_log.csv')
        self._debug_header_written = False

        self.env = SumoEnvironment(self.config_file, use_gui=self.use_gui, num_seconds=self.duration)
        self.adjacency, self.edge_info = build_adjacency_and_edgeinfo(str(self.repo_root / 'sumo_network' / 'network.net.xml'))
        self.priority_manager = EmergencyPriorityManager()
        self.detector = EmergencyVehicleDetector(priority_manager=self.priority_manager)
        self.accident_manager = AccidentManager()
        self.traffic_clearing = TrafficClearingSystem(use_gui=self.use_gui)
        self.aggressive_handler = AggressiveEmergencyHandler()
        self.green_corridor = GreenCorridorManager(use_gui=self.use_gui)
        self.multi_manager = MultiEmergencyManager(priority_manager=self.priority_manager)
        self.eta_predictor = EmergencyETAPredictor(edge_info=self.edge_info)
        self.hospital_router = HospitalRouter(hospitals=self.hospital_edges, edge_info=self.edge_info)

        pheromone = PheromoneManager(self.adjacency, initial_pheromone=1.0, evaporation_rate=0.03)
        heuristic = lambda edge_id: self.edge_info.get(edge_id, {}).get('length', 1.0)
        self.aco = ACORouter(pheromone, heuristic, alpha=1.0, beta=2.0, ants=12)
        self.emergency_router = EmergencyRouter(
            self.aco,
            self.adjacency,
            edge_info=self.edge_info,
            accident_manager=self.accident_manager,
            traffic_clearing=self.traffic_clearing,
            priority_manager=self.priority_manager,
        )
        self.route_optimizer = RouteOptimizer(self.aco, self.adjacency, edge_info=self.edge_info, reroute_interval=self.reroute_interval, cost_weights=dict(DEFAULT_COST_WEIGHTS))

        if not self.pinn_model_path:
            logger.warning('No PINN model path provided; running with zeroed PINN fallback.')
            self.pinn_predictor = lambda edges, **kwargs: {edge: 0.0 for edge in edges}
        elif not os.path.exists(self.pinn_model_path):
            logger.warning('PINN model path does not exist: %s; running with zeroed PINN fallback.', self.pinn_model_path)
            self.pinn_predictor = lambda edges, **kwargs: {edge: 0.0 for edge in edges}
        else:
            self.pinn_predictor = load_pinn(self.pinn_model_path)
        self._last_emergency_reroute: Dict[str, int] = {}
        self._pinn_edges = list(self.adjacency.keys())
        self.last_pinn_step = 0
        self.last_checkpoint_step = 0
        self.perf_records = {
            'rl_decision': [],
            'pinn_inference': [],
            'aco_routing': [],
            'emergency_corridor': [],
            'traffic_clearing': [],
            'telemetry_logging': [],
            'sumo_traci': [],
            'run_loop': [],
        }
        self._debug_buffer = []
        self._debug_buffer_size = 50
        self.metrics_output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_csv = self.metrics_output_dir / f'{self.run_name}_metrics.csv'
        self.latest_json = self.metrics_output_dir / f'{self.run_name}_latest.json'
        self.events_dir = self.repo_root / 'results' / 'response_analysis'
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.emergency_events: List[Dict[str, object]] = []
        # stop audit structures
        self.stop_audit_enabled = False
        self._previous_speeds: Dict[str, float] = {}
        self._stop_state: Dict[str, Dict[str, object]] = {}
        self.stop_events: List[Dict[str, object]] = []
        self._last_clearing: Dict[str, int] = {'slowed': 0, 'rerouted': 0}

        try:
            HybridSmartCityController.decision_interval = max(1, self.decision_interval)
        except Exception:
            logger.exception('Failed to set HybridSmartCityController decision_interval')

    def _append_debug_row(self, row: Dict[str, object]) -> None:
        if self.debug_log_path is None:
            return
        self._debug_buffer.append(row)
        if len(self._debug_buffer) >= self._debug_buffer_size:
            self._flush_debug_rows()

    def _flush_debug_rows(self) -> None:
        if self.debug_log_path is None or not self._debug_buffer:
            return
        self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.debug_log_path.exists() or not self._debug_header_written
        with self.debug_log_path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self._debug_buffer[0].keys()))
            if write_header:
                writer.writeheader()
                self._debug_header_written = True
            writer.writerows(self._debug_buffer)
        self._debug_buffer.clear()

    def _profile(self, section: str, elapsed: float) -> None:
        self.perf_records.setdefault(section, []).append(elapsed)

    def _write_performance_breakdown(self, total_runtime: float) -> None:
        payload = {}
        for section, values in self.perf_records.items():
            section_total = sum(values)
            payload[section] = {
                'average_seconds': float(section_total / len(values)) if values else 0.0,
                'max_seconds': float(max(values)) if values else 0.0,
                'total_seconds': float(section_total),
                'percent_of_total_runtime': float((section_total / total_runtime * 100.0) if total_runtime > 0 else 0.0),
                'samples': int(len(values)),
            }
        payload['total_runtime_seconds'] = float(total_runtime)
        payload['total_measured_seconds'] = float(sum(sum(values) for values in self.perf_records.values()))
        performance_path = self.metrics_output_dir / 'performance_breakdown.json'
        performance_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    def _route_signalized_intersections(self, route_edges: Sequence[str]) -> int:
        if traci is None or not route_edges:
            return 0
        # Check if connection is alive before making traci calls
        try:
            if not traci.isLoaded():
                return 0
        except Exception:
            return 0
        route_edges = set(route_edges)
        tls_ids = set()
        try:
            for ts_id in traci.trafficlight.getIDList():
                lanes = traci.trafficlight.getControlledLanes(ts_id)
                for lane_id in lanes:
                    try:
                        edge_id = traci.lane.getEdgeID(lane_id)
                    except Exception:
                        continue
                    if edge_id in route_edges:
                        tls_ids.add(ts_id)
                        break
        except Exception:
            logger.exception('Failed to derive signalized intersections on route')
        return len(tls_ids)

    def _write_checkpoint(self, sim_step: int, current_predictions: Dict[str, float]) -> Dict[str, object]:
        metrics = {}
        try:
            metrics = self.env.get_episode_metrics()
        except Exception:
            metrics = {}

        active_vehicle_ids = self.detector.active_vehicle_ids()
        active_corridor_tls = self.green_corridor.active_tls()
        eta_summary = {}
        if active_vehicle_ids:
            first_vehicle = active_vehicle_ids[0]
            corridor_route = self.green_corridor.active_corridors.get(first_vehicle)
            if corridor_route is not None:
                eta_summary = self.eta_predictor.predict(corridor_route.route_edges, pinn_predictions=current_predictions, priority=corridor_route.priority).to_dict()

        preds = list(current_predictions.values()) if current_predictions else []
        payload: Dict[str, object] = {
            'step': int(sim_step),
            'timestamp': time.time(),
            'mode': 'full_emergency',
            'emergency_vehicle_count': len(active_vehicle_ids),
            'active_corridor_tls': len(active_corridor_tls),
            'blocked_edges': len(self.accident_manager.blocked_edges(sim_step)),
            'emergency_phase_count': len(self.green_corridor.active_corridors),
            'pinn_prediction_count': int(len(preds)),
            'pinn_prediction_mean': float(sum(preds) / len(preds)) if preds else 0.0,
            'pinn_prediction_max': float(max(preds)) if preds else 0.0,
            **metrics,
            **eta_summary,
        }
        # include clearing stats for observability
        try:
            payload['last_cleared_slowed'] = int(self._last_clearing.get('slowed', 0))
            payload['last_cleared_rerouted'] = int(self._last_clearing.get('rerouted', 0))
        except Exception:
            logger.exception('Failed to read last_clearing for checkpoint payload')
            payload['last_cleared_slowed'] = 0
            payload['last_cleared_rerouted'] = 0

        write_header = not self.checkpoint_csv.exists()
        with self.checkpoint_csv.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(payload)
        self.latest_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        return payload

    def _current_predictions(self, sim_step: int) -> Dict[str, float]:
        start = time.perf_counter()
        if self.pinn_predictor is None:
            self._profile('pinn_inference', time.perf_counter() - start)
            return {}
        if sim_step - self.last_pinn_step < self.pinn_interval:
            self._profile('pinn_inference', time.perf_counter() - start)
            return getattr(self, '_last_predictions', {})
        self.last_pinn_step = sim_step
        predictions = self.pinn_predictor(self._pinn_edges, future_step=sim_step, lookahead_steps=self.lookahead_steps)
        predictions = normalize_predictions(predictions)
        self._last_predictions = predictions
        self._profile('pinn_inference', time.perf_counter() - start)
        return predictions

    def _build_predictive_corridor_sequence(self, route_edges: Sequence[str], current_position_index: int, current_predictions: Dict[str, float], priority: int, lookahead_edges: int = 5) -> Optional[List[str]]:
        """
        Build a sequence of upcoming TLS for predictive corridor activation.
        
        Args:
            route_edges: Full route
            current_position_index: Current position in route (0-based)
            current_predictions: PINN predictions dict
            priority: Emergency priority
            lookahead_edges: How many edges ahead to look for TLS
        
        Returns:
            Ordered list of upcoming TLS IDs, or None if unable to compute
        """
        if not route_edges or current_position_index >= len(route_edges):
            return None
        
        # Extract lookahead edges from current position
        lookahead_start = max(0, current_position_index)
        lookahead_end = min(len(route_edges), current_position_index + lookahead_edges)
        lookahead_edges_list = route_edges[lookahead_start:lookahead_end]
        
        if not lookahead_edges_list:
            return None
        
        # Get all TLS that control these lookahead edges
        upcoming_tls = self.green_corridor._route_tls_candidates(lookahead_edges_list)
        return upcoming_tls if upcoming_tls else None

    def _route_index_for_edge(self, route_edges: Sequence[str], current_edge: str, route_index: int) -> int:
        if not route_edges:
            return 0
        if 0 <= int(route_index) < len(route_edges) and route_edges[int(route_index)] == current_edge:
            return int(route_index)
        try:
            return max(0, route_edges.index(current_edge))
        except Exception:
            return max(0, min(int(route_index), len(route_edges) - 1))

    def _build_visual_corridor(self, route_edges: Sequence[str], current_edge: str, route_index: int, max_distance_m: float = 200.0) -> Dict[str, object]:
        corridor_edges: List[str] = []
        corridor_length = 0.0
        idx = self._route_index_for_edge(route_edges, current_edge, route_index)

        for edge_id in route_edges[idx:]:
            corridor_edges.append(edge_id)
            try:
                corridor_length += float(self.edge_info.get(edge_id, {}).get('length', 200.0))
            except Exception:
                corridor_length += 200.0
            if corridor_length >= float(max_distance_m):
                break

        return {
            'corridor_edges': corridor_edges,
            'corridor_length_m': corridor_length,
            'future_route': list(route_edges[idx:]),
            'route_index': idx,
        }

    def _vehicles_remaining_in_corridor(self, corridor_edges: Sequence[str], emergency_vehicle_id: str) -> List[str]:
        if traci is None:
            return []
        remaining: List[str] = []
        try:
            for edge_id in corridor_edges:
                for vehicle_id in traci.edge.getLastStepVehicleIDs(edge_id):
                    if vehicle_id == emergency_vehicle_id:
                        continue
                    if vehicle_id not in remaining:
                        remaining.append(vehicle_id)
        except Exception:
            return remaining
        return remaining

    def _handle_emergencies(self, sim_step: int, predictions: Dict[str, float]) -> None:
        detections = self.detector.scan(sim_step)
        routes: Dict[str, List[str]] = {}
        corridor_tls: Dict[str, List[str]] = {}
        eta_by_vehicle: Dict[str, float] = {}

        for vehicle_id, obs in detections.items():
            route = list(obs.route)
            if not route:
                continue

            logger.info(
                'Detected emergency vehicle %s at step %s on edge %s speed=%.1f priority=%s',
                vehicle_id, sim_step, obs.road_id, obs.speed, obs.priority,
            )

            debug_row = {
                'step': int(sim_step),
                'vehicle_id': vehicle_id,
                'current_edge': obs.road_id,
                'current_lane': obs.lane_id,
                'current_speed': float(obs.speed),
                'route_destination': route[-1] if route else '',
                'route_length': len(route),
                'future_route': ';'.join(route[max(0, obs.route_index): max(0, obs.route_index) + 20]),
                'active_corridor_intersections': 0,
                'traffic_light_changes': 0,
                'vehicles_cleared': 0,
                'vehicles_rerouted': 0,
                'waiting_time': 0.0,
                'signalized_intersections_on_route': self._route_signalized_intersections(route),
                'predictive_corridor_tls': 0,
            }

            # If a hospital edge is available, route toward the fastest one.
            if self.hospital_edges:
                hospital_plan = self.hospital_router.choose_best_hospital(vehicle_id, self.hospital_edges, self.emergency_router, pinn_predictions=predictions, priority=obs.priority)
                if hospital_plan.route:
                    route = hospital_plan.route

            original_route = list(route)
            if sim_step - self._last_emergency_reroute.get(vehicle_id, -10**9) >= self.reroute_interval:
                aco_start = time.perf_counter()
                rerouted = self.emergency_router.reroute_vehicle(vehicle_id, route[-1], predictions=predictions, priority=obs.priority, avoid_edges=self.accident_manager.blocked_edges(sim_step))
                self._profile('aco_routing', time.perf_counter() - aco_start)
                self._last_emergency_reroute[vehicle_id] = sim_step
                if rerouted and rerouted != original_route:
                    logger.info(
                        'Rerouted emergency vehicle %s from %s to %s',
                        vehicle_id, original_route, rerouted,
                    )
                    route = rerouted
                elif rerouted:
                    route = rerouted
            else:
                rerouted = route

            routes[vehicle_id] = route
            
            if self.enable_corridor:
                corridor_start = time.perf_counter()
                existing_corridor = self.green_corridor.active_corridors.get(vehicle_id)
                corridor_plan = self._build_visual_corridor(route, obs.road_id, obs.route_index, max_distance_m=200.0)
                corridor_edges = corridor_plan['corridor_edges']
                debug_row['corridor_length_m'] = float(corridor_plan['corridor_length_m'])
                debug_row['corridor_edges'] = ';'.join(corridor_edges)
                debug_row['remaining_in_corridor'] = 0
                if existing_corridor and existing_corridor.route_edges == route and existing_corridor.priority == obs.priority:
                    existing_corridor.release_step = max(existing_corridor.release_step, sim_step + self.emergency_hold_steps)
                    corridor = existing_corridor
                    debug_row['predictive_corridor_tls'] = len(corridor.tls_ids)
                    debug_row['active_corridor_intersections'] = len(corridor.tls_ids)
                    debug_row['traffic_light_changes'] = len(corridor.phase_plan)
                    corridor_tls[vehicle_id] = corridor.tls_ids
                else:
                    predictive_tls_sequence = self._build_predictive_corridor_sequence(
                        route_edges=route,
                        current_position_index=max(0, obs.route_index),
                        current_predictions=predictions,
                        priority=obs.priority,
                        lookahead_edges=5,
                    )

                    if predictive_tls_sequence:
                        corridor = self.green_corridor.activate_predictive(
                            vehicle_id, corridor_edges or route, sim_step=sim_step,
                            hold_steps=self.emergency_hold_steps, priority=obs.priority,
                            future_tls_sequence=predictive_tls_sequence
                        )
                        debug_row['predictive_corridor_tls'] = len(predictive_tls_sequence)
                    else:
                        corridor = self.green_corridor.sync_citywide(
                            vehicle_id, corridor_edges or route, sim_step=sim_step,
                            hold_steps=self.emergency_hold_steps, priority=obs.priority
                        )

                    logger.info(
                        'Emergency %s corridor active at step %s on TLS %s; hold_steps=%s; predicted=%s',
                        vehicle_id, sim_step, corridor.tls_ids, self.emergency_hold_steps, bool(predictive_tls_sequence),
                    )
                    corridor_tls[vehicle_id] = corridor.tls_ids
                    debug_row['active_corridor_intersections'] = len(corridor.tls_ids)
                    debug_row['traffic_light_changes'] = len(corridor.phase_plan)
                self._profile('emergency_corridor', time.perf_counter() - corridor_start)

                clear_start = time.perf_counter()
                cleared = self.traffic_clearing.clear_around_route(
                    vehicle_id,
                    route,
                    reroute_router=self.emergency_router,
                    destination=route[-1] if route else obs.destination,
                    clear_all_corridor=True,
                    corridor_edges=corridor_edges or None,
                )
                self._profile('traffic_clearing', time.perf_counter() - clear_start)
                debug_row['remaining_in_corridor'] = len(cleared.get('vehicles_remaining_in_corridor', []))
            else:
                corridor_tls[vehicle_id] = []
                cleared = {'slowed': 0, 'rerouted': 0, 'slowed_vehicle_ids': [], 'rerouted_vehicle_ids': [], 'vehicles_remaining_in_corridor': []}
            
            debug_row['vehicles_cleared'] = int(cleared.get('slowed', 0))
            debug_row['vehicles_rerouted'] = int(cleared.get('rerouted', 0))
            debug_row['vehicles_cleared_ids'] = ';'.join(cleared.get('slowed_vehicle_ids', []) or [])
            debug_row['vehicles_rerouted_ids'] = ';'.join(cleared.get('rerouted_vehicle_ids', []) or [])
            debug_row['vehicles_remaining_in_corridor_ids'] = ';'.join(cleared.get('vehicles_remaining_in_corridor', []) or [])
            try:
                self._last_clearing['slowed'] = int(cleared.get('slowed', 0))
                self._last_clearing['rerouted'] = int(cleared.get('rerouted', 0))
            except Exception:
                self._last_clearing = {'slowed': 0, 'rerouted': 0}

            try:
                debug_row['waiting_time'] = float(traci.vehicle.getAccumulatedWaitingTime(vehicle_id))
            except Exception:
                debug_row['waiting_time'] = 0.0
            self._append_debug_row(debug_row)

            try:
                get_monitor().record_emergency_event({
                    'step': int(sim_step),
                    'vehicle_id': vehicle_id,
                    'current_edge': obs.road_id,
                    'current_lane': obs.lane_id,
                    'future_route': route[max(0, obs.route_index):],
                    'corridor_length_m': float(debug_row.get('corridor_length_m', 0.0)),
                    'corridor_edges': corridor_edges if self.enable_corridor else [],
                    'vehicles_moved_aside': list(dict.fromkeys((cleared.get('slowed_vehicle_ids', []) or []) + (cleared.get('rerouted_vehicle_ids', []) or []))),
                    'lane_changes_forced': int(len(cleared.get('slowed_vehicle_ids', []) or []) + len(cleared.get('rerouted_vehicle_ids', []) or [])),
                    'vehicles_rerouted': int(cleared.get('rerouted', 0)),
                    'vehicles_slowed': int(cleared.get('slowed', 0)),
                    'vehicles_remaining_in_corridor': list(cleared.get('vehicles_remaining_in_corridor', []) or []),
                })
            except Exception:
                pass

        self.multi_manager.update(detections, routes, corridor_tls, eta_by_vehicle=eta_by_vehicle)

    def run(self) -> Dict[str, object]:
        self.env.start_simulation(seed=self.seed)
        sim_step = 0
        current_predictions: Dict[str, float] = {}
        run_start = time.perf_counter()

        connection_alive = True
        try:
            while sim_step < self.duration and connection_alive:
                sim_start = time.perf_counter()
                try:
                    traci.simulationStep()
                except Exception as ex:
                    logger.exception('TraCI simulationStep failed at step %s; terminating run', sim_step)
                    connection_alive = False
                    break
                self._profile('sumo_traci', time.perf_counter() - sim_start)

                # Audit stops before other processing if enabled
                if self.stop_audit_enabled:
                    try:
                        self._audit_stops(sim_step)
                    except Exception:
                        logger.exception('Stop auditing failed at step %s', sim_step)
                sim_step += 1

                # Check connection health before continuing
                try:
                    if not traci.isLoaded():
                        logger.error('TraCI connection lost at step %s', sim_step)
                        connection_alive = False
                        break
                except Exception:
                    logger.error('Lost connection to TraCI at step %s', sim_step)
                    connection_alive = False
                    break

                update_episode_metrics(self.env)
                loop_start = time.perf_counter()
                self.accident_manager.step(sim_step)
                self.traffic_clearing.step()
                self.aggressive_handler.step()  # NEW: Aggressive emergency handling
                self.green_corridor.update(sim_step)

                current_predictions = self._current_predictions(sim_step)
                if HybridSmartCityController is not None:
                    try:
                        rl_start = time.perf_counter()
                        HybridSmartCityController.step_traci(traci, sim_step)
                        self._profile('rl_decision', time.perf_counter() - rl_start)
                    except Exception:
                        logger.exception('HybridSmartCityController.step_traci failed at step %s', sim_step)

                self._handle_emergencies(sim_step, current_predictions)

                if sim_step - self.last_checkpoint_step >= max(1, self.pinn_interval):
                    checkpoint_start = time.perf_counter()
                    self._write_checkpoint(sim_step, current_predictions)
                    self.last_checkpoint_step = sim_step
                    self._profile('telemetry_logging', time.perf_counter() - checkpoint_start)
                self._profile('run_loop', time.perf_counter() - loop_start)
        finally:
            try:
                self._write_checkpoint(sim_step, current_predictions)
            except Exception:
                logger.exception('Failed to write final checkpoint at step %s', sim_step)
            try:
                self._flush_debug_rows()
            except Exception:
                logger.exception('Failed to flush debug rows during shutdown')
            try:
                traci.close()
            except Exception:
                logger.exception('Failed to close TraCI connection cleanly')
            
            # Force close the socket to ensure connection is fully cleaned
            try:
                if traci.isLoaded():
                    traci.close(False)  # Close without waiting
            except Exception:
                pass
            
            # Hard reset of the traci connection
            try:
                import traci.main as traci_main
                if hasattr(traci_main, '_connections'):
                    traci_main._connections.clear()
            except Exception:
                pass

            try:
                total_runtime = time.perf_counter() - run_start
                self._write_performance_breakdown(total_runtime)
            except Exception:
                logger.exception('Failed to write performance breakdown')

            # write emergency lane-clearing reports
            try:
                try:
                    get_monitor().write_reports(str(self.metrics_output_dir))
                    logger.info('Wrote emergency lane clearing reports to %s', self.metrics_output_dir)
                except Exception:
                    logger.exception('Failed to write clearing reports')
            except Exception:
                logger.exception('Failed to finalize clearing reports')

            # finalize stop audit report if enabled
            try:
                if self.stop_audit_enabled:
                    report_path = self.events_dir / 'emergency_stop_root_cause_report.json'
                    report = self._generate_stop_report()
                    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
                    logger.info('Wrote stop audit report to %s', report_path)
            except Exception:
                logger.exception('Failed to write stop audit report')

        metrics = self.env.get_episode_metrics()
        eta_summary = {}
        if self.detector.active_vehicles:
            first = next(iter(self.detector.active_vehicles.values()))
            eta_summary = self.eta_predictor.predict(first.route, priority=first.priority).to_dict()

        summary = {
            'mode': 'full_emergency',
            'duration': self.duration,
            'emergency_vehicle_count': len(self.detector.active_vehicles),
            'active_corridors': len(self.green_corridor.active_corridors),
            'blocked_edges': len(self.accident_manager.blocked_edges()),
            **metrics,
            **eta_summary,
        }
        summary_path = self.metrics_output_dir / f'{self.run_name}_summary.json'
        summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return summary

    def _audit_stops(self, sim_step: int) -> None:
        if traci is None:
            return
        # Check all active emergency vehicles
        active = list(self.detector.active_vehicle_ids())
        for vid in active:
            try:
                speed = float(traci.vehicle.getSpeed(vid))
            except Exception:
                # Vehicle may have exited simulation; skip it
                self._previous_speeds.pop(vid, None)
                continue
            prev_speed = float(self._previous_speeds.get(vid, speed))

            # Transition from moving to stopped
            if prev_speed > 0.1 and speed <= 0.1:
                # record stop start
                edge = None
                lane = None
                try:
                    lane = traci.vehicle.getLaneID(vid)
                    edge = traci.lane.getEdgeID(lane)
                except Exception:
                    edge = traci.vehicle.getRoadID(vid)
                reason = 'unknown'
                signal_state = None
                blocking = []
                # check leader
                try:
                    leader = traci.vehicle.getLeader(vid)
                    if leader is not None:
                        leader_id, dist = leader
                        if dist is not None and dist < 3.0:
                            reason = 'queue_induced'
                            blocking.append(leader_id)
                except Exception:
                    leader = None

                # check traffic light control for lane
                try:
                    for tls in traci.trafficlight.getIDList():
                        lanes = traci.trafficlight.getControlledLanes(tls)
                        if lane in lanes:
                            # get state for this TLS and lane index
                            states = traci.trafficlight.getRedYellowGreenState(tls)
                            try:
                                idx = lanes.index(lane)
                                signal_state = states[idx] if idx < len(states) else states
                            except Exception:
                                signal_state = states
                            if reason == 'unknown':
                                reason = 'red_light'
                            break
                except Exception:
                    pass

                self._stop_state[vid] = {'start_step': int(sim_step), 'edge': edge, 'speed_before': float(prev_speed), 'reason': reason, 'signal_state': signal_state, 'blocking': blocking}

            # Transition from stopped to moving -> finalize stop event
            if prev_speed <= 0.1 and speed > 0.1 and vid in self._stop_state:
                s = self._stop_state.pop(vid)
                start = int(s.get('start_step', sim_step))
                end = int(sim_step)
                duration_steps = max(0, end - start)
                active_corridor = bool(self.green_corridor.active_corridors.get(vid))
                # augment blocking vehicles list
                if not s.get('blocking'):
                    try:
                        leader = traci.vehicle.getLeader(vid)
                        if leader is not None:
                            leader_id, dist = leader
                            if leader_id:
                                s['blocking'] = [leader_id]
                    except Exception:
                        pass

                event = {
                    'vehicle_id': vid,
                    'edge': s.get('edge'),
                    'start_step': start,
                    'end_step': end,
                    'duration_steps': duration_steps,
                    'duration_seconds': duration_steps, # 1s per step
                    'speed_before_stop': s.get('speed_before'),
                    'reason': s.get('reason'),
                    'signal_state': s.get('signal_state'),
                    'vehicles_blocking': s.get('blocking', []),
                    'active_corridor': active_corridor,
                }
                self.stop_events.append(event)

            # update previous speed
            self._previous_speeds[vid] = speed

    def _generate_stop_report(self) -> Dict[str, object]:
        total = len(self.stop_events)
        avg_duration = float(sum(e['duration_seconds'] for e in self.stop_events) / total) if total else 0.0
        counts = {'red_light': 0, 'queue_induced': 0, 'intersection': 0, 'unknown': 0}
        for e in self.stop_events:
            r = e.get('reason', 'unknown')
            if r not in counts:
                counts[r] = 0
            counts[r] += 1

        dominant = max(counts.items(), key=lambda x: x[1])[0] if counts else 'unknown'

        report = {
            'total_stop_count': total,
            'average_stop_duration_seconds': avg_duration,
            'counts_by_reason': counts,
            'stops': self.stop_events,
            'dominant_cause': dominant,
        }
        return report

