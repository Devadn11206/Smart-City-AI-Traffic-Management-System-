from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple
import logging
import time
import os

logger = logging.getLogger(__name__)

try:
    import traci
except Exception:  # pragma: no cover
    traci = None


def _has_traci_gui() -> bool:
    return traci is not None and hasattr(traci, 'gui') and callable(getattr(traci.gui, 'getIDList', None))

from .clearing_monitor import get_monitor


class TrafficClearingSystem:
    def __init__(self, enable_permissive_ev: bool = True, permissive_lane_change_mode: int = 511, use_gui: bool = False):
        self.reserved_edges: Dict[str, float] = {}
        self.reserved_routes: Dict[str, List[str]] = {}
        self._reroute_cooldowns: Dict[str, float] = {}
        self._clearing_cooldowns: Dict[str, float] = {}
        self.enable_permissive_ev = bool(enable_permissive_ev)
        self.permissive_lane_change_mode = int(permissive_lane_change_mode)
        self.use_gui = bool(use_gui)
        self._screenshot_warning_emitted = False
        # pending lane-change requests: vehicle_id -> {start_time, original_lane, target_index, edge_id, original_mode, reroute_router, destination}
        self._pending_lane_changes: Dict[str, Dict[str, object]] = {}
        # recorded failures to be written to JSON report
        self._lane_change_failures: List[Dict[str, object]] = []
        self._lane_change_config_written = False
        self._active_reroute_router = None
        # track yielding vehicles for visual feedback
        self._yielding_vehicles: Dict[str, float] = {}  # vehicle_id -> timestamp when started yielding
        self._yielded_count: int = 0
        self._speed_cache: Dict[str, float] = {}
        self._occupancy_cache: Dict[str, float] = {}
        self._route_cache: Dict[str, List[str]] = {}
        self._vehicle_list: List[str] = []

    def reserve_route(self, vehicle_id: str, route_edges: Iterable[str], hold_steps: int = 60) -> None:
        route_edges = list(dict.fromkeys(route_edges))
        self.reserved_routes[vehicle_id] = set(route_edges)
        for edge_id in route_edges:
            self.reserved_edges[edge_id] = int(hold_steps)
        logger.debug("Reserved route for %s: %s (hold_steps=%s)", vehicle_id, route_edges, hold_steps)

    def get_yielding_report(self) -> Dict[str, object]:
        """Get a report of visual yielding events."""
        return {
            'total_vehicles_that_yielded': self._yielded_count,
            'currently_yielding_vehicles': len(self._yielding_vehicles),
            'yielding_vehicle_ids': list(self._yielding_vehicles.keys()),
            'average_yield_duration_seconds': self._calculate_average_yield_duration() if self._yielding_vehicles else 0.0,
        }
    
    def _calculate_average_yield_duration(self) -> float:
        """Calculate average duration vehicles have been yielding."""
        if not self._yielding_vehicles:
            return 0.0
        now = time.time()
        durations = [now - start_time for start_time in self._yielding_vehicles.values()]
        return sum(durations) / len(durations) if durations else 0.0
    
    def step(self) -> None:
        """Update yielding vehicles and clean up expired ones."""
        # Clean up old yielding vehicles (those that have yielded for >60 seconds)
        now = time.time()
        self._speed_cache.clear()
        self._occupancy_cache.clear()
        self._route_cache.clear()
        self._vehicle_list.clear()
        for vehicle_id in list(self._yielding_vehicles.keys()):
            if now - self._yielding_vehicles[vehicle_id] > 60.0:
                self._yielding_vehicles.pop(vehicle_id, None)
        for edge_id in list(self.reserved_edges.keys()):
            self.reserved_edges[edge_id] -= 1
            if self.reserved_edges[edge_id] <= 0:
                self.reserved_edges.pop(edge_id, None)

        if not self.reserved_edges:
            self._active_reroute_router = None

        for vehicle_id, route_edges in list(self.reserved_routes.items()):
            if not any(edge in self.reserved_edges for edge in route_edges):
                self.reserved_routes.pop(vehicle_id, None)
        if not self.reserved_edges:
            self._reroute_cooldowns.clear()
            self._clearing_cooldowns.clear()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Reserved edges now: %s", self.reserved_edges)

        if traci is not None and self._active_reroute_router is not None and self.reserved_edges:
            try:
                reserved_edges = set(self.reserved_edges.keys())
                for vehicle_id in traci.vehicle.getIDList():
                    if self._is_emergency_vehicle(vehicle_id) or vehicle_id in self._pending_lane_changes:
                        continue
                    try:
                        route = list(traci.vehicle.getRoute(vehicle_id))
                    except Exception:
                        continue
                    if self._should_reroute_approaching_vehicle(route, reserved_edges):
                        destination = route[-1] if route else None
                        if destination is None:
                            continue
                        try:
                            self._active_reroute_router.reroute_vehicle(vehicle_id, destination, avoid_edges=list(reserved_edges), ants=3, max_steps=20)
                            get_monitor().record_reroute()
                        except Exception:
                            continue
            except Exception:
                pass

        if traci is not None and self._pending_lane_changes:
            current_vehicles = set(traci.vehicle.getIDList())
            for vehicle_id in list(self._pending_lane_changes.keys()):
                if vehicle_id not in current_vehicles:
                    self._pending_lane_changes.pop(vehicle_id, None)
                    continue
                pending = self._pending_lane_changes.get(vehicle_id) or {}
                try:
                    current_lane = traci.vehicle.getLaneID(vehicle_id)
                except Exception:
                    current_lane = None

                start_time = float(pending.get('start_time', 0.0))
                original_lane = pending.get('original_lane')
                target_index = int(pending.get('target_index', -1))
                edge_id = pending.get('edge_id')
                original_mode = pending.get('original_mode')
                reroute_router = pending.get('reroute_router')
                destination = pending.get('destination')

                elapsed_sec = time.time() - start_time
                logger.debug(f"AUDIT: verifying_pending_lane_change vehicle={vehicle_id} original_lane={original_lane} current_lane={current_lane} elapsed_sec={elapsed_sec:.2f}")

                # success if lane changed
                if current_lane is not None and original_lane is not None and current_lane != original_lane:
                    logger.debug(f"AUDIT: lane_change_success vehicle={vehicle_id} original_lane={original_lane} new_lane={current_lane}")
                    try:
                        get_monitor().record_lane_change_success(vehicle_id)
                    except Exception:
                        pass
                    try:
                        get_monitor().record_vehicle_moved_aside(vehicle_id)
                    except Exception:
                        pass
                    # screenshot for validation
                    try:
                        if self.use_gui and _has_traci_gui():
                            views = traci.gui.getIDList()
                            if views:
                                out_dir = os.path.join('results', 'emergency_metrics', 'screenshots')
                                os.makedirs(out_dir, exist_ok=True)
                                filename = os.path.join(out_dir, f"cleared_{int(time.time())}_{edge_id}_{vehicle_id}.png")
                                traci.gui.screenshot(views[0], filename)
                        elif not self.use_gui and not self._screenshot_warning_emitted:
                            logger.warning('GUI disabled: skipping traffic-clearing screenshot capture')
                            self._screenshot_warning_emitted = True
                    except Exception:
                        pass
                    # restore lane change mode if we changed it
                    if original_mode is not None:
                        try:
                            traci.vehicle.setLaneChangeMode(vehicle_id, int(original_mode))
                        except Exception:
                            pass
                    self._pending_lane_changes.pop(vehicle_id, None)
                    continue

                # timeout: consider as failure and fallback (3s)
                if time.time() - start_time > 3.0:
                    logger.debug(f"AUDIT: lane_change_timeout vehicle={vehicle_id} elapsed_sec={elapsed_sec:.2f}")

                    failure = {
                        'vehicle_id': vehicle_id,
                        'edge': edge_id,
                        'current_lane': current_lane,
                        'original_lane': original_lane,
                        'target_index': target_index,
                        'time': time.time(),
                        'reason': 'timeout_waiting_for_lane_change',
                    }
                    try:
                        leader = traci.vehicle.getLeader(vehicle_id)
                    except Exception:
                        leader = None
                    try:
                        follower = traci.vehicle.getFollower(vehicle_id)
                    except Exception:
                        follower = None
                    failure['leader'] = {
                        'vehicle_id': leader[0] if leader else None,
                        'distance': float(leader[1]) if leader and leader[1] is not None else None,
                    } if leader else None
                    failure['follower'] = {
                        'vehicle_id': follower[0] if follower else None,
                        'distance': float(follower[1]) if follower and follower[1] is not None else None,
                    } if follower else None
                    # check adjacent availability
                    try:
                        lane_count = int(traci.edge.getLaneNumber(edge_id)) if edge_id else 0
                        allowed = []
                        for li in range(lane_count):
                            try:
                                lane_id = f"{edge_id}_{li}"
                                ids = list(traci.lane.getLastStepVehicleIDs(lane_id))
                                allowed.append({
                                    'lane_id': lane_id,
                                    'allowed': traci.lane.getAllowed(lane_id),
                                    'vehicle_count': len(ids),
                                    'vehicles': ids,
                                })
                            except Exception:
                                continue
                        failure['lane_permissions'] = allowed
                    except Exception:
                        failure['lane_permissions'] = []

                    self._lane_change_failures.append(failure)

                    # attempt reroute fallback if available
                    if reroute_router is not None and destination is not None:
                        try:
                            reroute_router.reroute_vehicle(vehicle_id, destination, avoid_edges=list(self.reserved_edges.keys()), ants=3, max_steps=20)
                            try:
                                get_monitor().record_reroute()
                            except Exception:
                                pass
                        except Exception:
                            logger.exception('Fallback reroute failed for %s', vehicle_id)

                    # try slowing as last resort (but NOT emergency vehicles)
                    if self._should_slow_vehicle(vehicle_id):
                        try:
                            cur_speed = float(traci.vehicle.getSpeed(vehicle_id))
                            traci.vehicle.setSpeed(vehicle_id, max(0.5, cur_speed * 0.35))
                            try:
                                get_monitor().record_vehicle_moved_aside(vehicle_id)
                            except Exception:
                                pass
                        except Exception:
                            logger.exception('Failed to slow vehicle %s during fallback', vehicle_id)

                    # restore mode and drop pending
                    if original_mode is not None:
                        try:
                            traci.vehicle.setLaneChangeMode(vehicle_id, int(original_mode))
                        except Exception:
                            pass
                    self._pending_lane_changes.pop(vehicle_id, None)

            # write failure report if any failures collected
            if self._lane_change_failures:
                try:
                    out_path = os.path.join('results', 'emergency_metrics', 'lane_change_failure_report.json')
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    import json

                    with open(out_path, 'w', encoding='utf-8') as fh:
                        json.dump(self._lane_change_failures, fh, indent=2)
                except Exception:
                    logger.exception('Failed to write lane change failure report')

            if (self._lane_change_failures or self._pending_lane_changes) and not self._lane_change_config_written:
                self._write_lane_change_configuration_report()
                if not self._lane_change_failures:
                    try:
                        out_path = os.path.join('results', 'emergency_metrics', 'lane_change_failure_report.json')
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        import json

                        with open(out_path, 'w', encoding='utf-8') as fh:
                            json.dump([], fh, indent=2)
                    except Exception:
                        logger.exception('Failed to write empty lane change failure report')

    def is_reserved(self, edge_id: str) -> bool:
        return edge_id in self.reserved_edges

    def _is_emergency_vehicle(self, vehicle_id: str) -> bool:
        try:
            vehicle_type = traci.vehicle.getTypeID(vehicle_id).lower()
        except Exception:
            vehicle_type = vehicle_id.lower()
        return any(token in vehicle_type for token in ('ambulance', 'fire', 'police', 'rescue', 'emergency'))

    def _should_slow_vehicle(self, vehicle_id: str) -> bool:
        """Check if vehicle should be slowed. Return False for emergency vehicles."""
        # NEVER slow emergency vehicles - they have priority
        if self._is_emergency_vehicle(vehicle_id):
            return False
        return True

    def _apply_visual_yielding(self, vehicle_id: str, reduction_factor: float = 0.4) -> None:
        """Apply immediate speed reduction to encourage visible yielding.
        
        Args:
            vehicle_id: Vehicle to slow
            reduction_factor: Reduction factor (0.4 = 60% speed reduction)
        """
        # NEVER yield emergency vehicles
        if self._is_emergency_vehicle(vehicle_id):
            return
            
        if traci is None:
            return
        try:
            current_speed = float(traci.vehicle.getSpeed(vehicle_id))
            if current_speed > 1.0:  # Only slow if moving
                target_speed = max(0.5, current_speed * reduction_factor)
                traci.vehicle.setSpeed(vehicle_id, target_speed)
                self._yielding_vehicles[vehicle_id] = time.time()
                self._yielded_count += 1
                logger.info(f"🚗 VISUAL YIELD: {vehicle_id} reduced speed {current_speed:.1f} m/s → {target_speed:.1f} m/s")
        except Exception:
            pass

    def _get_preferred_lane_candidates(self, edge_id: str, current_lane_index: int) -> List[int]:
        """Get preferred adjacent lanes, favoring lanes that provide maximum clearance."""
        if traci is None:
            return []
        try:
            lane_count = int(traci.edge.getLaneNumber(edge_id))
        except Exception:
            return []
        
        candidates = []
        # Prefer outer lanes (higher indices for left side in USA, lower for right side)
        # Try both directions alternately for best coverage
        for offset in range(1, lane_count):
            # Try left first
            left_lane = current_lane_index + offset
            if 0 <= left_lane < lane_count and left_lane not in candidates:
                candidates.append(left_lane)
            # Then try right
            right_lane = current_lane_index - offset
            if 0 <= right_lane < lane_count and right_lane not in candidates:
                candidates.append(right_lane)
        
        return candidates[:3]  # Return top 3 preferred lanes

    def _attempt_courtesy_yielding_wave(
        self, 
        vehicle_id: str, 
        edge_id: str, 
        lane_index: int, 
        original_mode: Optional[int]
    ) -> bool:
        """Attempt multiple yielding strategies in sequence: speed reduction → lane change → reroute.
        
        Returns True if any yielding action succeeded.
        """
        if traci is None:
            return False
        
        # Step 1: Immediate visual speed reduction
        self._apply_visual_yielding(vehicle_id, reduction_factor=0.4)
        
        # Step 2: Attempt multiple lane changes with short pauses between
        candidate_lanes = self._get_preferred_lane_candidates(edge_id, lane_index)
        for attempt_num, target_lane in enumerate(candidate_lanes):
            try:
                time.sleep(0.01)  # Small pause between attempts
                logger.debug(f"Courtesy wave attempt {attempt_num + 1}: {vehicle_id} → lane {target_lane}")
                traci.vehicle.changeLane(vehicle_id, target_lane, 20)  # Shorter duration = more aggressive
                
                # Track as pending
                self._pending_lane_changes[vehicle_id] = {
                    'start_time': time.time(),
                    'original_lane': traci.vehicle.getLaneID(vehicle_id),
                    'target_index': target_lane,
                    'edge_id': edge_id,
                    'original_mode': original_mode,
                    'reroute_router': None,
                    'destination': None,
                }
                logger.info(f"✓ Courtesy wave: {vehicle_id} yielding to lane {target_lane}")
                return True
            except Exception as e:
                logger.debug(f"Courtesy wave attempt {attempt_num + 1} failed: {e}")
                continue
        
        return False

    def _adjacent_lane_candidates(self, edge_id: str, lane_index: int) -> List[int]:
        try:
            lane_count = int(traci.edge.getLaneNumber(edge_id))
        except Exception:
            lane_count = 0

        candidates: List[int] = []
        if lane_index - 1 >= 0:
            candidates.append(lane_index - 1)
        if lane_index + 1 < lane_count:
            candidates.append(lane_index + 1)

        for extra in range(2, max(2, lane_count)):
            left = lane_index - extra
            right = lane_index + extra
            if left >= 0 and left not in candidates:
                candidates.append(left)
            if right < lane_count and right not in candidates:
                candidates.append(right)
                
        # Filter by occupancy
        filtered_candidates = []
        for cand in candidates:
            try:
                lane_id = f"{edge_id}_{cand}"
                occupancy = self._get_occupancy(lane_id)
                if occupancy < 0.8:
                    filtered_candidates.append(cand)
            except Exception:
                filtered_candidates.append(cand)
        return filtered_candidates

    def _vehicle_ids_on_edges(self, edge_ids: Iterable[str]) -> List[str]:
        vehicle_ids: List[str] = []
        for edge_id in edge_ids:
            try:
                for vehicle_id in traci.edge.getLastStepVehicleIDs(edge_id):
                    if vehicle_id not in vehicle_ids:
                        vehicle_ids.append(vehicle_id)
            except Exception:
                continue
        return vehicle_ids

    def _find_tls_for_edge(self, edge_id: str) -> List[str]:
        """Return list of TLS ids that control lanes on the given edge."""
        if traci is None:
            return []
        tls_ids: List[str] = []
        try:
            for ts_id in traci.trafficlight.getIDList():
                try:
                    lanes = traci.trafficlight.getControlledLanes(ts_id)
                except Exception:
                    continue
                for lane in lanes:
                    try:
                        if traci.lane.getEdgeID(lane) == edge_id:
                            tls_ids.append(ts_id)
                            break
                    except Exception:
                        continue
        except Exception:
            logger.exception('Failed to enumerate TLS for edge %s', edge_id)
        return tls_ids

    def _edge_length(self, edge_id: str) -> float:
        if traci is None:
            return 0.0
        try:
            return float(traci.edge.getLength(edge_id))
        except Exception:
            return 0.0

    def _edges_within_distance(self, route_edges: Iterable[str], target_distance: float = 200.0) -> List[str]:
        selected: List[str] = []
        total_distance = 0.0
        for edge_id in route_edges:
            if edge_id in selected:
                continue
            selected.append(edge_id)
            total_distance += self._edge_length(edge_id)
            if total_distance >= float(target_distance):
                break
        return selected

    def _sort_blocking_vehicles(self, emergency_vehicle_id: str, edges_to_clear: Iterable[str], route_edges: Iterable[str]) -> List[Tuple[str, str]]:
        emergency_edge = None
        emergency_lane = None
        emergency_position = 0.0
        try:
            emergency_edge = traci.vehicle.getRoadID(emergency_vehicle_id)
            emergency_lane = traci.vehicle.getLaneID(emergency_vehicle_id)
            emergency_position = float(traci.vehicle.getLanePosition(emergency_vehicle_id))
        except Exception:
            pass

        route_index = {edge: idx for idx, edge in enumerate(route_edges)}
        candidates = []
        for edge_id in edges_to_clear:
            try:
                vehicle_ids = list(traci.edge.getLastStepVehicleIDs(edge_id))
            except Exception:
                vehicle_ids = []
            for vehicle_id in vehicle_ids:
                if self._is_emergency_vehicle(vehicle_id):
                    continue
                try:
                    lane_id = traci.vehicle.getLaneID(vehicle_id)
                    position = float(traci.vehicle.getLanePosition(vehicle_id))
                except Exception:
                    lane_id = None
                    position = 0.0

                if emergency_edge == edge_id and emergency_lane is not None and lane_id == emergency_lane and position <= emergency_position:
                    continue

                same_edge = 0 if edge_id == emergency_edge else 1
                same_lane = 0 if lane_id == emergency_lane else 1
                edge_rank = route_index.get(edge_id, len(route_edges))
                distance_score = position if edge_id == emergency_edge else float(edge_rank * 1000.0 + position)
                candidates.append((same_edge, same_lane, distance_score, vehicle_id, edge_id))

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return [(vehicle_id, edge_id) for _, _, _, vehicle_id, edge_id in candidates]

    def _should_reroute_approaching_vehicle(self, route_edges: List[str], reserved_edges: Set[str]) -> bool:
        if not route_edges or not reserved_edges:
            return False
        for edge_id in route_edges[1:3]:
            if edge_id in reserved_edges:
                return True
        return False

    def _reroute_approaching_vehicles(self, reroute_router, avoid_edges: List[str]) -> List[str]:
        if traci is None or reroute_router is None or not avoid_edges:
            return []
        rerouted: List[str] = []
        reserved_set = set(avoid_edges)
        try:
            for vehicle_id in self._get_vehicle_list():
                if self._is_emergency_vehicle(vehicle_id) or vehicle_id in self._pending_lane_changes or vehicle_id in self._reroute_cooldowns:
                    continue
                try:
                    route = self._get_route(vehicle_id)
                except Exception:
                    continue
                if not route:
                    continue
                if self._should_reroute_approaching_vehicle(route, reserved_set):
                    try:
                        destination = route[-1] if route else None
                        if destination is None:
                            continue
                        current_time = float(traci.simulation.getTime())
                        if current_time - self._reroute_cooldowns.get(vehicle_id, -float('inf')) < 50:
                            continue
                        reroute_router.reroute_vehicle(vehicle_id, destination, avoid_edges=avoid_edges, ants=3, max_steps=20)
                        self._reroute_cooldowns[vehicle_id] = current_time
                        rerouted.append(vehicle_id)
                    except Exception:
                        continue
        except Exception:
            pass
        return rerouted

    def _write_lane_change_configuration_report(self) -> None:
        if traci is None:
            return
        try:
            cfg = {}
            vehicles = []
            try:
                vehicles = list(traci.vehicle.getIDList())
            except Exception:
                vehicles = []
            for vid in vehicles:
                try:
                    vinfo = {
                        'type': traci.vehicle.getTypeID(vid),
                        'laneChangeMode': traci.vehicle.getLaneChangeMode(vid),
                        'speedMode': traci.vehicle.getSpeedMode(vid),
                        'lane': traci.vehicle.getLaneID(vid),
                        'road': traci.vehicle.getRoadID(vid),
                    }
                    vehicles_allowed = []
                    try:
                        edge_id = vinfo.get('road')
                        if edge_id and edge_id != '':
                            lane_count = int(traci.edge.getLaneNumber(edge_id))
                            for li in range(lane_count):
                                lane_id = f"{edge_id}_{li}"
                                try:
                                    vehicles_allowed.append({
                                        'lane_id': lane_id,
                                        'allowed': traci.lane.getAllowed(lane_id),
                                    })
                                except Exception:
                                    continue
                    except Exception:
                        vehicles_allowed = []
                    vinfo['lane_permissions_on_edge'] = vehicles_allowed
                    cfg[vid] = vinfo
                except Exception:
                    continue
            config_path = os.path.join('results', 'emergency_metrics', 'lane_change_configuration_report.json')
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            import json
            with open(config_path, 'w', encoding='utf-8') as fh:
                json.dump(cfg, fh, indent=2)
            self._lane_change_config_written = True
        except Exception:
            logger.exception('Failed to write lane change configuration report')

    def _force_tls_for_edge(self, edge_id: str, route_edges: Optional[Iterable[str]] = None, hold_seconds: int = 5) -> None:
        """Aggressively set TLS phases for TLSs that control the given edge to favor movement on that edge.

        This is a last-resort preemption used when vehicles remain blocking the corridor at a signal.
        """
        if traci is None:
            return
        try:
            tls_list = self._find_tls_for_edge(edge_id)
            for ts_id in tls_list:
                try:
                    # reuse green corridor selection logic if available via trafficlight API
                    # compute best phase for this TLS that serves the provided route_edges (if any)
                    if route_edges is not None:
                        try:
                            # safe best-phase selection using trafficlight program logics
                            logics = traci.trafficlight.getAllProgramLogics(ts_id)
                            if logics and len(logics) > 0:
                                phases = logics[0].getPhases()
                                # fallback to phase 0
                                best_phase = 0
                                best_score = float('-inf')
                                controlled_links = traci.trafficlight.getControlledLinks(ts_id)
                                route_set = set(route_edges or [])
                                for phase_idx, phase in enumerate(phases):
                                    score = 0.0
                                    for signal_idx, links in enumerate(controlled_links):
                                        if signal_idx >= len(phase.state):
                                            continue
                                        if phase.state[signal_idx] not in ('G', 'g'):
                                            continue
                                        for link in links:
                                            if not link:
                                                continue
                                            try:
                                                incoming_edge = traci.lane.getEdgeID(link[0])
                                            except Exception:
                                                incoming_edge = None
                                            if incoming_edge in route_set:
                                                score += 2.0
                                            else:
                                                score += 0.2
                                    if score > best_score:
                                        best_score = score
                                        best_phase = phase_idx
                                traci.trafficlight.setPhase(ts_id, max(0, min(best_phase, len(phases) - 1)))
                            else:
                                traci.trafficlight.setPhase(ts_id, 0)
                        except Exception:
                            traci.trafficlight.setPhase(ts_id, 0)
                    else:
                        # no route info: set first phase (best-effort)
                        traci.trafficlight.setPhase(ts_id, 0)
                    # attempt to keep the phase for a short duration
                    try:
                        traci.trafficlight.setPhaseDuration(ts_id, max(1, int(hold_seconds)))
                    except Exception:
                        pass
                    logger.info('Forcing TLS %s to favor edge %s (hold=%ss)', ts_id, edge_id, hold_seconds)
                except Exception:
                    logger.exception('Failed to force TLS %s for edge %s', ts_id, edge_id)
        except Exception:
            logger.exception('Failed to force TLS for edge %s', edge_id)

    def _get_speed(self, vehicle_id: str) -> float:
        if traci is None: return 0.0
        if vehicle_id in self._speed_cache:
            return self._speed_cache[vehicle_id]
        try:
            s = float(traci.vehicle.getSpeed(vehicle_id))
            self._speed_cache[vehicle_id] = s
            return s
        except Exception:
            self._speed_cache[vehicle_id] = 0.0
            return 0.0

    def _get_occupancy(self, lane_id: str) -> float:
        if traci is None: return 0.0
        if lane_id in self._occupancy_cache:
            return self._occupancy_cache[lane_id]
        try:
            o = float(traci.lane.getLastStepOccupancy(lane_id))
            self._occupancy_cache[lane_id] = o
            return o
        except Exception:
            self._occupancy_cache[lane_id] = 1.0
            return 1.0

    def _get_route(self, vehicle_id: str) -> List[str]:
        if traci is None: return []
        if vehicle_id in self._route_cache:
            return self._route_cache[vehicle_id]
        try:
            r = list(traci.vehicle.getRoute(vehicle_id))
            self._route_cache[vehicle_id] = r
            return r
        except Exception:
            return []

    def _get_vehicle_list(self) -> List[str]:
        if traci is None: return []
        if not self._vehicle_list:
            try:
                self._vehicle_list = list(traci.vehicle.getIDList())
            except Exception:
                pass
        return self._vehicle_list

    def _detect_gridlock(self, blocking_vehicles: List[Tuple[str, str]]) -> bool:
        if traci is None or not blocking_vehicles:
            return False
        if len(blocking_vehicles) < 20:
            return False
        total_speed = 0.0
        sample_size = min(15, len(blocking_vehicles))
        for vid, _ in blocking_vehicles[:sample_size]:
            total_speed += self._get_speed(vid)
        avg_speed = total_speed / sample_size
        return avg_speed < 2.0

    def clear_around_route(
        self,
        emergency_vehicle_id: str,
        route_edges: Iterable[str],
        reroute_router=None,
        destination: Optional[str] = None,
        clear_all_corridor: bool = True,
        corridor_edges: Optional[Iterable[str]] = None,
    ) -> Dict[str, int]:
        """
        Clear traffic around emergency route by forcing vehicles aside, rerouting blockers, and slowing only as fallback.
        """
        start = time.time()
        cleared = {
            'slowed': 0,
            'rerouted': 0,
            'slowed_vehicle_ids': [],
            'rerouted_vehicle_ids': [],
            'vehicles_remaining_in_corridor': [],
        }
        if traci is None:
            return cleared

        route_edges = list(dict.fromkeys(route_edges))
        if corridor_edges is not None:
            edges_to_clear = list(dict.fromkeys(corridor_edges))
        elif clear_all_corridor:
            edges_to_clear = list(route_edges)
        else:
            edges_to_clear = self._edges_within_distance(route_edges, target_distance=75.0)

        self.reserve_route(emergency_vehicle_id, edges_to_clear)
        self._active_reroute_router = reroute_router
        moved_vehicle_ids: Set[str] = set()
        ev_road = None
        try:
            ev_road = traci.vehicle.getRoadID(emergency_vehicle_id)
        except Exception:
            ev_road = None

        blocking_vehicles = self._sort_blocking_vehicles(emergency_vehicle_id, edges_to_clear, route_edges)
        logger.debug(f"AUDIT: blocking_vehicles_found={len(blocking_vehicles)} vehicles={[v[0] for v in blocking_vehicles[:5]]}")
        
        is_gridlocked = self._detect_gridlock(blocking_vehicles)
        current_time = float(traci.simulation.getTime()) if traci else 0.0

        proactively_rerouted = []
        if reroute_router is not None:
            proactively_rerouted = self._reroute_approaching_vehicles(reroute_router, list(self.reserved_edges.keys()))
            for vehicle_id in proactively_rerouted:
                if vehicle_id not in cleared['rerouted_vehicle_ids']:
                    cleared['rerouted'] += 1
                    cleared['rerouted_vehicle_ids'].append(vehicle_id)
                    moved_vehicle_ids.add(vehicle_id)
                    try:
                        get_monitor().record_reroute()
                    except Exception:
                        pass

        for vehicle_id, edge_id in blocking_vehicles:
            logger.debug(f"AUDIT: blocker_loop_processing vehicle={vehicle_id} edge={edge_id}")
            if vehicle_id in moved_vehicle_ids:
                continue
            if self._is_emergency_vehicle(vehicle_id):
                continue
                
            # Cooldown check
            if current_time - self._clearing_cooldowns.get(vehicle_id, -float('inf')) < 20:
                continue

            try:
                lane_id = traci.vehicle.getLaneID(vehicle_id)
            except Exception:
                lane_id = ''

            current_speed = self._get_speed(vehicle_id)

            if not lane_id:
                continue

            try:
                lane_index = int(lane_id.split('_')[-1]) if '_' in lane_id else 0
            except Exception:
                lane_index = 0

            try:
                get_monitor().record_lane_change_attempt(vehicle_id)
            except Exception:
                pass

            try:
                original_mode = traci.vehicle.getLaneChangeMode(vehicle_id)
            except Exception:
                original_mode = None

            # === ENHANCED CLEARING HIERARCHY ===
            moved = False

            # Phase 3 Priority A: changeLane()
            if not is_gridlocked:
                if original_mode is not None:
                    try:
                        traci.vehicle.setLaneChangeMode(vehicle_id, int(self.permissive_lane_change_mode))
                    except Exception:
                        pass

                candidate_lanes = self._adjacent_lane_candidates(edge_id, lane_index)
                for target_index in candidate_lanes:
                    try:
                        traci.vehicle.changeLane(vehicle_id, target_index, 50)
                        self._pending_lane_changes[vehicle_id] = {
                            'start_time': time.time(),
                            'original_lane': lane_id,
                            'target_index': int(target_index),
                            'edge_id': edge_id,
                            'original_mode': original_mode,
                            'reroute_router': reroute_router,
                            'destination': None,
                        }
                        moved = True
                        moved_vehicle_ids.add(vehicle_id)
                        self._clearing_cooldowns[vehicle_id] = current_time
                        logger.info(f"[CLEARING] changeLane() triggered for {vehicle_id} to lane {target_index} at step {current_time}")
                        break
                    except Exception:
                        continue

            # Phase 3 Priority B: slowDown()
            if not moved and not is_gridlocked and not self._is_emergency_vehicle(vehicle_id):
                try:
                    new_speed = max(0.5, current_speed * 0.35)
                    traci.vehicle.setSpeed(vehicle_id, new_speed)
                    cleared['slowed'] += 1
                    if vehicle_id not in cleared['slowed_vehicle_ids']:
                        cleared['slowed_vehicle_ids'].append(vehicle_id)
                    moved = True
                    moved_vehicle_ids.add(vehicle_id)
                    self._clearing_cooldowns[vehicle_id] = current_time
                    try:
                        get_monitor().record_vehicle_moved_aside(vehicle_id)
                    except Exception:
                        pass
                    logger.info(f"[CLEARING] slowDown() triggered for {vehicle_id} to {new_speed:.1f}m/s at step {current_time}")
                except Exception:
                    pass

            # Phase 3 Priority C: ACO reroute
            if not moved and reroute_router is not None and destination is not None:
                try:
                    # Check reroute cooldown again just in case (though already checked approaching)
                    if current_time - self._reroute_cooldowns.get(vehicle_id, -float('inf')) >= 50:
                        try:
                            vehicle_route = self._get_route(vehicle_id)
                            vehicle_destination = vehicle_route[-1] if vehicle_route else destination
                        except Exception:
                            vehicle_destination = destination
                        reroute_router.reroute_vehicle(vehicle_id, vehicle_destination, avoid_edges=list(self.reserved_edges.keys()), ants=3, max_steps=20)
                        cleared['rerouted'] += 1
                        if vehicle_id not in cleared['rerouted_vehicle_ids']:
                            cleared['rerouted_vehicle_ids'].append(vehicle_id)
                        moved = True
                        moved_vehicle_ids.add(vehicle_id)
                        self._reroute_cooldowns[vehicle_id] = current_time
                        try:
                            get_monitor().record_reroute()
                        except Exception:
                            pass
                        logger.info(f"[CLEARING] ACO reroute triggered for {vehicle_id} at step {current_time}")
                except Exception:
                    pass

            if original_mode is not None:
                try:
                    traci.vehicle.setLaneChangeMode(vehicle_id, int(original_mode))
                except Exception:
                    pass

            try:
                ev_road = traci.vehicle.getRoadID(emergency_vehicle_id)
            except Exception:
                ev_road = None

            try:
                if ev_road == edge_id:
                    original_mode = None
                    emergency_original_lane = None
                    if self.enable_permissive_ev:
                        try:
                            original_mode = traci.vehicle.getLaneChangeMode(emergency_vehicle_id)
                        except Exception:
                            original_mode = None
                        try:
                            traci.vehicle.setLaneChangeMode(emergency_vehicle_id, self.permissive_lane_change_mode)
                        except Exception:
                            pass

                    try:
                        emergency_original_lane = traci.vehicle.getLaneID(emergency_vehicle_id)
                    except Exception:
                        emergency_original_lane = None

                    try:
                        get_monitor().record_lane_change_attempt(emergency_vehicle_id)
                    except Exception:
                        pass

                    emergency_lane_changed = False
                    try:
                        traci.vehicle.changeLane(emergency_vehicle_id, 0, 10)
                        try:
                            self._pending_lane_changes[emergency_vehicle_id] = {
                                'start_time': time.time(),
                                'original_lane': emergency_original_lane,
                                'target_index': 0,
                                'edge_id': ev_road,
                                'original_mode': original_mode,
                                'reroute_router': None,
                                'destination': None,
                            }
                            emergency_lane_changed = True
                        except Exception:
                            pass
                    except Exception:
                        pass

                    if self.enable_permissive_ev and original_mode is not None and not emergency_lane_changed:
                        try:
                            traci.vehicle.setLaneChangeMode(emergency_vehicle_id, int(original_mode))
                        except Exception:
                            pass
            except Exception:
                pass

        try:
            for vehicle_id in self._vehicle_ids_on_edges(edges_to_clear):
                if vehicle_id not in moved_vehicle_ids and not self._is_emergency_vehicle(vehicle_id):
                    if vehicle_id not in cleared['vehicles_remaining_in_corridor']:
                        cleared['vehicles_remaining_in_corridor'].append(vehicle_id)
        except Exception:
            cleared['vehicles_remaining_in_corridor'] = []

        # If blockers remain in the corridor, attempt aggressive TLS preemption
        try:
            if cleared.get('vehicles_remaining_in_corridor'):
                for edge_id in edges_to_clear:
                    try:
                        self._force_tls_for_edge(edge_id, route_edges=route_edges, hold_seconds=5)
                    except Exception:
                        continue
        except Exception:
            pass

        # If the emergency vehicle itself is stopped at an edge, force the TLS at that edge
        try:
            if traci is not None and ev_road:
                try:
                    ev_speed = self._get_speed(emergency_vehicle_id)
                except Exception:
                    ev_speed = None
                if ev_speed is not None and ev_speed <= 0.5:
                    try:
                        self._force_tls_for_edge(ev_road, route_edges=route_edges, hold_seconds=5)
                    except Exception:
                        pass
        except Exception:
            pass

        elapsed = time.time() - start
        try:
            get_monitor().record_clear_call(len(edges_to_clear), len(edges_to_clear), cleared['slowed'], cleared['rerouted'])
        except Exception:
            pass
        logger.info(
            "Cleared around route %s: cleared_edges=%d slowed=%s rerouted=%s remaining=%s elapsed=%.3fs",
            emergency_vehicle_id,
            len(edges_to_clear),
            cleared['slowed'],
            cleared['rerouted'],
            len(cleared['vehicles_remaining_in_corridor']),
            elapsed,
        )
        return cleared

    def relocate_vehicle(self, vehicle_id: str, speed_factor: float = 0.5) -> bool:
        # Never relocate emergency vehicles
        if self._is_emergency_vehicle(vehicle_id):
            return False
            
        if traci is None:
            return False
        try:
            current_speed = float(traci.vehicle.getSpeed(vehicle_id))
            traci.vehicle.setSpeed(vehicle_id, max(0.0, current_speed * float(speed_factor)))
            logger.debug("Relocated vehicle %s speed %s -> %s", vehicle_id, current_speed, max(0.0, current_speed * float(speed_factor)))
            return True
        except Exception:
            return False