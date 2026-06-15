import os
import sys
import time
import argparse
import csv
import json
import logging
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traci
from sumolib import net

from aco_routing.pheromone_manager import PheromoneManager
from aco_routing.aco_router import ACORouter
import logging
from aco_routing.route_optimizer import RouteOptimizer
from aco_routing.congestion_costs import normalize_predictions, DEFAULT_COST_WEIGHTS

from environment.sumo_environment import SumoEnvironment
from utils.reliability import extract_state_dict, load_torch_checkpoint, save_json
from utils.monitoring import start_monitoring

try:
    from integration.hybrid_smart_city_controller import HybridSmartCityController
except Exception:
    HybridSmartCityController = None

try:
    from emergency_system.traffic_clearing import TrafficClearingSystem
    from emergency_system.emergency_router import EmergencyRouter
    from emergency_system.multi_emergency_manager import MultiEmergencyManager
except Exception:
    TrafficClearingSystem = None
    EmergencyRouter = None
    MultiEmergencyManager = None



logger = logging.getLogger(__name__)

def build_adjacency_and_edgeinfo(netfile):
    """Build adjacency map and basic edge info using sumolib network."""
    adjacency = {}
    edge_info = {}
    try:
        netobj = net.readNet(netfile)
        for edge in netobj.getEdges():
            eid = edge.getID()
            # Skip junction edges (starting with ':')
            if eid.startswith(':'):
                continue
            adjacency.setdefault(eid, [])
            edge_info[eid] = {'length': edge.getLength()}
            # outgoing edges via to node (but skip junction edges)
            for out in edge.getToNode().getOutgoing():
                out_edge = out.getID()
                # Only include regular edges, skip junction edges
                if not out_edge.startswith(':'):
                    adjacency[eid].append(out_edge)
    except Exception:
        # fallback: use empty adjacency (will still run but can't reroute)
        adjacency = {}
        edge_info = {}
    return adjacency, edge_info


def select_emergency_phase(ts_id, route_edges):
    """Select the best phase index for a TLS based on route edges it serves.

    This approximates a route-aligned emergency phase by scoring each available phase
    against the controlled links whose incoming lanes belong to the emergency route.
    """
    try:
        logics = traci.trafficlight.getAllProgramLogics(ts_id)
        if not logics:
            return 0
        phases = logics[0].getPhases()
        controlled_links = traci.trafficlight.getControlledLinks(ts_id)
        route_edge_set = set(route_edges)

        best_phase = 0
        best_score = float('-inf')
        for phase_idx, phase in enumerate(phases):
            state = phase.state
            score = 0.0
            for signal_idx, links in enumerate(controlled_links):
                if signal_idx >= len(state):
                    continue
                signal = state[signal_idx]
                if signal not in ('G', 'g'):
                    continue
                for link in links:
                    if not link:
                        continue
                    in_lane = link[0]
                    try:
                        in_edge = traci.lane.getEdgeID(in_lane)
                    except Exception:
                        in_edge = None
                    if in_edge in route_edge_set:
                        score += 2.0
                    else:
                        score += 0.1
            if score > best_score:
                best_score = score
                best_phase = phase_idx
        return best_phase
    except Exception:
        return 0

def load_pinn(pinn_path, device=None):
    # returns a callable predict(edges) -> {edge_id: pred_value}
    try:
        logger.info("PINN requested path: %s", pinn_path)
        import torch
        from pinn_models.pinn_model import TrafficPINN

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        checkpoint = load_torch_checkpoint(pinn_path, map_location=device)

        pinn = TrafficPINN().to(device)
        pinn.load_state_dict(extract_state_dict(checkpoint))
        pinn.eval()

        logger.info("PINN loaded successfully from %s (device=%s)", pinn_path, device)
        if isinstance(checkpoint, dict):
            logger.info("PINN checkpoint keys: %s", list(checkpoint.keys()))

        edge_map = checkpoint.get('edge_map', {})
        t_max = checkpoint.get('t_max', 1.0)
        x_max = checkpoint.get('x_max', 1.0)

        def predict(edges, future_step=0, lookahead_steps=0):
            preds = {}
            import torch
            with torch.no_grad():
                for e in edges:
                    if e in edge_map:
                        norm_t = float(max(0, future_step + lookahead_steps)) / max(t_max, 1.0)
                        norm_x = edge_map[e] / max(x_max, 1.0)
                        t_tensor = torch.tensor([[norm_t]], dtype=torch.float32).to(device)
                        x_tensor = torch.tensor([[norm_x]], dtype=torch.float32).to(device)
                        rho_pred, v_pred = pinn(t_tensor, x_tensor)
                        # normalize density to [0,1] by dividing by target_rhomax
                        try:
                            score = float(rho_pred.item()) / float(pinn.target_rhomax)
                        except Exception:
                            score = float(rho_pred.item())
                        preds[e] = max(0.0, min(1.0, score))
                    else:
                        preds[e] = 0.0
            return preds

        return predict
    except Exception as e:
        logger.exception('Failed to load PINN checkpoint from %s', pinn_path)
        logger.error("PINN fallback predictor enabled due to error: %s", e)
        return lambda edges, **kwargs: {e: 0.0 for e in edges}



def update_episode_metrics(env):
    """Update the environment's rolling metrics directly from TraCI."""
    if getattr(env, 'controlled_ts', None) is None:
        return

    try:
        lanes = traci.trafficlight.getControlledLanes(env.controlled_ts)
    except Exception:
        return

    lanes = set(lanes)
    current_total_wait = sum(traci.lane.getWaitingTime(lane) for lane in lanes) if lanes else 0.0
    current_queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes) if lanes else 0.0
    speeds = [traci.lane.getLastStepMeanSpeed(lane) for lane in lanes] if lanes else []
    current_avg_speed = float(sum(speeds) / len(speeds)) if speeds else 0.0

    delta_wait = current_total_wait - getattr(env, '_last_total_wait', 0.0)
    if delta_wait < 0:
        delta_wait = current_total_wait

    env.cumulative_wait = getattr(env, 'cumulative_wait', 0.0) + float(delta_wait)
    env.cumulative_queue = getattr(env, 'cumulative_queue', 0.0) + float(current_queue)
    env.cumulative_speed = getattr(env, 'cumulative_speed', 0.0) + float(current_avg_speed)
    env.metric_samples = getattr(env, 'metric_samples', 0) + 1
    env._last_total_wait = float(current_total_wait)
    env._last_total_queue = float(current_queue)

    try:
        arrived = traci.simulation.getArrivedNumber()
        throughput = max(0, arrived - getattr(env, '_last_arrived', 0))
        env._last_arrived = arrived
        env.cumulative_throughput = getattr(env, 'cumulative_throughput', 0.0) + float(throughput)
    except Exception:
        pass


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def snapshot_metrics(env, sim_step, mode_name, out_dir, current_predictions=None):
    """Write a metrics checkpoint so long runs preserve intermediate state."""
    ensure_dir(out_dir)
    try:
        metrics = env.get_episode_metrics()
    except Exception:
        metrics = {}

    emergency_vehicle_count = 0
    active_corridor_tls = 0
    emergency_phase_count = 0
    if HybridSmartCityController is not None:
        try:
            emergency_vehicle_count = sum(
                1 for vid in traci.vehicle.getIDList()
                if traci.vehicle.getTypeID(vid).lower() in ('emergency', 'ambulance', 'police', 'firetruck', 'fire_truck')
            )
        except Exception:
            emergency_vehicle_count = 0
        try:
            active_corridor_tls = len(getattr(HybridSmartCityController, '_emergency_corridor', []) or [])
        except Exception:
            active_corridor_tls = 0
        try:
            emergency_phase_count = len(getattr(HybridSmartCityController, 'emergency_state', {}) or {})
        except Exception:
            emergency_phase_count = 0

    payload = {
        'step': int(sim_step),
        'mode': mode_name,
        'timestamp': time.time(),
        'emergency_vehicle_count': int(emergency_vehicle_count),
        'active_corridor_tls': int(active_corridor_tls),
        'emergency_phase_count': int(emergency_phase_count),
        'pinn_prediction_count': int(len(current_predictions or {})),
        'pinn_prediction_mean': float(np.mean(list((current_predictions or {}).values()))) if current_predictions else 0.0,
        'pinn_prediction_max': float(np.max(list((current_predictions or {}).values()))) if current_predictions else 0.0,
        **metrics,
    }

    csv_path = os.path.join(out_dir, f'{mode_name}_metrics_checkpoints.csv')
    json_path = os.path.join(out_dir, f'{mode_name}_metrics_latest.json')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(payload)

    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)

    return payload

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='sumo_network/config.sumocfg')
    parser.add_argument('--pinn', default=None)
    parser.add_argument('--duration', type=int, default=2000)
    parser.add_argument('--lookahead-steps', type=int, default=10, help='PINN lookahead steps for future congestion prediction')
    parser.add_argument('--reroute-interval', type=int, default=30, help='Vehicle reroute interval in simulation steps')
    parser.add_argument('--emergency-hold-steps', type=int, default=20, help='Duration to hold emergency phase plans')
    parser.add_argument('--decision-interval', type=int, default=5, help='Traffic-light decision interval in simulation steps')
    parser.add_argument('--pinn-interval', type=int, default=30, help='How often to refresh PINN predictions')
    parser.add_argument('--metrics-checkpoint-interval', type=int, default=250, help='Write periodic metrics checkpoints every N steps')
    parser.add_argument('--metrics-output-dir', default='results/aco_results/checkpoints', help='Directory for metrics checkpoint files')
    parser.add_argument('--run-name', default='integration', help='Label used for checkpoint filenames')
    parser.add_argument('--device', default=None, help="Device to use for PINN: 'cpu','cuda' or device id")
    parser.add_argument('--enable-aco', type=int, default=1, help='Enable ACO routing (1/0)')
    parser.add_argument('--enable-pinn', type=int, default=1, help='Enable PINN predictions (1/0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for SUMO evaluation')
    parser.add_argument('--gui', action='store_true', default=False, help='Launch SUMO-GUI instead of SUMO')
    
    parser.add_argument('--reroute-log', default='', help='Optional CSV path for ACO reroute logging')
    parser.add_argument('--aco-debug-log', default='', help='Optional CSV path for detailed ACO debug logging')
    args = parser.parse_args()

    # normalize flags
    args.enable_aco = bool(int(args.enable_aco))
    args.enable_pinn = bool(int(args.enable_pinn))

    ensure_dir(args.metrics_output_dir)

    if HybridSmartCityController is not None:
        try:
            HybridSmartCityController.decision_interval = max(1, int(args.decision_interval))
        except Exception:
            pass

    adjacency, edge_info = build_adjacency_and_edgeinfo('sumo_network/network.net.xml')
    pheromone = PheromoneManager(adjacency, initial_pheromone=1.0, evaporation_rate=0.03)
    print(f'INITIAL GRAPH STATE: adjacency_size={len(adjacency)}, pheromone_count={len(pheromone.pheromones)}')
    print('INITIAL ADJ_KEYS=', list(adjacency.keys())[:20])
    print('INITIAL PH_KEYS=', [f'{u}->{v}' for (u,v) in list(pheromone.pheromones.keys())[:20]])

    # heuristic uses edge_info and pinn predictions (updated per interval)
    # device selection for PINN/torch
    import torch
    if hasattr(args, 'device') and args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pinn_predictor = load_pinn(args.pinn, device=device) if args.pinn else (lambda es, **kwargs: {e: 0.0 for e in es})
    logger.info("PINN enabled=%s predictor=%s", bool(args.enable_pinn), 'callable' if args.pinn else 'disabled')

    def heuristic(edge_id):
        return edge_info.get(edge_id, {}).get('length', 1.0)

    aco = ACORouter(pheromone, lambda e: edge_info.get(e, {}).get('length', 1.0), alpha=1.0, beta=2.0, ants=30)
    optimizer = None
    if args.enable_aco:
        cost_weights = dict(DEFAULT_COST_WEIGHTS)
        cost_weights.update({
            'travel_time': 1.0,
            'queue_length': 1.25,
            'waiting_time': 1.35,
            'traffic_density': 0.9,
            'future_congestion': 2.5,
        })
        optimizer = RouteOptimizer(
            acorouter=aco,
            adjacency=adjacency,
            edge_info=edge_info,
            reroute_interval=args.reroute_interval,
            cost_weights=cost_weights,
            reroute_log_path=args.reroute_log or None,
            debug_log_path=args.aco_debug_log or None,
            protected_edge_multiplier=1000.0,
            reserved_edges_by_emergency={},
            aco_reserved_avoid_non_emergency=True,
        )

    # start SUMO via SumoEnvironment so RL controller for signals can be used
    env = SumoEnvironment(args.config, use_gui=bool(args.gui), num_seconds=args.duration)

    # start background monitoring (CPU/memory)
    try:
        start_monitoring(out_dir=os.path.join('results', 'monitoring'), interval=10.0)
    except Exception:
        logger.exception('Failed to start background monitoring')
    env.start_simulation(seed=args.seed)

    # If adjacency appears empty (sumolib failed), try to rebuild adjacency from TraCI now that SUMO is running
    try:
        if not adjacency:
            logger.warning('Adjacency empty from sumolib; rebuilding from TraCI edge API')
            adjacency = {}
            edge_info = {}
            edge_ids = traci.edge.getIDList()
            # Only keep regular edges (skip junction edges starting with ':')
            regular_edges = [e for e in edge_ids if not e.startswith(':')]
            test_edges = ['hw1_to_hw2', 'hw2_to_hw3', 'hw4_to_hw5']
            for edge_id in regular_edges:
                adjacency.setdefault(edge_id, [])
                try:
                    # Get outgoing edges via the edge's target junction
                    to_junction = traci.edge.getToJunction(edge_id)
                    # getOutgoingEdges returns all edges (both junction and regular)
                    # We need only REGULAR edges that start from to_junction
                    all_junction_edges = traci.junction.getOutgoingEdges(to_junction) or []
                    out_edges = []
                    for cand_edge in all_junction_edges:
                        # Skip junction edges (starting with ':') - only use regular edges
                        if cand_edge.startswith(':'):
                            continue
                        # Only add if the candidate edge starts from our target junction
                        try:
                            cand_from = traci.edge.getFromJunction(cand_edge)
                            if cand_from == to_junction:
                                out_edges.append(cand_edge)
                        except Exception:
                            pass
                    adjacency[edge_id].extend(out_edges)
                    # Diagnostic for specific test edges
                    if edge_id in test_edges:
                        print(f'[REBUILD DEBUG] {edge_id} -> junction {to_junction} -> {len(out_edges)} regular outgoing (from {len(all_junction_edges)} total)')
                except Exception as e:
                    # fallback: use empty outgoing list
                    logger.debug(f'Failed to get outgoing edges for {edge_id}: {e}')
                    if edge_id in test_edges:
                        print(f'[REBUILD DEBUG] {edge_id} raised exception: {e}')
                    pass
                try:
                    edge_info[edge_id] = {'length': float(traci.edge.getLength(edge_id))}
                except Exception:
                    edge_info[edge_id] = {'length': 1.0}
            # update pheromone manager and optimizer instances if already created
            try:
                # pheromone may exist in outer scope
                pheromone.adjacency = adjacency
                # (re)initialize pheromone table entries for all directed edges
                pheromone.pheromones = {}
                for u, outs in adjacency.items():
                    for v in outs:
                        pheromone.pheromones[(u, v)] = float(1.0)
                print(f'REBUILT GRAPH STATE: adjacency_size={len(adjacency)}, pheromone_count={len(pheromone.pheromones)}')
                print('REBUILT ADJ_KEYS=', list(adjacency.keys())[:20])
                print('REBUILT PH_KEYS=', [f'{u}->{v}' for (u,v) in list(pheromone.pheromones.keys())[:20]])
                # CRITICAL: verify regular edges are in adjacency
                for test_edge in test_edges:
                    if test_edge in adjacency:
                        print(f'[REBUILD VERIFY] {test_edge} in adjacency with {len(adjacency[test_edge])} outgoing')
                    else:
                        print(f'[REBUILD VERIFY] {test_edge} NOT in adjacency!')
            except Exception:
                logger.exception('Failed to update pheromone/optimizer during rebuild')
                pass
            try:
                if optimizer is not None:
                    optimizer.adjacency = adjacency
                    optimizer.edge_info = edge_info
            except Exception:
                pass
    except Exception:
        logger.exception('Failed to rebuild adjacency from TraCI')

    sim_step = 0
    last_checkpoint_step = 0
    vehicle_first_seen = {}
    completed_travel_times = []
    completed_waiting_times = []
    route_change_count = 0
    congested_edge_samples = 0
    congested_edge_total = 0

    # Phase 2 (Protected Emergency Corridor) state
    # emergency_id -> {'reserved_edges': set(edge_id), 'start_step': int, 'end_step': Optional[int]}
    _emergency_corridor_state = {}
    _emergency_corridor_logs = []

    # ---------------------------------------------------------------------
    # Runtime emergency detection diagnostics (LOGGING ONLY)
    # ---------------------------------------------------------------------
    _runtime_first_seen = {}  # vehicle_id -> dict of first observation fields
    _runtime_detection_events = []  # list of dicts: per filter evaluation event
    _runtime_active_emergency_counts = []  # list of dicts per periodic step
    _runtime_step_emergency_ids = {}  # step -> list of emergency ids detected by our filter

    # emergency filter tokens (must match the corridor wiring predicate)
    _EMERGENCY_TOKENS = ('emergency', 'ambulance', 'police', 'firetruck', 'fire_truck', 'rescue')

    def _best_effort_vehicle_class(vid: str) -> str:
        try:
            # Some SUMO versions support getVehicleClass
            if hasattr(traci.vehicle, 'getVehicleClass'):
                return str(traci.vehicle.getVehicleClass(vid))
        except Exception:
            pass
        return ''

    def _is_emergency_by_typeid(type_id: str) -> tuple[bool, str]:
        """Return (is_emergency, failure_reason)"""
        if not type_id:
            return False, 'empty_type_id'
        v = str(type_id).lower()
        if any(tok in v for tok in _EMERGENCY_TOKENS):
            return True, ''
        # detect close variants for debugging
        if 'fire' in v:
            return False, 'fire_present_but_no_token_match'
        if 'ambulance' in v:
            return False, 'ambulance_present_but_no_token_match'
        if 'police' in v:
            return False, 'police_present_but_no_token_match'
        return False, 'no_token_match'

    # Runtime emergency detection helpers (logging-only; do not affect control logic)
    def _vehicle_route_edges(vid: str):
        try:
            r = list(traci.vehicle.getRoute(vid) or [])
            return r
        except Exception:
            return []

    def _current_emergency_ids_for_diagnostics():
        """Compute emergency set using the same emergency token logic used elsewhere."""
        emergency_ids = []
        all_ids = []
        try:
            all_ids = list(traci.vehicle.getIDList())
        except Exception:
            all_ids = []
        for vid in all_ids:
            try:
                type_id = str(traci.vehicle.getTypeID(vid) or '')
            except Exception:
                type_id = ''
            is_em, _reason = _is_emergency_by_typeid(type_id)
            if is_em:
                emergency_ids.append(vid)
        return emergency_ids

    def _record_first_observation_if_new(vid: str):
        """Log first observation for every new vehicle."""
        if vid in _runtime_first_seen:
            return
        try:
            type_id = ''
            vehicle_class = ''
            route_edges = _vehicle_route_edges(vid)
            if hasattr(traci.vehicle, 'getTypeID'):
                type_id = str(traci.vehicle.getTypeID(vid) or '')
            vehicle_class = _best_effort_vehicle_class(vid)
            first_edge = route_edges[0] if route_edges else ''
            last_edge = route_edges[-1] if route_edges else ''
            route_length = int(len(route_edges))
        except Exception:
            type_id = ''
            vehicle_class = ''
            route_edges = []
            first_edge = ''
            last_edge = ''
            route_length = 0
        _runtime_first_seen[vid] = {
            'vehicle_id': vid,
            'type_id': type_id,
            'vehicle_class': vehicle_class,
            'route_length': int(route_length),
            'first_edge': first_edge,
            'last_edge': last_edge,
            'first_seen_step': int(sim_step),
        }
        logger.info(
            "EMERGENCY_DIAG_FIRST_OBS VID=%s TYPE=%s CLASS=%s ROUTE_LEN=%s FIRST_EDGE=%s LAST_EDGE=%s",
            vid,
            type_id,
            vehicle_class,
            route_length,
            first_edge,
            last_edge,
        )

    def _trace_emergency_filter_for_vehicle(vid: str):
        """Emergency filter tracing at detection point (logs PASS/FAIL + reason)."""
        try:
            type_id = str(traci.vehicle.getTypeID(vid) or '')
        except Exception:
            type_id = ''
        vehicle_class = _best_effort_vehicle_class(vid)
        is_em, failure_reason = _is_emergency_by_typeid(type_id)
        filt = 'PASS' if is_em else 'FAIL'
        diag_event = {
            'step': int(sim_step),
            'vehicle_id': vid,
            'type_id': type_id,
            'vehicle_class': vehicle_class,
            'filter': filt,
            'failure_reason': failure_reason if not is_em else '',
        }
        _runtime_detection_events.append(diag_event)
        if is_em:
            logger.info(
                "EMERGENCY_DIAG_FILTER_V1 VID=%s TYPE=%s CLASS=%s FILTER=PASS",
                vid,
                type_id,
                vehicle_class,
            )
        else:
            logger.info(
                "EMERGENCY_DIAG_FILTER_V1 VID=%s TYPE=%s CLASS=%s FILTER=FAIL REASON=%s",
                vid,
                type_id,
                vehicle_class,
                failure_reason,
            )
        return is_em, failure_reason

    try:
        while sim_step < args.duration:
            traci.simulationStep()
            sim_step += 1


            try:
                current_vehicle_ids = set(traci.vehicle.getIDList())
            except Exception:
                current_vehicle_ids = set()
            try:
                arrived_vehicle_ids = set(traci.simulation.getArrivedIDList())
            except Exception:
                arrived_vehicle_ids = set()

            for vehicle_id in current_vehicle_ids:
                is_new = vehicle_id not in _runtime_first_seen
                vehicle_first_seen.setdefault(vehicle_id, sim_step)
                if is_new:
                    _record_first_observation_if_new(vehicle_id)


            try:
                congested_edge_count = 0
                for edge_id in traci.edge.getIDList():
                    try:
                        if traci.edge.getLastStepHaltingNumber(edge_id) > 0:
                            congested_edge_count += 1
                    except Exception:
                        continue
                congested_edge_total += congested_edge_count
                congested_edge_samples += 1
            except Exception:
                pass

            # keep the environment's rolling metrics in sync even though we are driving SUMO directly here
            update_episode_metrics(env)

            # Runtime diagnostics: per-step emergency filter tracing + counts (logging-only)
            try:
                current_vehicle_count = int(len(current_vehicle_ids)) if current_vehicle_ids is not None else 0
            except Exception:
                current_vehicle_count = 0

            active_emergency_ids_for_diag = []
            try:
                for vid in list(current_vehicle_ids):
                    is_em, _reason = _trace_emergency_filter_for_vehicle(vid)
                    if is_em:
                        active_emergency_ids_for_diag.append(vid)
            except Exception:
                pass

            if sim_step % 10 == 0:
                logger.info(
                    "EMERGENCY_DIAG_STEP current_step=%s total_vehicle_count=%s active_emergency_count=%s",
                    int(sim_step),
                    int(current_vehicle_count),
                    int(len(active_emergency_ids_for_diag)),
                )

            _runtime_active_emergency_counts.append({
                'step': int(sim_step),
                'total_vehicle_count': int(current_vehicle_count),
                'active_emergency_count': int(len(active_emergency_ids_for_diag)),
            })

            # record detected emergency ids in a per-step map (for report)
            _runtime_step_emergency_ids[int(sim_step)] = list(active_emergency_ids_for_diag)

            preds = {}
            # every N steps, call PINN to update predicted congestion

            if args.enable_pinn and (sim_step % max(1, args.pinn_interval) == 0):
                edges = list(adjacency.keys())
                preds = pinn_predictor(edges, future_step=sim_step, lookahead_steps=args.lookahead_steps)
                preds = normalize_predictions(preds)

            # Phase 2 protected emergency corridor edge reservation wiring
            if args.enable_aco and optimizer is not None:
                try:
                    # Build emergency_id -> reserved_edges each cycle (lifecycle-managed)
                    active_emergencies = {}
                    all_ids = set(traci.vehicle.getIDList())
                    for vid in all_ids:
                        try:
                            vtype = traci.vehicle.getTypeID(vid).lower()
                        except Exception:
                            vtype = ''
                        if not any(tok in vtype for tok in ('emergency', 'ambulance', 'police', 'firetruck', 'fire_truck', 'rescue')):
                            continue
                        try:
                            route = list(traci.vehicle.getRoute(vid) or [])
                        except Exception:
                            route = []
                        if not route:
                            continue
                        # Determine reserved corridor edges ahead (best-effort lookahead)
                        reserved_edges = []
                        # keep within ~150m ahead using edge lengths when available
                        dist_acc = 0.0
                        for e in route:
                            if e is None:
                                continue
                            reserved_edges.append(e)
                            try:
                                dist_acc += float(edge_info.get(e, {}).get('length', 1.0))
                            except Exception:
                                dist_acc += 1.0
                            if dist_acc >= 120.0:
                                break
                        reserved_set = set(reserved_edges)
                        active_emergencies[vid] = reserved_set

                    # Lifecycle: remove corridors for completed/left emergencies
                    # start when newly detected, end when no longer active
                    # Since we don't have explicit "destination reached" events here,
                    # we use: corridor ends when vehicle is no longer active in this step.
                    current_active_ids = set(active_emergencies.keys())
                    # end missing emergencies
                    for eid in list(_emergency_corridor_state.keys()):
                        if eid not in current_active_ids:
                            st = _emergency_corridor_state.get(eid) or {}
                            st['end_step'] = int(sim_step)
                            _emergency_corridor_logs.append({
                                'emergency_id': eid,
                                'reserved_edge_count': int(len(st.get('reserved_edges', set()) or [])),
                                'corridor_start_time': int(st.get('start_step', sim_step)),
                                'corridor_end_time': int(st.get('end_step', sim_step)),
                                'reroutes_caused': None,
                            })
                            _emergency_corridor_state.pop(eid, None)

                    # refresh state and optimizer reservations
                    for eid, reserved_set in active_emergencies.items():
                        if eid not in _emergency_corridor_state:
                            _emergency_corridor_state[eid] = {
                                'reserved_edges': set(reserved_set),
                                'start_step': int(sim_step),
                                'end_step': None,
                            }
                        else:
                            _emergency_corridor_state[eid]['reserved_edges'] = set(reserved_set)

                    optimizer.reserved_edges_by_emergency = {
                        eid: st.get('reserved_edges', set())
                        for eid, st in _emergency_corridor_state.items()
                        if st.get('reserved_edges')
                    }
                except Exception:
                    # never block ACO operation
                    pass

                # update optimizer using current predictions
                optimizer.step(sim_step, pinn_predictions=preds)


            try:
                for vehicle_id in list(current_vehicle_ids):
                    try:
                        # stale-vehicle guard: vehicle may have left the network between IDList() and now
                        if vehicle_id not in traci.vehicle.getIDList():
                            continue
                        current_route = tuple(traci.vehicle.getRoute(vehicle_id))
                    except Exception:
                        current_route = tuple()
                    previous_route = getattr(args, '_vehicle_prev_routes', {}).get(vehicle_id)
                    if previous_route is not None and previous_route != current_route:
                        route_change_count += 1
                    if not hasattr(args, '_vehicle_prev_routes'):
                        args._vehicle_prev_routes = {}
                    args._vehicle_prev_routes[vehicle_id] = current_route

                for vehicle_id in list(arrived_vehicle_ids):
                    if vehicle_id in vehicle_first_seen:
                        completed_travel_times.append(float(sim_step - vehicle_first_seen[vehicle_id]))
                        try:
                            if vehicle_id not in traci.vehicle.getIDList():
                                completed_waiting_times.append(0.0)
                            else:
                                completed_waiting_times.append(float(traci.vehicle.getAccumulatedWaitingTime(vehicle_id)))
                        except Exception:
                            completed_waiting_times.append(0.0)
            except Exception:
                pass


            # emergency corridor coordination: phase-plan multiple TLSs along active emergency routes
            if HybridSmartCityController is not None:
                try:
                    # stale-vehicle guard: re-check membership before querying route/state
                    all_ids = set(traci.vehicle.getIDList())
                    emergency_ids = [
                        vid for vid in all_ids
                        if vid in all_ids and traci.vehicle.getTypeID(vid).lower() in ('emergency', 'ambulance', 'police', 'firetruck', 'fire_truck')
                    ]
                    for vid in list(emergency_ids):
                        try:
                            if vid not in traci.vehicle.getIDList():
                                continue
                            route = traci.vehicle.getRoute(vid)
                        except Exception:
                            continue
                        if not route:
                            continue
                        tls_list = traci.trafficlight.getIDList()
                        corridor = []
                        for ts in tls_list:
                            lanes = traci.trafficlight.getControlledLanes(ts)
                            edges = set(traci.lane.getEdgeID(l) for l in lanes)
                            if any(e in edges for e in route[:6]):
                                corridor.append(ts)
                        if corridor:
                            HybridSmartCityController.apply_coordinated_emergency_green(
                                corridor,
                                sim_step=sim_step,
                                hold_steps=args.emergency_hold_steps,
                                route_edges=route,
                            )
                except Exception:
                    pass


            # integrate RL controller for traffic signals if available
            if HybridSmartCityController is not None:
                try:
                    # controller can manage signals per step
                    HybridSmartCityController.step_traci(traci, sim_step)
                except Exception:
                    pass

            if args.metrics_checkpoint_interval > 0 and (sim_step - last_checkpoint_step) >= args.metrics_checkpoint_interval:
                snapshot_metrics(env, sim_step, args.run_name, args.metrics_output_dir, current_predictions=preds)
                last_checkpoint_step = sim_step
                try:
                    save_json(os.path.join(args.metrics_output_dir, f'{args.run_name}_state.json'), {
                        'step': sim_step,
                        'mode': args.run_name,
                        'pinn_enabled': args.enable_pinn,
                        'aco_enabled': args.enable_aco,
                        'current_predictions': preds,
                    })
                except Exception:
                    logger.exception('Failed to save integration state snapshot at step %s', sim_step)

    except KeyboardInterrupt:
        logger.warning('Integration run interrupted at step %s; saving final snapshot.', sim_step)
    finally:
        # collect environment metrics before closing TraCI; keep them even if close() fails
        try:
            metrics = env.get_episode_metrics()
        except Exception:
            metrics = {}
        try:
            snapshot_metrics(env, sim_step, args.run_name, args.metrics_output_dir, current_predictions=preds if 'preds' in locals() else {})
        except Exception:
            pass
        try:
            traci.close()
        except Exception:
            pass
        # dump optimizer debug summary if available
        try:
            if args.enable_aco and optimizer is not None:
                summary_out = None
                if args.aco_debug_log:
                    summary_out = args.aco_debug_log + '.summary.json'
                else:
                    summary_out = os.path.join('results', 'aco_results', 'aco_debug_summary.json')
                try:
                    optimizer.dump_debug_summary(summary_out)
                except Exception:
                    logger.exception('Failed to dump optimizer debug summary')
        except Exception:
            pass

    # Save pheromone CSV for post-run visualization
    try:
        os.makedirs('results/aco_results', exist_ok=True)
        pheromone.dump_csv('results/aco_results/pheromones_final.csv')
    except Exception:
        pass

    reroute_log_count = 0
    if args.reroute_log and os.path.exists(args.reroute_log):
        try:
            with open(args.reroute_log, 'r', encoding='utf-8') as handle:
                reroute_log_count = max(0, sum(1 for _ in handle) - 1)
        except Exception:
            reroute_log_count = 0

    # Phase 2 + Phase 3 required report output
    # Note: current Phase 2 logging state is populated in _emergency_corridor_logs, and
    # Phase 3 clearing is not yet integrated in this runner.
    emergency_detected = int(len(_emergency_corridor_logs))
    emergency_completed = int(len(_emergency_corridor_logs))  # best-effort for this integration runner
    corridor_activation_count = int(len(_emergency_corridor_logs))

    corridor_success_rate = 0.0
    if corridor_activation_count > 0:
        # No explicit corridor KPI computed yet; assume success if at least 1 protected edge.
        successes = 0
        for e in _emergency_corridor_logs:
            if int(e.get('reserved_edge_count', 0)) > 0:
                successes += 1
        corridor_success_rate = float(successes) / float(corridor_activation_count)

    emergency_stop_count = 0
    emergency_stop_rate = 0.0
    average_emergency_delay = 0.0

    # Runtime emergency detection report (markdown) - required diagnostics output
    runtime_md_path = os.path.join('results', 'runtime_emergency_detection_report.md')
    try:
        # Determine detected vs missed emergencies from filter tracing events
        detected_vehicle_ids = sorted({e.get('vehicle_id') for e in _runtime_detection_events if e.get('filter') == 'PASS'})
        missed_vehicle_ids = sorted({e.get('vehicle_id') for e in _runtime_detection_events if e.get('filter') == 'FAIL'})

        first_detection_times = {}
        for step, vids in _runtime_step_emergency_ids.items():
            for v in vids:
                first_detection_times.setdefault(v, step)

        # Root-cause inference when emergency_detected==0
        root_cause = None
        if emergency_detected == 0:
            reasons = []
            for ev in _runtime_detection_events:
                if ev.get('filter') == 'FAIL':
                    r = ev.get('failure_reason')
                    if r:
                        reasons.append(r)
            if reasons:
                # Most frequent failure reason
                freq = {}
                for r in reasons:
                    freq[r] = freq.get(r, 0) + 1
                root_cause = max(freq.items(), key=lambda kv: kv[1])[0]
            else:
                root_cause = 'no_filter_failures_recorded'

        # Render markdown
        md_lines = []
        md_lines.append('# Runtime Emergency Detection Diagnostics')
        md_lines.append('')
        md_lines.append(f"- emergency_detected (corridor log proxy): {emergency_detected}")
        md_lines.append(f"- corridor_activation_count (corridor log proxy): {corridor_activation_count}")
        md_lines.append('')

        md_lines.append('## Detected emergency vehicles')
        if detected_vehicle_ids:
            md_lines.extend([f'- {vid}' for vid in detected_vehicle_ids])
        else:
            md_lines.append('- (none)')
        md_lines.append('')

        md_lines.append('## Missed emergency vehicles (filter FAIL events)')
        if missed_vehicle_ids:
            md_lines.extend([f'- {vid}' for vid in missed_vehicle_ids])
        else:
            md_lines.append('- (none)')
        md_lines.append('')

        md_lines.append('## First detection time (step)')
        if first_detection_times:
            for vid, st in sorted(first_detection_times.items(), key=lambda x: x[1]):
                md_lines.append(f'- {vid}: step {st}')
        else:
            md_lines.append('- (no PASS detections recorded)')
        md_lines.append('')

        md_lines.append('## Active emergency counts (every 10 steps)')
        if _runtime_active_emergency_counts:
            for row in _runtime_active_emergency_counts:
                md_lines.append(f"- step {row.get('step')}: active_emergency_count={row.get('active_emergency_count')}, total_vehicle_count={row.get('total_vehicle_count')}")
        else:
            md_lines.append('- (none)')
        md_lines.append('')

        if emergency_detected == 0:
            md_lines.append('## Root cause (emergency_detected == 0)')
            md_lines.append(f'- exact_root_cause: {root_cause}')
        md_lines.append('')

        with open(runtime_md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))
    except Exception:
        logger.exception('Failed to write runtime_emergency_detection_report.md')

    report = {
        'emergency_detected': emergency_detected,

        'emergency_completed': emergency_completed,
        'emergency_stop_count': emergency_stop_count,
        'emergency_stop_rate': emergency_stop_rate,
        'average_emergency_delay': average_emergency_delay,
        'corridor_activation_count': corridor_activation_count,
        'corridor_success_rate': float(corridor_success_rate),
        'green_wave_success_rate': 0.0,
        'traffic_clearing_actions': 0,
        'lane_change_success_rate': 0.0,
        'aco_reroutes': int(reroute_log_count),
        'emergency_eta_improvement': 0.0,
        # For debugging / traceability
        'protected_corridor_logs': _emergency_corridor_logs,
    }

    try:
        os.makedirs('results', exist_ok=True)
        save_json('results/emergency_optimization_report.json', report)
    except Exception:
        logger.exception('Failed to write results/emergency_optimization_report.json')

    summary = {
        'mode': args.run_name,
        'seed': args.seed,
        'enable_aco': args.enable_aco,
        'enable_pinn': args.enable_pinn,
        'duration': args.duration,
        'average_travel_time': float(sum(completed_travel_times) / len(completed_travel_times)) if completed_travel_times else 0.0,
        'average_waiting_time': float(sum(completed_waiting_times) / len(completed_waiting_times)) if completed_waiting_times else 0.0,
        'average_queue_length': float(env.get_episode_metrics().get('queue', 0.0)),
        'throughput': float(env.get_episode_metrics().get('throughput', 0.0)),
        'reroutes': int(reroute_log_count),
        'route_changes': int(route_change_count),
        'average_congested_edge_utilization': float(congested_edge_total / congested_edge_samples) if congested_edge_samples else 0.0,
        'completed_vehicles': int(len(completed_travel_times)),
    }
    try:
        save_json(os.path.join(args.metrics_output_dir, f'{args.run_name}_summary.json'), summary)
    except Exception:
        logger.exception('Failed to write ACO summary JSON')

    print('ACO integrated run complete.')
    print('METRICS:', summary)


if __name__ == '__main__':
    main()
