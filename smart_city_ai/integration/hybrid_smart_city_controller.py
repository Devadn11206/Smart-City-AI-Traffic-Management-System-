import os
import sys
import time
import numpy as np
from collections import deque
import logging
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traci
from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent


logger = logging.getLogger(__name__)


class HybridSmartCityController:
    """Simple hybrid controller that loads RL agents per traffic light and
    applies actions by calling TraCI. Also exposes emergency priority hooks.
    """
    _initialized = False
    _agents = {}
    _ts_list = []
    _last_decision = 0
    decision_interval = 5
    emergency_state = {}
    # Predictive green-wave reservation window (Phase 1)
    # When an emergency corridor is activated we will reserve the next N TLS ahead
    # along the emergency route and refresh/extend hold until the reserved corridor is cleared.
    green_wave_lookahead_tls = 5

    _phase_cache = {}
    _last_phase_change_step = {}
    _emergency_corridor = []
    _emergency_phase_plan = {}
    min_green_steps = 8
    phase_cooldown_steps = 3
    max_switches_per_window = 6
    _switch_history = {}

    @classmethod
    def _safe_phase_index(cls, ts, desired_phase):
        try:
            logics = traci.trafficlight.getAllProgramLogics(ts)
            if logics and len(logics) > 0:
                phase_count = max(1, len(logics[0].getPhases()))
                return max(0, min(int(desired_phase), phase_count - 1))
        except Exception:
            logger.exception('Failed to compute safe phase index for %s', ts)
        return max(0, int(desired_phase))

    @classmethod
    def initialize(cls):
        if cls._initialized:
            return
        cls._ts_list = traci.trafficlight.getIDList()
        cls._agents = {}
        repo_root = Path(__file__).resolve().parent.parent
        model_paths = {
            'dqn': [repo_root / "models" / "dqn" / "dqn_model_best.pth", repo_root / "results" / "hybrid_results" / "hybrid_dqn_model_best.pth"],
            'ddqn': [repo_root / "models" / "ddqn" / "ddqn_model_best.pth", repo_root / "results" / "hybrid_results" / "hybrid_ddqn_model_best.pth"],
            'ppo': [repo_root / "models" / "demo" / "ppo_demo_model.pth", repo_root / "results" / "hybrid_results" / "hybrid_ppo_model_best.pth"]
        }
        types = ['ddqn', 'ppo', 'dqn']
        for i, ts in enumerate(cls._ts_list):
            a_type = types[i % len(types)]
            try:
                if a_type == 'dqn':
                    agent = DQNAgent(15, 4)
                elif a_type == 'ddqn':
                    agent = DDQNAgent(15, 4)
                else:
                    agent = PPOAgent(15, 4)
            except Exception:
                logger.exception('Failed to initialize agent of type %s', a_type)
                agent = None
            # try load
            for p in model_paths.get(a_type, []):
                try:
                    if not Path(p).exists():
                        continue
                    if agent is not None:
                        agent.load(p)
                        break
                except Exception:
                    logger.exception('Failed to load %s agent checkpoint from %s', a_type, p)
            cls._agents[ts] = agent
            cls._last_phase_change_step[ts] = 0
            cls._switch_history[ts] = deque(maxlen=20)
        cls._initialized = True

    @classmethod
    def _tls_state(cls, ts):
        lanes = list(set(traci.trafficlight.getControlledLanes(ts)))
        try:
            state_str = traci.trafficlight.getRedYellowGreenState(ts)
            links = traci.trafficlight.getControlledLinks(ts)
        except Exception:
            return np.zeros(15, dtype=np.float32)

        green_lanes = set()
        red_lanes = set()

        for i, link_info in enumerate(links):
            if not link_info:
                continue
            lane = link_info[0][0]
            if i < len(state_str) and state_str[i].lower() == 'g':
                green_lanes.add(lane)
            else:
                red_lanes.add(lane)

        red_lanes = red_lanes - green_lanes

        green_metrics = {'queue_len': 0, 'wait_time': 0, 'vehicles': 0, 'ev_present': 0.0, 'nearest_ev_dist': 1000.0, 'max_ev_wait': 0.0}
        red_metrics = {'queue_len': 0, 'wait_time': 0, 'vehicles': 0, 'ev_present': 0.0, 'nearest_ev_dist': 1000.0, 'max_ev_wait': 0.0}

        for lane in lanes:
            is_green = lane in green_lanes
            metrics = green_metrics if is_green else red_metrics
            
            metrics['queue_len'] += traci.lane.getLastStepHaltingNumber(lane)
            metrics['wait_time'] += traci.lane.getWaitingTime(lane)
            metrics['vehicles'] += traci.lane.getLastStepVehicleNumber(lane)
            
            lane_length = traci.lane.getLength(lane)
            for vid in traci.lane.getLastStepVehicleIDs(lane):
                try:
                    vtype = traci.vehicle.getTypeID(vid).lower()
                    if "emergency" in vtype or "ambulance" in vtype or "police" in vtype or "fire" in vtype:
                        metrics['ev_present'] = 1.0
                        pos = traci.vehicle.getLanePosition(vid)
                        dist = max(0.0, lane_length - pos)
                        metrics['nearest_ev_dist'] = min(metrics['nearest_ev_dist'], dist)
                        metrics['max_ev_wait'] = max(metrics['max_ev_wait'], traci.vehicle.getWaitingTime(vid))
                except Exception:
                    pass

        green_density = green_metrics['vehicles'] / max(1, len(green_lanes))
        red_density = red_metrics['vehicles'] / max(1, len(red_lanes))
        total_occupancy = sum(traci.lane.getLastStepOccupancy(l) for l in lanes) / max(1, len(lanes))
        congestion_indicator = 1.0 if total_occupancy > 0.7 else 0.0

        try:
            current_phase = traci.trafficlight.getPhase(ts)
            logics = traci.trafficlight.getAllProgramLogics(ts)
            phase_count = max(1, len(logics[0].getPhases())) if logics else 1
            normalized_phase = float(current_phase) / float(phase_count)
        except Exception:
            normalized_phase = 0.0

        state = np.array([
            normalized_phase,
            0.0,  # _phase_persistence not easily tracked per ts here
            float(green_metrics['queue_len']) / 50.0,
            float(red_metrics['queue_len']) / 50.0,
            float(green_metrics['wait_time']) / 1000.0,
            float(red_metrics['wait_time']) / 1000.0,
            float(green_density) / 50.0,
            float(red_density) / 50.0,
            float(green_metrics['ev_present']),
            float(red_metrics['ev_present']),
            float(green_metrics['nearest_ev_dist']) / 1000.0,
            float(red_metrics['nearest_ev_dist']) / 1000.0,
            float(green_metrics['max_ev_wait']) / 100.0,
            float(red_metrics['max_ev_wait']) / 100.0,
            float(congestion_indicator)
        ], dtype=np.float32)

        return state

    @classmethod
    def step_traci(cls, traci_module, sim_step):
        """Apply RL-controlled traffic light actions.

        NOTE: This method MUST NOT call traci.simulationStep().
        The main simulation loop owns the TraCI simulation stepping.
        """
        # Initialize agents if not yet
        if not cls._initialized:
            cls.initialize()

        if sim_step - cls._last_decision < cls.decision_interval:
            return
        cls._last_decision = sim_step

        # Required stability logging (best-effort; must not crash)
        try:
            emergency_ids = [
                vid for vid in traci.vehicle.getIDList()
                if traci.vehicle.getTypeID(vid).lower() in (
                    'emergency', 'ambulance', 'police', 'firetruck', 'fire_truck'
                )
            ]
            active_emergency_count = len(emergency_ids)
            active_corridors = len(getattr(cls, '_emergency_corridor', []) or [])
            active_tls = len(getattr(cls, '_ts_list', []) or [])
            simulation_time = None
            try:
                simulation_time = traci.simulation.getTime()
            except Exception:
                simulation_time = sim_step
            logger.info(
                'TraCI step_traci telemetry: current_step=%s active_emergency_count=%s active_corridors=%s active_tls=%s simulation_time=%s',
                int(sim_step),
                int(active_emergency_count),
                int(active_corridors),
                int(active_tls),
                simulation_time,
            )
        except traci.exceptions.FatalTraCIError:
            # fatal: connection is broken; let outer loop close gracefully
            logger.exception('FatalTraCIError in step_traci telemetry at step %s', sim_step)
            return
        except Exception:
            # never block simulation on telemetry issues
            logger.debug('Telemetry collection failed at step %s', sim_step)

        # apply pending green before making new decisions

        try:
            pending = getattr(cls, '_pending_green', {}) or {}
            for ts, plan in list(pending.items()):
                try:
                    if int(plan.get('scheduled_step', 10**9)) <= int(sim_step):
                        traci.trafficlight.setPhase(ts, cls._safe_phase_index(ts, plan.get('green_phase', 0)))
                        pending.pop(ts, None)
                except traci.exceptions.FatalTraCIError:
                    logger.exception('FatalTraCIError applying pending green at step %s', sim_step)
                    return
                except Exception:
                    logger.exception('Failed applying pending green TLS %s at step %s', ts, sim_step)
            cls._pending_green = pending
        except traci.exceptions.FatalTraCIError:
            logger.exception('FatalTraCIError during pending green processing at step %s', sim_step)
            return
        except Exception:
            logger.debug('Pending green processing failed at step %s', sim_step)

        except Exception:
            # never block simulation on telemetry issues
            logger.debug('Telemetry collection failed at step %s', sim_step)

        # complete any previously scheduled green phases

        try:
            pending = getattr(cls, '_pending_green', {}) or {}
            for ts, plan in list(pending.items()):
                try:
                    if int(plan.get('scheduled_step', 10**9)) <= int(sim_step):
                        traci.trafficlight.setPhase(ts, cls._safe_phase_index(ts, plan.get('green_phase', 0)))
                        pending.pop(ts, None)
                except Exception:
                    logger.exception('Failed applying pending green TLS %s at step %s', ts, sim_step)
            cls._pending_green = pending
        except Exception:
            logger.debug('Pending green processing failed at step %s', sim_step)

        for ts in cls._ts_list:
            agent = cls._agents.get(ts)

            if agent is None:
                continue
            try:
                # Emergency preemption: if a corridor is active, do not let RL control override it.
                if ts in cls.emergency_state:
                    continue
                state = cls._tls_state(ts)
                action = agent.act(state, eval_mode=True)

                if isinstance(action, tuple):
                    action = action[0]
                current_phase = traci.trafficlight.getPhase(ts)
                phase_count = 4
                try:
                    logic_list = traci.trafficlight.getAllProgramLogics(ts)
                    if logic_list and len(logic_list) > 0:
                        phase_count = max(1, len(logic_list[0].getPhases()))
                except Exception:
                    logger.exception('Failed to inspect program logics for TLS %s', ts)
                macro_action = int(action) % 2
                last_change = cls._last_phase_change_step.get(ts, 0)
                if macro_action == 0:
                    continue
                if sim_step - last_change < cls.min_green_steps:
                    continue
                switch_hist = cls._switch_history.setdefault(ts, deque(maxlen=20))
                if len(switch_hist) >= switch_hist.maxlen and sum(switch_hist) >= cls.max_switches_per_window:
                    continue
                if phase_count <= 1:
                    continue
                if current_phase % 2 == 0:
                    phase_index = cls._safe_phase_index(ts, (current_phase + 2) % phase_count)
                else:
                    phase_index = cls._safe_phase_index(ts, (current_phase + 1) % phase_count)
                if current_phase != phase_index:
                    cls._phase_cache[ts] = current_phase
                    # mandatory yellow transition before the next safe green phase
                    # IMPORTANT: do NOT call traci.simulationStep() here.
                    # The main simulation loop owns simulationStep() cadence.
                    yellow_phase = cls._safe_phase_index(ts, current_phase + 1)
                    traci.trafficlight.setPhase(ts, yellow_phase)
                    # schedule the green phase for the next controller tick
                    cls._pending_green = getattr(cls, '_pending_green', {})
                    cls._pending_green[ts] = {
                        'green_phase': int(phase_index),
                        'scheduled_step': int(sim_step) + 1,
                    }

                    cls._last_phase_change_step[ts] = sim_step
                    switch_hist.append(1)

                else:
                    switch_hist.append(0)
            except Exception:
                logger.exception('Hybrid controller action failed for TLS %s at step %s', ts, sim_step)

        cls._release_expired_emergency_phases(sim_step)

    @classmethod
    def request_emergency_green(cls, vehicle_id, hold_steps=15):
        """Best-effort method: set green on TLSs controlling the vehicle's upcoming edges.

        This also records a corridor candidate for synchronized phase control on adjacent TLSs.
        """
        try:
            route = traci.vehicle.getRoute(vehicle_id)
            if not route:
                return
            # find TLS that control lanes on upcoming edges
            tls_list = traci.trafficlight.getIDList()
            affected = []
            for ts in tls_list:
                lanes = traci.trafficlight.getControlledLanes(ts)
                edges = set(traci.lane.getEdgeID(l) for l in lanes)
                # if any edge in route intersects
                if any(e in edges for e in route[:3]):
                    affected.append(ts)

            for ts in affected:
                try:
                    # compute and store a route-aligned phase plan instead of a hard-coded override
                    phase_index = cls._safe_phase_index(ts, select_best_phase_for_tls(ts, route))
                    cls._emergency_phase_plan[ts] = phase_index
                    if ts not in cls._emergency_corridor:
                        cls._emergency_corridor.append(ts)
                    if ts not in cls._phase_cache:
                        cls._phase_cache[ts] = traci.trafficlight.getPhase(ts)
                    traci.trafficlight.setPhase(ts, cls._safe_phase_index(ts, phase_index))
                    cls.emergency_state[ts] = {'release_step': sim_step_from_traci() + hold_steps, 'hold_steps': hold_steps}
                except Exception:
                    logger.exception('Failed to request emergency green for TLS %s', ts)
        except Exception:
            logger.exception('Emergency green request failed for vehicle %s', vehicle_id)

    @classmethod
    def synchronize_emergency_corridor(cls, corridor_tls_ids, sim_step, hold_steps=15):
        """Force a synchronized emergency corridor across a list of TLS IDs.

        The current implementation sets the same green phase on each TLS and stores the
        original phase for restoration after hold_steps.
        """
        cls._emergency_corridor = list(dict.fromkeys(corridor_tls_ids))
        for ts in cls._emergency_corridor:
            try:
                phase_index = cls._safe_phase_index(ts, cls._emergency_phase_plan.get(ts, 0))
                if phase_index is None:
                    phase_index = select_best_phase_for_tls(ts, [])
                if ts not in cls._phase_cache:
                    cls._phase_cache[ts] = traci.trafficlight.getPhase(ts)
                traci.trafficlight.setPhase(ts, cls._safe_phase_index(ts, phase_index))
                cls.emergency_state[ts] = {'release_step': sim_step + hold_steps, 'hold_steps': hold_steps}
            except Exception:
                logger.exception('Failed to synchronize emergency corridor for TLS %s', ts)

    @classmethod
    def apply_coordinated_emergency_green(cls, route_tls_ids, sim_step, hold_steps=15, route_edges=None):
        """Convenience method to synchronize a corridor across a route sequence of TLS IDs.

        Phase 1 (Predictive Green Wave): reserve the next N TLS ahead (ordered by
        route edge overlap) and refresh/extend holds until the vehicle no longer overlaps
        the reserved TLS controlled edges.
        """
        corridor = [ts for ts in route_tls_ids if ts in traci.trafficlight.getIDList()]
        if not corridor:
            return

        # Route-ordered TLS lookahead selection: keep only the first K TLS that appear ahead
        # in the emergency route.
        reserved = []
        reserved_set = set()
        route_edges_list = list(route_edges or [])
        for ts in corridor:
            if ts in reserved_set:
                continue

            # Compute the earliest index along route_edges where this TLS controls an edge.
            earliest_idx = None
            try:
                lanes = traci.trafficlight.getControlledLanes(ts)
                controlled_edges = []
                for l in lanes:
                    try:
                        controlled_edges.append(traci.lane.getEdgeID(l))
                    except Exception:
                        pass
                controlled_edge_set = set(controlled_edges)
                for idx, e in enumerate(route_edges_list):
                    if e in controlled_edge_set:
                        earliest_idx = idx
                        break
            except Exception:
                earliest_idx = None

            # If we can't determine ordering, treat as far ahead.
            if earliest_idx is None:
                earliest_idx = 10**9

            # Insert in order by earliest_idx.
            reserved.append((earliest_idx, ts))

        reserved.sort(key=lambda x: x[0])
        tls_ordered = []
        for _, ts in reserved:
            if ts in reserved_set:
                continue
            reserved_set.add(ts)
            tls_ordered.append(ts)
            if len(tls_ordered) >= cls.green_wave_lookahead_tls:
                break

        reserved = tls_ordered


        # derive a TLS-specific emergency phase plan so the reserved corridor is synchronized
        for ts in reserved:
            cls._emergency_phase_plan[ts] = cls._safe_phase_index(ts, select_best_phase_for_tls(ts, route_edges_list))

        # Track reserved corridor explicitly so we can extend holds while progressing.
        cls._emergency_corridor = list(dict.fromkeys(reserved))
        cls._reserved_corridor_edges = set(route_edges_list)
        # Per-TLS reserved edges controlled by that TLS (computed best-effort)
        cls._tls_reserved_edge_map = {}
        for ts in cls._emergency_corridor:

            try:
                cls._phase_cache[ts] = traci.trafficlight.getPhase(ts)
            except Exception:
                pass
            cls.emergency_state[ts] = {
                'release_step': sim_step + int(hold_steps),
                'hold_steps': int(hold_steps),
                'reserved': True,
            }


    @classmethod
    def _release_expired_emergency_phases(cls, sim_step):
        # TraCI time is in seconds (getTime). Release is compared against sim_step, which is also tracked in seconds.
        expired = [ts for ts, st in cls.emergency_state.items() if st.get('release_step', 10**9) <= sim_step]

        # Phase 1 extension: keep reserved TLS locked as long as the emergency vehicle is still approaching.
        # Best-effort: if an emergency vehicle exists and its route overlaps this TLS's controlled edges,
        # extend release_step. Otherwise allow normal restoration.
        try:
            active_emergencies = [
                vid for vid in traci.vehicle.getIDList()
                if traci.vehicle.getTypeID(vid).lower() in ('emergency', 'ambulance', 'police', 'firetruck', 'fire_truck')
            ]
        except Exception:
            active_emergencies = []

        if active_emergencies and expired:
            for ts in list(expired):
                try:
                    lanes = traci.trafficlight.getControlledLanes(ts)
                    controlled_edges = set()
                    for l in lanes:
                        try:
                            controlled_edges.add(traci.lane.getEdgeID(l))
                        except Exception:
                            pass

                    still_relevant = False
                    for vid in active_emergencies:
                        try:
                            route = traci.vehicle.getRoute(vid)
                        except Exception:
                            continue
                        if route and any(e in controlled_edges for e in route[:10]):
                            still_relevant = True
                            break

                    if still_relevant:
                        cls.emergency_state[ts]['release_step'] = int(sim_step) + int(cls.green_wave_lookahead_tls)
                        expired.remove(ts)
                except Exception:
                    # keep original expiration behavior on failure
                    pass

        for ts in expired:
            try:
                prev = cls._phase_cache.get(ts)
                if prev is not None:
                    traci.trafficlight.setPhase(ts, cls._safe_phase_index(ts, prev))
                    cls._last_phase_change_step[ts] = sim_step
            except Exception:
                logger.exception('Failed to restore previous phase for TLS %s', ts)
            cls.emergency_state.pop(ts, None)
            if ts in cls._emergency_corridor:
                cls._emergency_corridor.remove(ts)


def select_best_phase_for_tls(ts_id, route_edges):
    """Pick the TLS phase that best aligns with route edges.

    The selection uses controlled-link overlap and prefers green phases serving route edges.
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
                if state[signal_idx] not in ('G', 'g'):
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
                        score += 0.05
            if score > best_score:
                best_score = score
                best_phase = phase_idx
        return best_phase
    except Exception:
        return 0


def sim_step_from_traci():
    try:
        return int(traci.simulation.getTime())
    except Exception:
        return 0
