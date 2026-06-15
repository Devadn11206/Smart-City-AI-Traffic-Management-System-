import os
import sys
import numpy as np
import traci
import time
from sumolib import checkBinary
from collections import deque
import logging


logger = logging.getLogger(__name__)


class SumoEnvironment:
    def __init__(self, config_file, use_gui=False, num_seconds=2000,
                 min_green=10, max_green=50, action_repeat=5, seed: int | None = 42,
                 reward_clip: float = 2000.0):
        self.config_file = config_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds

        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = 3
        self.action_repeat = max(1, int(action_repeat))
        self.base_seed = None if seed is None else int(seed)
        self.reward_clip = float(reward_clip)
        self.max_switches_per_window = 6
        self._switch_history = deque(maxlen=20)

        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("Please declare environment variable 'SUMO_HOME'")

        self.sumo_binary = checkBinary('sumo-gui') if self.use_gui else checkBinary('sumo')
        self.ts_ids = []
        self.current_step = 0

        # State space size and Action space size
        # We will focus on the first available traffic light for simplicity of RL
        # but collect stats for all
        self.controlled_ts = None
        self.state_dim = 15
        self.action_dim = 4  # four phase combinations (N-S G, E-W G, N-S L, E-W L etc.)

        # Observation normalization (initialized after state_dim possibly changed)
        self._init_obs_normalizer(self.state_dim)

        # Reward / throughput tracking
        self._last_arrived = 0
        self.switch_count = 0
        self.last_action = None
        self._last_phase_change_step = -10**9
        self._last_switch_penalty = 0.0
        self._phase_persistence = 0.0
        self._reward_ema = 0.0
        self.last_reward_breakdown = {
            'queue_delta': 0.0,
            'wait_delta': 0.0,
            'throughput': 0.0,
            'speed': 0.0,
            'congestion': 0.0,
            'switch_penalty': 0.0,
        }

    def start_simulation(self, seed=None):
        # Convert config path to absolute before building sumo_cmd
        abs_config = os.path.abspath(self.config_file)

        sumo_cmd = [
            self.sumo_binary,
            "-c", abs_config,
            "--no-warnings", "true",
        ]

        sim_seed = self.base_seed if seed is None else int(seed)
        if sim_seed is not None:
            sumo_cmd += ["--seed", str(sim_seed)]

        # Ensure logs directory exists
        os.makedirs(os.path.join('results', 'logs'), exist_ok=True)
        sumo_log_path = os.path.join('results', 'logs', f'sumo_{int(time.time())}.log')
        sumo_cmd += ['--log', sumo_log_path]

        # Ensure any existing connection is closed first
        try:
            if traci.isLoaded():
                traci.close(False)
        except Exception:
            pass

        # Hard reset connections dictionary to ensure clean start
        try:
            import traci.main as traci_main
            if hasattr(traci_main, '_connections'):
                traci_main._connections.clear()
        except Exception:
            pass

        os.makedirs('results', exist_ok=True)
        traci_log_path = os.path.join('results', 'traci_debug.log')
        try:
            with open(traci_log_path, 'w', encoding='utf-8') as f:
                f.write('')
        except Exception:
            pass

        # Log cwd/config/sumo command and environment
        try:
            with open(traci_log_path, 'a', encoding='utf-8') as f:
                f.write('\n===== START_SIMULATION =====\n')
                f.write(f"cwd={os.getcwd()}\n")
                f.write(f"use_gui={self.use_gui} duration={self.num_seconds} seed={seed}\n")
                f.write(f"config_file_input={self.config_file}\n")
                f.write(f"config_file_abs={abs_config}\n")
                f.write('sumo_cmd=' + ' '.join(map(str, sumo_cmd)) + '\n')
        except Exception:
            pass

        # Try to connect with explicit retries to capture startup errors
        last_err = None
        for attempt in range(1, 6):
            try:
                with open(traci_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"TraCI start attempt {attempt}/5\n")

                # Standard SUMO/TraCI startup pattern: do NOT specify port
                traci.start(sumo_cmd)

                with open(traci_log_path, 'a', encoding='utf-8') as f:
                    f.write('TraCI connected successfully\n')

                last_err = None
                break
            except Exception as e:
                last_err = e
                try:
                    with open(traci_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"TraCI start failed attempt {attempt}/5: {repr(e)}\n")
                except Exception:
                    pass
                time.sleep(1)

        if last_err is not None:
            with open(traci_log_path, 'a', encoding='utf-8') as f:
                f.write(f"FINAL TraCI connection failure: {repr(last_err)}\n")
            raise last_err

        # Initialize controlled traffic light
        self.ts_ids = traci.trafficlight.getIDList()
        if len(self.ts_ids) > 0:
            self.controlled_ts = self.ts_ids[0]

        # initialize arrival counter for throughput
        try:
            self._last_arrived = traci.simulation.getArrivedNumber()
        except Exception:
            logger.exception('Failed to read initial arrived count from TraCI')
            self._last_arrived = 0

    def reset(self, seed=None):
        if self.current_step != 0:
            try:
                traci.close()
            except Exception:
                logger.exception('Failed to close TraCI cleanly during reset')

        self._episode_seed = self.base_seed if seed is None else int(seed)
        self.start_simulation(seed=self._episode_seed)

        self.current_step = 0
        self.switch_count = 0
        self.last_action = None
        self._last_phase_change_step = -10**9
        self._last_switch_penalty = 0.0
        self._phase_persistence = 0.0
        self._reward_ema = 0.0
        self.cumulative_wait = 0.0
        self.cumulative_queue = 0.0
        self.cumulative_speed = 0.0
        self.cumulative_throughput = 0.0
        self.cumulative_emergency_delay = 0.0
        self.cumulative_emergency_stops = 0.0
        self._prev_evs = set()
        self._prev_emergency_wait = 0.0
        self._ev_distances = {}
        self.rl_same_phase_steps = 0
        self.metric_samples = 0
        self._last_total_wait = 0.0
        self._last_total_queue = 0.0
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        try:
            self._last_arrived = traci.simulation.getArrivedNumber()
        except Exception:
            logger.exception('Failed to read arrived count during reset')
            self._last_arrived = 0

        return self._get_state()

    def step(self, action):
        previous_action = getattr(self, 'current_action', None)
        self.current_action = action
        act_val = int(action)
        if act_val in getattr(self, 'action_counts', {}):
            self.action_counts[act_val] += 1
        else:
            if not hasattr(self, 'action_counts'):
                self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            self.action_counts[act_val] = 1
            
        # Check if the exact target phase changed
        action_changed = 1.0 if previous_action is not None and int(action) != int(previous_action) else 0.0

        prev_change_step = getattr(self, '_last_phase_change_step', -10**9)
        if self.controlled_ts is not None:
            self._set_phase(self.controlled_ts, action)

        if self._last_phase_change_step != prev_change_step:
            self.rl_same_phase_steps = 0
        else:
            self.rl_same_phase_steps = getattr(self, 'rl_same_phase_steps', 0) + 1

        # simulate for a few steps per action
        for _ in range(self.action_repeat):
            try:
                traci.simulationStep()
            except Exception:
                logger.exception('TraCI simulationStep failed at step %s', self.current_step)
                raise
            self.current_step += 1

        next_state = self._get_state()
        reward = self._compute_reward(action_changed)
        self._accumulate_metrics()
        done = self.current_step >= self.num_seconds

        return next_state, reward, done, {}

    def _set_phase(self, ts_id, action):
        # Direct Phase Selection Control: Action directly selects target green phase index.
        # action_dim = 4 implies 4 possible green phases.
        try:
            current_phase = traci.trafficlight.getPhase(ts_id)
            phase_count = max(1, len(traci.trafficlight.getCompleteRedYellowGreenDefinition(ts_id)[0].phases))
            
            # Map action (0 to 3) to actual green phase indices in SUMO (typically 0, 2, 4, 6)
            # Assuming even phases are green.
            target_phase = (int(action) * 2) % phase_count

            if target_phase == current_phase:
                self._phase_persistence = min(1.0, getattr(self, '_phase_persistence', 0.0) + 0.1)
                self._last_switch_penalty = 0.0
                if not hasattr(self, '_switch_history'):
                    self._switch_history = deque(maxlen=10)
                self._switch_history.append(0)
                return

            if self.current_step - getattr(self, '_last_phase_change_step', -10**9) < self.min_green:
                self._last_switch_penalty = 1.0
                self._phase_persistence = 0.0
                return

            if hasattr(self, '_switch_history') and len(self._switch_history) >= self._switch_history.maxlen and sum(self._switch_history) >= getattr(self, 'max_switches_per_window', 5):
                self._last_switch_penalty = 1.0
                self._phase_persistence = 0.0
                return

            # Transition: Yellow -> Target Green
            yellow_phase = min(current_phase + 1, phase_count - 1)
            traci.trafficlight.setPhase(ts_id, yellow_phase)
            traci.simulationStep()
            self.current_step += 1
            
            traci.trafficlight.setPhase(ts_id, target_phase)
            self._last_phase_change_step = self.current_step
            self.switch_count += 1
            if not hasattr(self, '_switch_history'):
                self._switch_history = deque(maxlen=10)
            self._switch_history.append(1)
            self.last_action = int(action)
            self._phase_persistence = 0.0

        except Exception:
            logger.exception('Failed to update TLS phase for %s', ts_id)

    def _get_raw_state(self):
        if self.controlled_ts is None:
            return np.zeros(self.state_dim, dtype=np.float32)

        ts_id = self.controlled_ts
        lanes = list(set(traci.trafficlight.getControlledLanes(ts_id)))
        state_str = traci.trafficlight.getRedYellowGreenState(ts_id)
        links = traci.trafficlight.getControlledLinks(ts_id)

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

        # Remove green lanes from red lanes if a lane has multiple links and one is green
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
                        if dist < metrics['nearest_ev_dist']:
                            metrics['nearest_ev_dist'] = dist
                        
                        ev_wait = traci.vehicle.getWaitingTime(vid)
                        if ev_wait > metrics['max_ev_wait']:
                            metrics['max_ev_wait'] = ev_wait
                except Exception:
                    pass

        # Calculate derived metrics
        num_green_lanes = max(1, len(green_lanes))
        num_red_lanes = max(1, len(red_lanes))
        
        green_density = green_metrics['vehicles'] / num_green_lanes
        red_density = red_metrics['vehicles'] / num_red_lanes
        
        # Calculate total occupancy for congestion indicator
        occupancy = 0
        for lane in lanes:
            occupancy += traci.lane.getLastStepOccupancy(lane)
        num_total_lanes = max(1, len(lanes))
        occupancy /= num_total_lanes
        congestion_indicator = 1.0 if occupancy > 0.7 else 0.0

        current_phase = traci.trafficlight.getPhase(ts_id)
        phase_count = max(1, len(traci.trafficlight.getCompleteRedYellowGreenDefinition(ts_id)[0].phases))
        normalized_phase = float(current_phase) / float(phase_count)

        state = np.array([
            normalized_phase,
            float(getattr(self, '_phase_persistence', 0.0)),
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

    def _get_state(self):
        raw_state = self._get_raw_state()
        return self._normalize_observation(raw_state)

    def _compute_reward(self, action_changed=0.0):
        if self.controlled_ts is None:
            return 0

        lanes = traci.trafficlight.getControlledLanes(self.controlled_ts)
        total_wait_time = sum([traci.lane.getWaitingTime(lane) for lane in set(lanes)])
        total_queue = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in set(lanes)])
        max_lane_wait = max([traci.lane.getWaitingTime(lane) for lane in set(lanes)]) if lanes else 0.0

        # throughput: arrived vehicles since last sample
        try:
            arrived = traci.simulation.getArrivedNumber()
            throughput = max(0, arrived - self._last_arrived)
            self._last_arrived = arrived
            
            # Count arrived emergency vehicles
            arrived_vids = traci.simulation.getArrivedIDList()
            completed_evs = 0
            for vid in arrived_vids:
                if vid in getattr(self, '_all_seen_evs', set()):
                    completed_evs += 1
        except Exception:
            logger.exception('Failed to compute throughput from TraCI')
            throughput = 0
            completed_evs = 0

        if not hasattr(self, '_all_seen_evs'):
            self._all_seen_evs = set()

        # average speed across lanes
        speeds = [traci.lane.getLastStepMeanSpeed(lane) for lane in set(lanes)]
        avg_speed = np.mean(speeds) if len(speeds) > 0 else 0.0
        num_lanes = max(1, len(set(lanes)))

        prev_wait = getattr(self, '_last_total_wait', total_wait_time)
        prev_queue = getattr(self, '_last_total_queue', total_queue)
        wait_delta = (prev_wait - total_wait_time) / max(1.0, float(num_lanes) * 60.0)
        queue_delta = (prev_queue - total_queue) / max(1.0, float(num_lanes) * 8.0)

        # Calculate emergency priority metrics
        ev_progress_total = 0.0
        ev_waits = []
        current_evs = set()
        
        try:
            # We want to check all vehicles on the controlled lanes
            for lane in set(lanes):
                for vid in traci.lane.getLastStepVehicleIDs(lane):
                    try:
                        vtype = traci.vehicle.getTypeID(vid).lower()
                        if "emergency" in vtype or "ambulance" in vtype or "police" in vtype or "fire" in vtype:
                            current_evs.add(vid)
                            self._all_seen_evs.add(vid)
                            
                            ev_wait = traci.vehicle.getWaitingTime(vid)
                            ev_waits.append(ev_wait)
                            
                            current_distance = traci.vehicle.getDistance(vid)
                            previous_distance = getattr(self, '_ev_distances', {}).get(vid, current_distance)
                            ev_progress_total += max(0.0, current_distance - previous_distance)
                            
                            if not hasattr(self, '_ev_distances'):
                                self._ev_distances = {}
                            self._ev_distances[vid] = current_distance
                    except Exception:
                        pass
        except Exception:
            pass

        # Cleanup distances for departed EVs
        if hasattr(self, '_ev_distances'):
            active_vids = set(current_evs)
            keys_to_remove = [k for k in self._ev_distances.keys() if k not in active_vids]
            for k in keys_to_remove:
                del self._ev_distances[k]

        worst_emergency_wait = max(ev_waits) if ev_waits else 0.0
        avg_emergency_wait = sum(ev_waits) / len(ev_waits) if ev_waits else 0.0

        # Calculate intersection clearance
        prev_evs = getattr(self, '_prev_evs', set())
        cleared_evs = prev_evs - current_evs
        self._prev_evs = current_evs

        lanes = traci.trafficlight.getControlledLanes(self.controlled_ts) if self.controlled_ts is not None else []
        current_total_wait = sum([traci.lane.getWaitingTime(lane) for lane in set(lanes)]) if lanes else 0.0

        gridlock = (avg_speed < 2.0 and total_queue > num_lanes * 15)
        r_throughput = 30.0 * float(throughput)
        r_queue = -0.25 * float(total_queue)
        # current_total_wait is accumulating, so we penalize the current step's total wait directly
        r_wait = -0.02 * float(current_total_wait)
        r_gridlock = -50.0 if gridlock else 0.0
        r_starvation = -3.0 * (max_lane_wait / 100.0)
        
        # Stale-phase penalty
        r_stale_penalty = 0.0
        same_phase_steps = getattr(self, 'rl_same_phase_steps', 0)
        if same_phase_steps > 120:
            r_stale_penalty = -20.0
        elif same_phase_steps > 60:
            r_stale_penalty = -5.0
        elif same_phase_steps > 30:
            r_stale_penalty = -1.0

        # Emergency Wait Metric tracking for evaluation
        current_emergency_wait = 0.0
        for wait in ev_waits:
            current_emergency_wait += wait
            
        if not hasattr(self, 'cumulative_emergency_delay'):
            self.cumulative_emergency_delay = 0.0
        self.cumulative_emergency_delay += float(current_emergency_wait)

        # Progress-based EV rewards and fairness
        # Note: We completely removed ev_progress to avoid infinite farming. We use milestones instead.
        r_worst_ev_wait = -2.0 * float(worst_emergency_wait)
        r_fairness = -min(1000.0, (float(worst_emergency_wait) - float(avg_emergency_wait)) * 5.0)
        r_ev_clearance = 100.0 * float(len(cleared_evs))
        r_ev_complete = 2000.0 * float(completed_evs)

        reward = (r_throughput + r_queue + r_wait + r_gridlock + r_starvation + r_stale_penalty +
                  r_worst_ev_wait + r_fairness + r_ev_clearance + r_ev_complete)

        self.last_reward_breakdown = {
            'throughput': float(r_throughput),
            'queue': float(r_queue),
            'wait': float(r_wait),
            'gridlock': float(r_gridlock),
            'starvation': float(r_starvation),
            'stale_penalty': float(r_stale_penalty),
            'worst_ev_wait': float(r_worst_ev_wait),
            'fairness': float(r_fairness),
            'ev_clearance': float(r_ev_clearance),
            'ev_complete': float(r_ev_complete)
        }

        self._reward_ema = 0.9 * self._reward_ema + 0.1 * float(reward)
        reward = 0.7 * float(reward) + 0.3 * float(self._reward_ema)

        # Clip reward to a tight range so value estimates stay stable.
        reward = np.clip(reward, -self.reward_clip, self.reward_clip)

        # Track cumulative throughput across the episode
        self.cumulative_throughput += float(throughput)

        # Append per-step reward component breakdown to a CSV for analysis
        try:
            out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'hybrid_logs')
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, 'reward_components.csv')
            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'a') as f:
                if write_header:
                    f.write('step,throughput,queue,wait,gridlock,starvation,stale_penalty,worst_ev_wait,fairness,ev_clearance,ev_complete,reward\n')
                f.write(
                    f"{self.current_step},{r_throughput:.6f},{r_queue:.6f},{r_wait:.6f},{r_gridlock:.6f},{r_starvation:.6f},{r_stale_penalty:.6f},{r_worst_ev_wait:.6f},{r_fairness:.6f},{r_ev_clearance:.6f},{r_ev_complete:.6f},{float(reward):.6f}\n"
                )
        except Exception:
            logger.exception('Failed to append reward component metrics')

        return float(reward)

    def _accumulate_metrics(self):
        # get instantaneous metrics (note: wait is cumulative from SUMO)
        lanes = traci.trafficlight.getControlledLanes(self.controlled_ts) if self.controlled_ts is not None else []
        current_total_wait = sum([traci.lane.getWaitingTime(lane) for lane in set(lanes)]) if lanes else 0.0
        current_queue = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in set(lanes)]) if lanes else 0.0
        speeds = [traci.lane.getLastStepMeanSpeed(lane) for lane in set(lanes)] if lanes else []
        current_avg_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0

        delta_wait = current_total_wait - getattr(self, '_last_total_wait', 0.0)
        if delta_wait < 0:
            delta_wait = current_total_wait

        self.cumulative_wait += float(delta_wait)
        self.cumulative_queue += float(current_queue)
        self.cumulative_speed += float(current_avg_speed)
        self.metric_samples += 1

        self._last_total_wait = float(current_total_wait)
        self._last_total_queue = float(current_queue)

    def get_episode_metrics(self):
        actions = getattr(self, 'action_counts', {0: 0, 1: 0, 2: 0, 3: 0})
        if self.metric_samples == 0:
            return {
                'wait': 0.0,
                'queue': 0.0,
                'avg_speed': 0.0,
                'throughput': 0.0,
                'switches': self.switch_count,
                'reward_ema': self._reward_ema,
                'emergency_delay': 0.0,
                'actions': actions
            }
        return {
            'wait': self.cumulative_wait / self.metric_samples,
            'queue': self.cumulative_queue / self.metric_samples,
            'avg_speed': self.cumulative_speed / self.metric_samples,
            'throughput': self.cumulative_throughput,
            'switches': self.switch_count,
            'reward_ema': self._reward_ema,
            'emergency_delay': getattr(self, 'cumulative_emergency_delay', 0.0),
            'actions': actions
        }

    def get_metrics(self):
        """Return diagnostics for monitoring: wait, queue, avg_speed, throughput, switches"""
        if self.controlled_ts is None:
            return {'wait': 0.0, 'queue': 0.0, 'avg_speed': 0.0, 'throughput': 0.0, 'switches': 0}

        lanes = traci.trafficlight.getControlledLanes(self.controlled_ts)
        total_wait_time = sum([traci.lane.getWaitingTime(lane) for lane in set(lanes)])
        total_queue = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in set(lanes)])
        speeds = [traci.lane.getLastStepMeanSpeed(lane) for lane in set(lanes)]
        avg_speed = np.mean(speeds) if len(speeds) > 0 else 0.0
        return {
            'wait': total_wait_time,
            'queue': total_queue,
            'avg_speed': avg_speed,
            'throughput': 0.0,
            'switches': self.switch_count,
        }

    # Observation normalizer utilities
    def _init_obs_normalizer(self, dim):
        self.obs_dim = dim
        self.obs_mean = np.zeros(self.obs_dim, dtype=np.float64)
        self.obs_var = np.ones(self.obs_dim, dtype=np.float64)
        self.obs_count = 1e-4

    def _update_running_obs(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x
        batch_var = np.zeros_like(x)
        batch_count = 1.0

        delta = batch_mean - self.obs_mean
        tot_count = self.obs_count + batch_count

        new_mean = self.obs_mean + delta * (batch_count / tot_count)
        m_a = self.obs_var * (self.obs_count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * (self.obs_count * batch_count / tot_count)

        self.obs_mean = new_mean
        self.obs_var = M2 / (tot_count)
        self.obs_count = tot_count

    def _normalize_observation(self, obs):
        if obs.shape[0] != getattr(self, 'obs_dim', obs.shape[0]):
            self._init_obs_normalizer(obs.shape[0])

        std = np.sqrt(self.obs_var + 1e-8)
        norm = (obs - self.obs_mean) / std
        self._update_running_obs(obs)
        return norm.astype(np.float32)

    def close(self):
        try:
            traci.close()
        except Exception:
            logger.exception('Failed to close TraCI cleanly')

