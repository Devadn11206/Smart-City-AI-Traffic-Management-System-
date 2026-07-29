import os
import sys
import numpy as np
import traci
import time
from sumolib import checkBinary
from collections import deque
import logging

logger = logging.getLogger(__name__)

class SumoEnvironment6x6:
    """
    Multi-Agent SUMO Environment for a 6x6 grid network.
    Manages 36 distinct traffic lights, returning states and rewards as dictionaries.
    """
    def __init__(self, config_file, use_gui=False, num_seconds=3600,
                 min_green=10, max_green=50, action_repeat=5, seed=42):
        self.config_file = config_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds

        self.min_green = min_green
        self.max_green = max_green
        self.action_repeat = max(1, int(action_repeat))
        self.base_seed = seed

        if 'SUMO_HOME' in os.environ:
            sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
        else:
            sys.exit("Please declare environment variable 'SUMO_HOME'")

        self.sumo_binary = checkBinary('sumo-gui') if self.use_gui else checkBinary('sumo')
        self.ts_ids = []
        self.current_step = 0

        self.state_dim = 15
        self.action_dim = 4  # 4 phases per intersection
        
        self.ts_state_history = {} # store phase persistence etc per ts

    def start_simulation(self, seed=None):
        abs_config = os.path.abspath(self.config_file)
        sumo_cmd = [self.sumo_binary, "-c", abs_config, "--no-warnings", "true"]
        if seed is not None:
            sumo_cmd += ["--seed", str(seed)]

        try:
            if traci.isLoaded(): traci.close(False)
        except Exception: pass

        for attempt in range(5):
            try:
                traci.start(sumo_cmd)
                break
            except Exception as e:
                time.sleep(1)

        # Initialize controlled traffic lights
        all_ts_ids = traci.trafficlight.getIDList()
        # Keep only actual traffic lights (some nodes might be dummy)
        self.ts_ids = [ts for ts in all_ts_ids if len(traci.trafficlight.getCompleteRedYellowGreenDefinition(ts)[0].phases) > 0]
        
        for ts_id in self.ts_ids:
            self.ts_state_history[ts_id] = {
                'last_action': None,
                'last_change_step': -10**9,
                'phase_persistence': 0.0,
                'switch_count': 0
            }
            
        self._last_arrived = traci.simulation.getArrivedNumber()

    def reset(self, seed=None):
        if self.current_step != 0:
            try: traci.close()
            except Exception: pass

        self.start_simulation(seed=self.base_seed if seed is None else seed)
        self.current_step = 0
        self.cumulative_wait = 0.0
        self.cumulative_queue = 0.0
        self.cumulative_speed = 0.0
        self.metric_samples = 0
        self.cumulative_fuel = 0.0
        self.cumulative_throughput = 0
        self.vehicle_depart_times = {}
        self.cumulative_travel_time = 0.0
        self.arrived_vehicles_count = 0
        
        return self._get_all_states()

    def step(self, action_dict):
        """
        action_dict: {ts_id: action_int}
        """
        for ts_id, action in action_dict.items():
            self._set_phase(ts_id, action)

        for _ in range(self.action_repeat):
            traci.simulationStep()
            self.current_step += 1

        self._accumulate_metrics()
        next_states = self._get_all_states()
        rewards = {ts_id: self._compute_reward(ts_id) for ts_id in self.ts_ids}
        done = self.current_step >= self.num_seconds

        return next_states, rewards, done, {}

    def _set_phase(self, ts_id, action):
        hist = self.ts_state_history[ts_id]
        
        current_phase = traci.trafficlight.getPhase(ts_id)
        phase_count = max(1, len(traci.trafficlight.getCompleteRedYellowGreenDefinition(ts_id)[0].phases))
        target_phase = (int(action) * 2) % phase_count

        if target_phase == current_phase:
            hist['phase_persistence'] = min(1.0, hist['phase_persistence'] + 0.1)
            return

        if self.current_step - hist['last_change_step'] < self.min_green:
            hist['phase_persistence'] = 0.0
            return

        yellow_phase = min(current_phase + 1, phase_count - 1)
        traci.trafficlight.setPhase(ts_id, yellow_phase)
        
        # We don't advance the simulation here for yellow lights in MARL to keep agents synchronized.
        # In a strict environment we'd queue the yellow phases, but for a simplified MARL we instantly swap.
        traci.trafficlight.setPhase(ts_id, target_phase)
        
        hist['last_change_step'] = self.current_step
        hist['switch_count'] += 1
        hist['last_action'] = int(action)
        hist['phase_persistence'] = 0.0

    def _get_all_states(self):
        states = {}
        for ts_id in self.ts_ids:
            states[ts_id] = self._get_state(ts_id)
        return states

    def _get_state(self, ts_id):
        lanes = list(set(traci.trafficlight.getControlledLanes(ts_id)))
        state_str = traci.trafficlight.getRedYellowGreenState(ts_id)
        links = traci.trafficlight.getControlledLinks(ts_id)

        green_lanes, red_lanes = set(), set()
        for i, link_info in enumerate(links):
            if not link_info: continue
            lane = link_info[0][0]
            if i < len(state_str) and state_str[i].lower() == 'g':
                green_lanes.add(lane)
            else:
                red_lanes.add(lane)
        red_lanes = red_lanes - green_lanes

        g_veh, r_veh, g_wait, r_wait = 0, 0, 0, 0
        for lane in lanes:
            if lane in green_lanes:
                g_veh += traci.lane.getLastStepVehicleNumber(lane)
                g_wait += traci.lane.getWaitingTime(lane)
            else:
                r_veh += traci.lane.getLastStepVehicleNumber(lane)
                r_wait += traci.lane.getWaitingTime(lane)

        current_phase = traci.trafficlight.getPhase(ts_id)
        phase_count = max(1, len(traci.trafficlight.getCompleteRedYellowGreenDefinition(ts_id)[0].phases))
        norm_phase = float(current_phase) / float(phase_count)

        # Simplified 15-dim state to match the old agent architecture (padded with 0s for missing complex metrics)
        state = np.array([
            norm_phase,
            self.ts_state_history[ts_id]['phase_persistence'],
            float(g_veh) / 50.0,
            float(r_veh) / 50.0,
            float(g_wait) / 1000.0,
            float(r_wait) / 1000.0,
            float(g_veh) / max(1, len(green_lanes)) / 50.0,
            float(r_veh) / max(1, len(red_lanes)) / 50.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 # EV and Congestion placeholders
        ], dtype=np.float32)

        return state

    def _compute_reward(self, ts_id):
        lanes = set(traci.trafficlight.getControlledLanes(ts_id))
        total_wait = sum([traci.lane.getWaitingTime(lane) for lane in lanes])
        total_queue = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in lanes])
        
        # Penalize total wait and total queue
        reward = -0.02 * float(total_wait) - 0.25 * float(total_queue)
        return float(reward)

    def _accumulate_metrics(self):
        lanes = []
        for ts_id in self.ts_ids:
            lanes.extend(traci.trafficlight.getControlledLanes(ts_id))
        lanes = list(set(lanes))
        
        current_total_wait = sum([traci.lane.getWaitingTime(lane) for lane in lanes])
        current_queue = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in lanes])
        speeds = [traci.lane.getLastStepMeanSpeed(lane) for lane in lanes]
        current_avg_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0

        if not hasattr(self, 'cumulative_wait'):
            self.cumulative_wait = 0.0
            self.cumulative_queue = 0.0
            self.cumulative_speed = 0.0
            self.metric_samples = 0
            self.cumulative_fuel = 0.0
            self.cumulative_throughput = 0
            self.vehicle_depart_times = {}
            self.cumulative_travel_time = 0.0
            self.arrived_vehicles_count = 0

        self.cumulative_wait += current_total_wait
        self.cumulative_queue += current_queue
        self.cumulative_speed += current_avg_speed
        
        try:
            fuel = 0.0
            for lane in lanes:
                for vid in traci.lane.getLastStepVehicleIDs(lane):
                    fuel += traci.vehicle.getFuelConsumption(vid)
            self.cumulative_fuel += fuel
        except:
            pass

        self.metric_samples += 1
        
        try:
            arrived = traci.simulation.getArrivedNumber()
            self.cumulative_throughput += arrived
            
            for vid in traci.simulation.getDepartedIDList():
                self.vehicle_depart_times[vid] = self.current_step
                
            for vid in traci.simulation.getArrivedIDList():
                depart_time = self.vehicle_depart_times.pop(vid, None)
                if depart_time is not None:
                    travel_time = self.current_step - depart_time
                    self.cumulative_travel_time += travel_time
                    self.arrived_vehicles_count += 1
        except:
            pass

    def get_episode_metrics(self):
        avg_travel_time = self.cumulative_travel_time / max(1, self.arrived_vehicles_count) if hasattr(self, 'arrived_vehicles_count') else 0.0
            
        if getattr(self, 'metric_samples', 0) == 0:
            return {'wait': 0.0, 'queue': 0.0, 'avg_speed': 0.0, 'throughput': getattr(self, 'cumulative_throughput', 0), 'fuel': 0.0, 'travel_time': avg_travel_time}
            
        return {
            'wait': self.cumulative_wait / self.metric_samples,
            'queue': self.cumulative_queue / self.metric_samples,
            'avg_speed': self.cumulative_speed / self.metric_samples,
            'throughput': self.cumulative_throughput,
            'fuel': self.cumulative_fuel / self.metric_samples,
            'travel_time': avg_travel_time
        }

    def close(self):
        try: traci.close()
        except Exception: pass
