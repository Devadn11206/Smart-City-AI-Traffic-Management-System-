import os
import sys
import time
import numpy as np

from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sumolib import checkBinary
import traci

from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent


def get_lane_metrics(ts_id):
    lanes = traci.trafficlight.getControlledLanes(ts_id)
    queue_len = 0
    wait_time = 0
    occupancy = 0
    speed = 0
    vehicles = 0

    for lane in set(lanes):
        queue_len += traci.lane.getLastStepHaltingNumber(lane)
        wait_time += traci.lane.getWaitingTime(lane)
        occupancy += traci.lane.getLastStepOccupancy(lane)
        speed += traci.lane.getLastStepMeanSpeed(lane)
        vehicles += traci.lane.getLastStepVehicleNumber(lane)

    num_lanes = len(set(lanes))
    if num_lanes > 0:
        occupancy /= num_lanes
        speed /= num_lanes

    density = vehicles / max(1, num_lanes)
    congestion = 1 if occupancy > 0.7 else 0

    state = np.array([
        float(vehicles) / 100.0,
        float(queue_len) / 50.0,
        float(wait_time) / 1000.0,
        float(occupancy),
        float(speed) / 20.0,
        float(congestion),
        float(density) / 50.0
    ], dtype=np.float32)
    return state


def try_load(agent, paths):
    for p in paths:
        if os.path.exists(p):
            try:
                agent.load(p)
                print(f"Loaded {agent.__class__.__name__} from {p}")
                return True
            except Exception:
                pass
    return False


def main():
    import argparse

    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument('--config', type=str, default=str((repo_root / 'sumo_network' / 'config.sumocfg').resolve()))
    parser.add_argument('--duration', type=int, default=2000)
    parser.add_argument('--step-interval', type=int, default=5, help='Simulation steps per decision')
    args = parser.parse_args()

    if 'SUMO_HOME' not in os.environ:
        print("Please set SUMO_HOME environment variable and ensure SUMO is installed.")
        return

    sumo_binary = checkBinary('sumo-gui')
    sumo_cmd = [sumo_binary, '-c', args.config, '--no-warnings', 'true', '--random']
    traci.start(sumo_cmd)

    ts_ids = traci.trafficlight.getIDList()
    if not ts_ids:
        print('No traffic lights found in the network.')
        traci.close()
        return

    # Create agent instances and map them to traffic lights in a round-robin fashion
    agent_classes = {
        'dqn': DQNAgent,
        'ddqn': DDQNAgent,
        'ppo': PPOAgent
    }
    preferred_sequence = ['ddqn', 'ppo', 'dqn']

    agents = {}
    model_paths = {
        'dqn': [repo_root / 'models' / 'dqn' / 'dqn_model_best.pth', repo_root / 'results' / 'hybrid_results' / 'hybrid_dqn_model_best.pth'],
        'ddqn': [repo_root / 'models' / 'ddqn' / 'ddqn_model_best.pth', repo_root / 'results' / 'hybrid_results' / 'hybrid_ddqn_model_best.pth'],
        'ppo': [repo_root / 'models' / 'demo' / 'ppo_demo_model.pth', repo_root / 'results' / 'hybrid_results' / 'hybrid_ppo_model_best.pth']
    }

    # instantiate one agent object per type and attempt to load weights (if present)
    instantiated = {}
    for a_type, cls in agent_classes.items():
        try:
            inst = cls(state_dim=7, action_dim=4)
        except TypeError:
            # some agents expect positional args
            inst = cls(7, 4)
        ok = try_load(inst, model_paths.get(a_type, []))
        if not ok:
            print(f"Warning: could not load model for agent type '{a_type}', using uninitialized policy")
        instantiated[a_type] = inst

    # map each ts_id to an agent instance (cycle through preferred_sequence)
    ts_to_agent = {}
    for i, ts in enumerate(ts_ids):
        a_type = preferred_sequence[i % len(preferred_sequence)]
        ts_to_agent[ts] = instantiated[a_type]
        print(f"Assigned {a_type.upper()} to traffic light {ts}")

    current_step = 0
    max_steps = args.duration

    try:
        while current_step < max_steps:
            # decide actions every step-interval steps
            for _ in range(args.step_interval):
                traci.simulationStep()
                current_step += 1
                if current_step >= max_steps:
                    break

            # for each traffic light, observe and act
            for ts in ts_ids:
                agent = ts_to_agent.get(ts)
                if agent is None:
                    continue
                state = get_lane_metrics(ts)
                try:
                    action = agent.act(state, eval_mode=True)
                    if isinstance(action, tuple):
                        action = action[0]
                    # set phase: try to set even phases as green corresponding to action
                    try:
                        current_phase = traci.trafficlight.getPhase(ts)
                        if current_phase != action * 2:
                            traci.trafficlight.setPhase(ts, current_phase + 1)
                            traci.simulationStep()
                            traci.trafficlight.setPhase(ts, action * 2)
                    except Exception:
                        pass
                except Exception as e:
                    # if agent.act fails, skip
                    # print(e)
                    pass

        print('Visualization run complete.')

    finally:
        try:
            traci.close()
        except:
            pass


if __name__ == '__main__':
    main()
