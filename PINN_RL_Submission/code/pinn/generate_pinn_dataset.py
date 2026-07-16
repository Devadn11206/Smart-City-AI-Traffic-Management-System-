import sys
import os
import argparse
import numpy as np
import torch
import csv
import traci

# Add parent dir to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, '6x6_benchmark'))

# pyrefly: ignore [missing-import]
from env.sumo_env_6x6 import SumoMultiAgentEnv6x6
from algorithms.idqn import IDQNAgent
from algorithms.dqn import DQNAgent
from algorithms.ddqn import DDQNAgent
from algorithms.ppo import PPOAgent

def get_latest_checkpoint(algo, results_dir):
    algo_dir = os.path.join(results_dir, algo)
    if not os.path.exists(algo_dir): return None
    import glob
    checkpoints = glob.glob(os.path.join(algo_dir, 'checkpoint_ep*.pth'))
    checkpoints = [c for c in checkpoints if 'TL' not in c] # filter out IDQN individual
    if not checkpoints: return None
    # sort by episode number
    checkpoints.sort(key=lambda x: int(os.path.basename(x).split('ep')[1].split('.pth')[0]))
    return checkpoints[-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes_per_algo', type=int, default=3)
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(root_dir, '6x6_benchmark')
    net_file = os.path.join(base_dir, 'network', 'net.net.xml')
    route_file = os.path.join(base_dir, 'network', 'routes.rou.xml')
    results_dir = os.path.join(base_dir, 'results')
    
    historical_dir = os.path.join(root_dir, 'historical_data')
    os.makedirs(historical_dir, exist_ok=True)
    out_file = os.path.join(historical_dir, 'traffic_history_6x6.csv')

    algos = ['idqn', 'dqn', 'ddqn', 'ppo']
    
    env = SumoMultiAgentEnv6x6(net_file, route_file, use_gui=args.gui, num_seconds=3600, delta_time=10)
    
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'algorithm', 'episode', 'simulation_time', 'step_number', 'intersection_id',
            'traffic_density', 'queue_length', 'throughput', 'avg_speed', 'fuel_consumption',
            'current_phase', 'phase_duration'
        ])
        
        for algo in algos:
            checkpoint = get_latest_checkpoint(algo, results_dir)
            if not checkpoint and algo != 'idqn':
                print(f"No checkpoint found for {algo}, skipping...")
                continue
                
            print(f"Collecting data using {algo.upper()}...")
            
            if algo == 'idqn':
                agent = IDQNAgent(env.ts_ids)
                # Load idqn weights manually if checkpoint logic differs
                # We will attempt to load the individual files
                check_path = os.path.join(results_dir, algo, 'checkpoint_ep50_TL1.pth') # fallback
                import glob
                last_files = glob.glob(os.path.join(results_dir, algo, '*_TL1.pth'))
                if last_files:
                    last_files.sort(key=lambda x: int(os.path.basename(x).split('ep')[1].split('_TL')[0]))
                    ep_num = int(os.path.basename(last_files[-1]).split('ep')[1].split('_TL')[0])
                    for ts in env.ts_ids:
                        p = os.path.join(results_dir, algo, f'checkpoint_ep{ep_num}_{ts}.pth')
                        if os.path.exists(p):
                            agent.agents[ts].load(p)
            elif algo == 'dqn':
                agent = DQNAgent()
                agent.load(checkpoint)
            elif algo == 'ddqn':
                agent = DDQNAgent()
                agent.load(checkpoint)
            elif algo == 'ppo':
                agent = PPOAgent()
                agent.load(checkpoint)
            
            for ep in range(1, args.episodes_per_algo + 1):
                print(f"  Starting episode {ep}...")
                states, _ = env.reset(seed=42+ep)
                print(f"  Reset done. Warming up...")
                
                # Warmup
                for _ in range(300 // env.delta_time):
                    states, _, _, _, _ = env.step({ts: 0 for ts in env.ts_ids})
                print(f"  Warmup done. Simulating...")
                
                done = False
                step_idx = 0
                sim_time = 300
                
                # Manual phase tracking since TraCI might not give exact duration
                tracked_phase = {ts: -1 for ts in env.ts_ids}
                tracked_duration = {ts: 0 for ts in env.ts_ids}
                
                last_infos = None
                while not done:
                    if step_idx % 50 == 0:
                        print(f"    Simulating step {step_idx} (sim time: {sim_time}s)...")
                    # Action Selection (inference only)
                    if algo == 'idqn':
                        # disable eps-greedy
                        actions = {}
                        for ts in env.ts_ids:
                            s = torch.FloatTensor(states[ts]).unsqueeze(0).to(agent.agents[ts].device)
                            with torch.no_grad():
                                actions[ts] = agent.agents[ts].q_network(s).max(1)[1].item()
                    elif algo == 'ppo':
                        actions = {}
                        for ts in env.ts_ids:
                            a, _, _ = agent.act(states[ts], evaluate=True)
                            actions[ts] = a
                    else:
                        # DQN / DDQN
                        actions = {}
                        for ts in env.ts_ids:
                            s = torch.FloatTensor(states[ts]).unsqueeze(0).to(agent.device)
                            with torch.no_grad():
                                actions[ts] = agent.q_network(s).max(1)[1].item()
                    
                    # Instead of trusting env.step to return raw values, we grab them via TraCI BEFORE stepping
                    raw_data = {}
                    for ts in env.ts_ids:
                        lanes = env.tl_in_lanes[ts]
                        all_lanes = [l for d in lanes.values() for l in d]
                        
                        if len(all_lanes) > 0:
                            density = sum([traci.lane.getLastStepVehicleNumber(l) for l in all_lanes]) / (sum([traci.lane.getLength(l) for l in all_lanes]) + 1e-5)
                            queue = sum([traci.lane.getLastStepHaltingNumber(l) for l in all_lanes])
                            speed = np.mean([traci.lane.getLastStepMeanSpeed(l) for l in all_lanes])
                            fuel = sum([traci.lane.getFuelConsumption(l) for l in all_lanes])
                            
                            # Correct throughput extraction: from the previous step's infos
                            if last_infos is not None and ts in last_infos:
                                throughput = last_infos[ts]['throughput']
                            else:
                                throughput = 0
                            
                            phase = traci.trafficlight.getPhase(ts)
                            if phase == tracked_phase[ts]:
                                tracked_duration[ts] += env.delta_time
                            else:
                                tracked_phase[ts] = phase
                                tracked_duration[ts] = env.delta_time
                            duration = tracked_duration[ts]
                            
                            raw_data[ts] = {
                                'density': density,
                                'queue': queue,
                                'throughput': throughput,
                                'speed': speed,
                                'fuel': fuel,
                                'phase': phase,
                                'duration': duration
                            }

                    # Log data
                    for ts in env.ts_ids:
                        if ts in raw_data:
                            d = raw_data[ts]
                            writer.writerow([
                                algo.upper(), ep, sim_time, step_idx, ts,
                                d['density'], d['queue'], d['throughput'], d['speed'], d['fuel'],
                                d['phase'], d['duration']
                            ])
                    
                    next_states, rewards, dones, _, infos = env.step(actions)
                    last_infos = infos
                    states = next_states
                    done = dones[env.ts_ids[0]]
                    step_idx += 1
                    sim_time += env.delta_time
                    
    env.close()
    print(f"Data collection complete! Saved to {out_file}")

if __name__ == '__main__':
    main()
