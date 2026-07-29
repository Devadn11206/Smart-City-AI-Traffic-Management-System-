import sys
import os
import argparse
import numpy as np
import torch
import random
import time

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Also add benchmark_research for algorithms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from env.sumo_env_6x6 import SumoMultiAgentEnv6x6
from algorithms.idqn import IDQNAgent
from algorithms.dqn import DQNAgent
from algorithms.ddqn import DDQNAgent
from algorithms.ppo import PPOAgent
from algorithms.presslight import PressLightAgent
from algorithms.colight_6x6 import CoLightAgent

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, choices=['idqn', 'dqn', 'ddqn', 'ppo', 'presslight', 'colight'])
    parser.add_argument('--ep', type=int, required=True, help='Episode checkpoint to load (e.g. 200)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--delay', type=float, default=0.1, help='Delay in seconds between steps to visualize clearly in GUI')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    net_file = os.path.join(base_dir, 'network', 'net.net.xml')
    route_file = os.path.join(base_dir, 'network', 'routes.rou.xml')
    results_dir = os.path.join(base_dir, 'results')
    
    print("Initializing environment in GUI mode...")
    # Force GUI to True for evaluation visualization
    env = SumoMultiAgentEnv6x6(net_file, route_file, use_gui=True, num_seconds=3600, delta_time=10, reward_version='Version_A')
    
    print(f"Initializing {args.algo.upper()} agent...")
    if args.algo == 'idqn':
        agent = IDQNAgent(env.ts_ids)
    elif args.algo == 'dqn':
        agent = DQNAgent()
    elif args.algo == 'ddqn':
        agent = DDQNAgent()
    elif args.algo == 'ppo':
        agent = PPOAgent()
    elif args.algo == 'presslight':
        agent = PressLightAgent()
    elif args.algo == 'colight':
        agent = CoLightAgent(num_nodes=len(env.ts_ids), grid_size=6)
        
    # Load Weights
    checkpoint_path = os.path.join(results_dir, args.algo, f'checkpoint_ep{args.ep}.pth')
    check_path = checkpoint_path.replace(".pth", "_TL1.pth") if args.algo == 'idqn' else checkpoint_path
    
    if os.path.exists(check_path):
        agent.load(checkpoint_path)
        print(f"Successfully loaded {args.algo} from {checkpoint_path}")
    else:
        print(f"Error: Checkpoint {check_path} not found. Cannot evaluate.")
        return

    # Disable epsilon exploration for evaluation
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0.0

    print("Starting evaluation episode...")
    states, _ = env.reset(seed=args.seed)
    
    done = False
    ep_rewards = {ts: 0 for ts in env.ts_ids}
    ep_waits = {ts: [] for ts in env.ts_ids}
    ep_queues = {ts: [] for ts in env.ts_ids}
    ep_throughputs = {ts: 0 for ts in env.ts_ids}
    
    step_idx = 0
    
    while not done:
        # Action Selection (Evaluation Mode)
        if args.algo == 'idqn':
            actions = agent.act(states)
        elif args.algo == 'colight':
            state_list = [states[ts] for ts in env.ts_ids]
            acts = agent.act(state_list)
            actions = {ts: act for ts, act in zip(env.ts_ids, acts)}
        elif args.algo == 'ppo':
            actions = {}
            for ts in env.ts_ids:
                a, _, _ = agent.act(states[ts])
                actions[ts] = a
        else:
            actions = {ts: agent.act(states[ts]) for ts in env.ts_ids}
            
        next_states, rewards, dones, _, infos = env.step(actions)
        
        # Logging Metrics (No training)
        for ts in env.ts_ids:
            ep_rewards[ts] += rewards[ts]
            ep_waits[ts].append(infos[ts]['wait'])
            ep_queues[ts].append(infos[ts]['queue'])
            ep_throughputs[ts] += infos[ts]['throughput']
            
        states = next_states
        done = dones[env.ts_ids[0]]
        step_idx += 1
        
        # Add a tiny delay so the user can actually see what is happening in the GUI
        time.sleep(args.delay)
        
    global_infos = infos['global']
    print(f"\n--- EVALUATION COMPLETE ---")
    print(f"Algorithm: {args.algo.upper()} (Checkpoint Ep {args.ep})")
    print(f"Total Reward: {sum(ep_rewards.values()):.2f}")
    print(f"Average Waiting Time: {np.mean([np.mean(w) for w in ep_waits.values()]):.2f}s")
    print(f"Average Queue Length: {np.mean([np.mean(q) for q in ep_queues.values()]):.2f} veh")
    print(f"Total Throughput: {sum(ep_throughputs.values())} veh")
    print(f"Average Travel Time: {global_infos['avg_travel_time']:.2f}s")
    print(f"Fuel Consumption: {global_infos['fuel_consumption']:.2f} ml")
    
if __name__ == '__main__':
    main()
