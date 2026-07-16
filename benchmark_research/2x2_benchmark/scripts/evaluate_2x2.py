import sys
import os
import argparse
import numpy as np
import torch
import random

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from env.sumo_env import SumoMultiAgentEnv
from algorithms.idqn import IDQNAgent
from algorithms.dqn import DQNAgent
from algorithms.ddqn import DDQNAgent
from algorithms.ppo import PPOAgent
from algorithms.presslight import PressLightAgent
from algorithms.colight import CoLightAgent

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, choices=['idqn', 'dqn', 'ddqn', 'ppo', 'presslight', 'colight'])
    parser.add_argument('--checkpoint', type=int, required=True, help="Episode number of the checkpoint to load")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    net_file = os.path.join(base_dir, 'network', 'net.net.xml')
    route_file = os.path.join(base_dir, 'network', 'routes.rou.xml')
    results_dir = os.path.join(base_dir, 'results')
    
    reward_version = 'original' if args.algo in ['presslight', 'colight'] else 'Version_A'
    env = SumoMultiAgentEnv(net_file, route_file, use_gui=args.gui, num_seconds=3600, delta_time=10, reward_version=reward_version)
    
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
        agent = CoLightAgent(num_nodes=len(env.ts_ids))
        
    checkpoint_path = os.path.join(results_dir, args.algo, f'checkpoint_ep{args.checkpoint}.pth')
    check_path = checkpoint_path.replace(".pth", "_TL1.pth") if args.algo == 'idqn' else checkpoint_path
    
    if os.path.exists(check_path):
        agent.load(checkpoint_path)
        print(f"Loaded {args.algo} from {checkpoint_path}")
    else:
        print(f"Error: Checkpoint {check_path} not found.")
        return

    print(f"Starting Evaluation for {args.algo} (Checkpoint {args.checkpoint})...")
    states, _ = env.reset(seed=args.seed)
    
    # Warmup for 300 seconds
    for _ in range(300 // env.delta_time):
        warmup_actions = {ts: random.randint(0, 3) for ts in env.ts_ids}
        states, _, _, _, _ = env.step(warmup_actions)
        
    done = False
    total_reward = 0
    
    while not done:
        # Action Selection (Evaluation mode)
        if args.algo == 'idqn':
            actions = agent.act(states, evaluate=True)
        elif args.algo == 'colight':
            state_list = [states[ts] for ts in env.ts_ids]
            acts = agent.act(state_list, evaluate=True)
            actions = {ts: act for ts, act in zip(env.ts_ids, acts)}
        elif args.algo == 'ppo':
            actions = {}
            for ts in env.ts_ids:
                a, _, _ = agent.act(states[ts], evaluate=True)
                actions[ts] = a
        else:
            actions = {ts: agent.act(states[ts], evaluate=True) for ts in env.ts_ids}
            
        next_states, rewards, dones, _, infos = env.step(actions)
        
        # PressLight custom reward
        if args.algo == 'presslight':
            for ts in env.ts_ids:
                rewards[ts] = -abs(infos[ts]['pressure'])
                
        total_reward += sum(rewards.values())
        states = next_states
        done = dones[env.ts_ids[0]]
        
    print(f"Evaluation completed. Total Reward: {total_reward:.2f}")

if __name__ == '__main__':
    main()
