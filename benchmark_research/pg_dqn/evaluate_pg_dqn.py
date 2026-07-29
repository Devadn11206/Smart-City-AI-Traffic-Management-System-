import os
import sys
import yaml
import time
import numpy as np
import torch
import csv

# Paths setup
current_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.dirname(current_dir)
if benchmark_dir not in sys.path:
    sys.path.append(benchmark_dir)

env_dir = os.path.join(benchmark_dir, '6x6_benchmark')
if env_dir not in sys.path:
    sys.path.append(env_dir)

from pg_dqn.dqn_agent import AC_PG_DQNAgent
from env.sumo_env_6x6 import SumoMultiAgentEnv6x6  # type: ignore

def evaluate(model_path, num_episodes=5):
    with open(os.path.join(current_dir, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Env
    net_file = os.path.join(benchmark_dir, '6x6_benchmark', 'network', 'net.net.xml')
    route_file = os.path.join(benchmark_dir, '6x6_benchmark', 'network', 'routes.rou.xml')
    env = SumoMultiAgentEnv6x6(net_file=net_file, route_file=route_file, use_gui=False, num_seconds=config['training']['max_steps']*10)
    
    # Init Agent (Set epsilon to 0 for greedy evaluation)
    agent = AC_PG_DQNAgent(
        state_dim=config['environment']['state_dim'],
        action_dim=config['environment']['action_dim'],
        physics_dim=config['environment']['physics_dim'],
        lr=config['training']['lr'],
        gamma=config['training']['gamma'],
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1.0,
        device=device
    )
    
    agent.q_network.load_state_dict(torch.load(model_path, map_location=device))
    agent.q_network.eval()
    
    results = []
    
    for episode in range(1, num_episodes + 1):
        states, _ = env.reset()
        episode_reward = 0
        last_infos = None
        done = False
        
        while not done:
            ts_ids = list(states.keys())
            states_np = np.array([states[ts] for ts in ts_ids])
            
            pinn_preds = agent.get_pinn_predictions(env, ts_ids, last_infos)
            
            states_tensor = torch.FloatTensor(states_np).to(device)
            pinn_tensor = torch.FloatTensor(pinn_preds).to(device)
            
            actions_list = agent.act(states_tensor, pinn_tensor)
            actions = {ts_ids[i]: actions_list[i] for i in range(len(ts_ids))}
            
            next_states, rewards, dones, _, infos = env.step(actions)
            
            for ts in ts_ids:
                episode_reward += rewards[ts]
                
            states = next_states
            last_infos = infos
            
            if all(dones.values()):
                done = True
                
        global_info = infos.get('global', {})
        avg_wait = np.mean([infos[ts]['wait'] for ts in ts_ids])
        queue = np.sum([infos[ts]['queue'] for ts in ts_ids])
        tp = np.sum([infos[ts]['throughput'] for ts in ts_ids])
        
        print(f"Eval Ep {episode} | Reward: {episode_reward:.2f} | Wait: {avg_wait:.2f} | Q: {queue:.2f} | Tp: {tp:.2f}")
        results.append((episode_reward, avg_wait, queue, tp))
        
    env.close()
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_pg_dqn.py <path_to_model.pth>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    evaluate(model_path)
