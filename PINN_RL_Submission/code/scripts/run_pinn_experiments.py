import os
# Force CPU execution to prevent CUDA Out of Memory errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import numpy as np
import torch
import random
import time

# Add benchmark_research to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
# Add 6x6_benchmark to path to import env
sys.path.insert(0, os.path.join(base_dir, '6x6_benchmark'))

from env.sumo_env_6x6 import SumoMultiAgentEnv6x6  # type: ignore

import importlib.util
logger_spec = importlib.util.spec_from_file_location("logger_6x6", os.path.join(base_dir, '6x6_benchmark', 'utils', 'logger_6x6.py'))
logger_module = importlib.util.module_from_spec(logger_spec)
logger_spec.loader.exec_module(logger_module)
MetricLogger = logger_module.MetricLogger
from algorithms.ppo import PPOAgent
from algorithms.ppo_pinn import PINNStateWrapper

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_experiment(mode, algo_name, episodes=200, seed=42):
    print(f"\n{'='*50}\nStarting Experiment: {algo_name} (Mode: {mode})\n{'='*50}")
    set_seed(seed)
    
    net_file = os.path.join(base_dir, '6x6_benchmark', 'network', 'net.net.xml')
    route_file = os.path.join(base_dir, '6x6_benchmark', 'network', 'routes.rou.xml')
    results_dir = os.path.join(base_dir, 'results')
    
    reward_version = 'Version_A'
    env = SumoMultiAgentEnv6x6(net_file, route_file, use_gui=False, num_seconds=3600, delta_time=10, reward_version=reward_version)
    logger = MetricLogger(results_dir, algo_name)
    
    # Check if experiment is already done (approximate check)
    metrics_file = os.path.join(results_dir, algo_name, '6x6_episode_metrics.csv')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > episodes:
                print(f"Experiment {algo_name} already completed ({len(lines)-1} episodes found). Skipping.")
                env.close()
                return

    metadata = {
        "reward_version": reward_version,
        "reward_formula": "(Throughput - 0.1 * QueueLength - 0.1 * SwitchPenalty) / 100",
        "network_size": "6x6",
        "episodes": episodes,
        "seed": seed,
        "decision_interval": 10,
        "algorithm": algo_name,
        "pinn_mode": mode
    }
    logger.log_metadata(metadata)
    
    pinn_wrapper = PINNStateWrapper(mode=mode, device="cuda" if torch.cuda.is_available() else "cpu")
    state_dim = pinn_wrapper.get_augmented_state_dim()
    
    # IMPORTANT: instantiate PPO with the newly augmented state dimension!
    agent = PPOAgent(state_dim=state_dim, action_dim=4)
    
    print(f"Initialized PPO Agent with state_dim={state_dim}")
    
    for ep in range(1, episodes + 1):
        original_states, _ = env.reset(seed=seed + ep)
        pinn_wrapper.reset()
        
        # Warmup for 300 seconds
        for _ in range(300 // env.delta_time):
            warmup_actions = {ts: random.randint(0, 3) for ts in env.ts_ids}
            original_states, _, _, _, _ = env.step(warmup_actions)
            
        done = False
        ep_rewards = {ts: 0 for ts in env.ts_ids}
        ep_waits = {ts: [] for ts in env.ts_ids}
        ep_queues = {ts: [] for ts in env.ts_ids}
        ep_throughputs = {ts: 0 for ts in env.ts_ids}
        
        last_infos = None
        
        while not done:
            actions = {}
            probs = {}
            vals = {}
            augmented_states = {}
            
            for ts in env.ts_ids:
                aug_state = pinn_wrapper.extract_and_predict(ts, env, last_infos, original_states[ts])
                augmented_states[ts] = aug_state
                
                action, prob, val = agent.act(aug_state)
                actions[ts] = action
                probs[ts] = prob
                vals[ts] = val
                
            next_original_states, rewards, dones, _, infos = env.step(actions)
            last_infos = infos
            
            done = all(dones.values())
            
            for ts in env.ts_ids:
                agent.step(ts, augmented_states[ts], actions[ts], probs[ts], vals[ts], rewards[ts], dones[ts])
                
                ep_rewards[ts] += rewards[ts]
                ep_waits[ts].append(infos[ts]['wait'])
                ep_queues[ts].append(infos[ts]['queue'])
                ep_throughputs[ts] += infos[ts]['throughput']
                
            original_states = next_original_states
            
        # End of episode training
        train_stats = agent.train()
        
        # Logging
        avg_reward = np.mean(list(ep_rewards.values()))
        
        ep_metrics = {
            'episode': ep,
            'reward': avg_reward,
            'avg_queue_length': np.mean([np.mean(ep_queues[ts]) for ts in env.ts_ids]),
            'avg_waiting_time': np.mean([np.mean(ep_waits[ts]) for ts in env.ts_ids]),
            'throughput': sum(ep_throughputs.values()),
            'avg_speed': np.mean([infos[ts]['speed'] for ts in env.ts_ids]),
            'fuel_consumption': infos['global']['fuel_consumption'],
            'avg_travel_time': infos['global']['avg_travel_time'],
            'loss': train_stats.get('loss', 0.0)
        }
        
        if ep % 10 == 0 or ep == 1:
            print(f"Ep {ep}/{episodes} | R: {avg_reward:.2f} | Q: {ep_metrics['avg_queue_length']:.2f} | W: {ep_metrics['avg_waiting_time']:.2f} | TP: {ep_metrics['throughput']} | Spd: {ep_metrics['avg_speed']:.2f} | Fuel: {ep_metrics['fuel_consumption']:.0f}")
            
        logger.log_episode(ep_metrics)
        
        if ep % 50 == 0:
            agent.save(os.path.join(results_dir, algo_name, f'checkpoint_ep{ep}.pth'))
            
    print(f"Average PINN Inference Time per step: {pinn_wrapper.get_avg_inference_time()*1000:.2f} ms")
    env.close()

if __name__ == '__main__':
    import argparse
    import multiprocessing
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--parallel', action='store_true', help='Run experiments in parallel')
    parser.add_argument('--mode', type=str, help='Specific PINN mode to run (e.g. baseline, queue)')
    parser.add_argument('--algo', type=str, help='Specific folder name to save to (e.g. ppo, ppo_pinn)')
    args = parser.parse_args()

    if args.mode and args.algo:
        run_experiment(mode=args.mode, algo_name=args.algo, episodes=args.episodes, seed=42)
        print(f"Completed {args.algo}\n")
    else:
        experiments = [
            ('baseline', 'ppo'),
            ('queue', 'ppo_pinn_queue'),
            ('queue_density', 'ppo_pinn_queue_density'),
            ('all', 'ppo_pinn')
        ]
        
        if args.parallel:
            print("Starting experiments in PARALLEL mode. Monitor CPU/RAM usage.")
            processes = []
            for mode, algo_name in experiments:
                p = multiprocessing.Process(target=run_experiment, args=(mode, algo_name, args.episodes, 42))
                p.start()
                processes.append(p)
                
            for p in processes:
                p.join()
            print("All parallel experiments completed!")
        else:
            for mode, algo_name in experiments:
                run_experiment(mode=mode, algo_name=algo_name, episodes=args.episodes, seed=42)
                print(f"Completed {algo_name}\n")
