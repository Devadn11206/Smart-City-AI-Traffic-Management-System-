import sys
import os
import argparse
import numpy as np
import torch
import random

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Also add benchmark_research for algorithms


from env.sumo_env_6x6 import SumoMultiAgentEnv6x6
from algorithms.idqn import IDQNAgent
from algorithms.dqn import DQNAgent
from algorithms.ddqn import DDQNAgent
from algorithms.ppo import PPOAgent
from algorithms.presslight import PressLightAgent
from algorithms.colight_6x6 import CoLightAgent
from utils.logger_6x6 import MetricLogger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, choices=['idqn', 'dqn', 'ddqn', 'ppo', 'presslight', 'colight'])
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--resume_ep', type=int, default=0)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    net_file = os.path.join(base_dir, 'network', 'net.net.xml')
    route_file = os.path.join(base_dir, 'network', 'routes.rou.xml')
    results_dir = os.path.join(base_dir, 'results')
    
    reward_version = 'Version_A'
    env = SumoMultiAgentEnv6x6(net_file, route_file, use_gui=args.gui, num_seconds=3600, delta_time=10, reward_version=reward_version)
    logger = MetricLogger(results_dir, args.algo)
    
    metadata = {
        "reward_version": reward_version,
        "reward_formula": "(Throughput - 0.1 * QueueLength - 0.1 * SwitchPenalty) / 100" if reward_version == 'Version_A' else "original",
        "network_size": "6x6",
        "episodes": args.episodes,
        "seed": args.seed,
        "decision_interval": 10,
        "algorithm": args.algo
    }
    logger.log_metadata(metadata)
    
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
        
    checkpoints = [1, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    
    start_ep = 1
    if args.resume_ep > 0:
        checkpoint_path = os.path.join(results_dir, args.algo, f'checkpoint_ep{args.resume_ep}.pth')
        check_path = checkpoint_path.replace(".pth", "_TL1.pth") if args.algo == 'idqn' else checkpoint_path
        if os.path.exists(check_path):
            agent.load(checkpoint_path)
            start_ep = args.resume_ep + 1
            print(f"Resumed {args.algo} from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint {check_path} not found.")

    for ep in range(start_ep, args.episodes + 1):
        states, _ = env.reset(seed=args.seed + ep)
        
        # Warmup for 300 seconds
        for _ in range(300 // env.delta_time):
            warmup_actions = {ts: random.randint(0, 3) for ts in env.ts_ids}
            states, _, _, _, _ = env.step(warmup_actions)
            
        done = False
        ep_rewards = {ts: 0 for ts in env.ts_ids}
        ep_waits = {ts: [] for ts in env.ts_ids}
        ep_queues = {ts: [] for ts in env.ts_ids}
        ep_throughputs = {ts: 0 for ts in env.ts_ids}
        
        losses = []
        step_idx = 0
        
        ppo_actions = {}
        ppo_probs = {}
        ppo_vals = {}
        
        while not done:
            # Action Selection
            if args.algo == 'idqn':
                actions = agent.act(states)
            elif args.algo == 'colight':
                state_list = [states[ts] for ts in env.ts_ids]
                acts = agent.act(state_list)
                actions = {ts: act for ts, act in zip(env.ts_ids, acts)}
            elif args.algo == 'ppo':
                actions = {}
                for ts in env.ts_ids:
                    a, p, v = agent.act(states[ts])
                    actions[ts] = a
                    ppo_probs[ts] = p
                    ppo_vals[ts] = v
            else:
                actions = {ts: agent.act(states[ts]) for ts in env.ts_ids}
                
            next_states, rewards, dones, _, infos = env.step(actions)
            
            # PressLight custom reward
            if args.algo == 'presslight':
                for ts in env.ts_ids:
                    rewards[ts] = -abs(infos[ts]['pressure']) / 100.0
            
            # Step / Store Memory
            if args.algo == 'idqn':
                agent.step(states, actions, rewards, next_states, dones)
                loss_dict = agent.train()
                if loss_dict and list(loss_dict.values())[0] > 0:
                    losses.append(np.mean(list(loss_dict.values())))
            elif args.algo == 'colight':
                state_list = [states[ts] for ts in env.ts_ids]
                next_state_list = [next_states[ts] for ts in env.ts_ids]
                action_list = [actions[ts] for ts in env.ts_ids]
                reward_list = [rewards[ts] for ts in env.ts_ids]
                agent.step(state_list, action_list, reward_list, next_state_list, dones[env.ts_ids[0]])
                loss = agent.train()
                if loss > 0: losses.append(loss)
            elif args.algo == 'ppo':
                for ts in env.ts_ids:
                    agent.step(ts, states[ts], actions[ts], ppo_probs[ts], ppo_vals[ts], rewards[ts], dones[ts])
            else:
                for ts in env.ts_ids:
                    agent.step(states[ts], actions[ts], rewards[ts], next_states[ts], dones[ts])
                    loss = agent.train()
                    if loss > 0: losses.append(loss)
                    
            # Logging
            for ts in env.ts_ids:
                logger.log_sar(ep, ts, step_idx, states[ts], actions[ts], rewards[ts], next_states[ts])
                ep_rewards[ts] += rewards[ts]
                ep_waits[ts].append(infos[ts]['wait'])
                ep_queues[ts].append(infos[ts]['queue'])
                ep_throughputs[ts] += infos[ts]['throughput']
                
            states = next_states
            done = dones[env.ts_ids[0]]
            step_idx += 1
            
        # End of episode tasks
        if args.algo == 'ppo':
            loss_info = agent.train()
            if loss_info and 'actor_loss' in loss_info:
                losses.append(loss_info['actor_loss'] + loss_info['critic_loss'])
        
        if hasattr(agent, 'update_target'):
            agent.update_target()
        if hasattr(agent, 'decay_epsilon'):
            agent.decay_epsilon()
            
        # Log episode metrics
        for ts in env.ts_ids:
            logger.log_intersection(
                episode=ep, ts_id=ts, reward=ep_rewards[ts],
                avg_wait=np.mean(ep_waits[ts]), avg_queue=np.mean(ep_queues[ts]),
                throughput=ep_throughputs[ts], current_phase=actions[ts], selected_action=actions[ts]
            )
            
        global_infos = infos['global']
        logger.log_episode({
            'episode': ep,
            'reward': sum(ep_rewards.values()),
            'avg_waiting_time': np.mean([np.mean(w) for w in ep_waits.values()]),
            'avg_queue_length': np.mean([np.mean(q) for q in ep_queues.values()]),
            'throughput': sum(ep_throughputs.values()),
            'avg_travel_time': global_infos['avg_travel_time'],
            'fuel_consumption': global_infos['fuel_consumption'],
            'avg_speed': np.mean([infos[ts]['speed'] for ts in env.ts_ids]),
            'loss': np.mean(losses) if losses else 0.0,
            'epsilon': agent.epsilon if hasattr(agent, 'epsilon') else 0.0,
            'learning_rate': agent.optimizer.param_groups[0]['lr'] if not args.algo == 'idqn' else agent.agents[env.ts_ids[0]].optimizer.param_groups[0]['lr'],
            'vehicles_completed': global_infos['completed_vehicles'],
            'vehicles_generated': global_infos['generated_vehicles']
        })
        
        print(f"Algorithm: {args.algo.upper()} | Episode {ep}/{args.episodes} | Reward: {sum(ep_rewards.values()):.2f} | Throughput: {sum(ep_throughputs.values())} | Queue: {np.mean([np.mean(q) for q in ep_queues.values()]):.2f}")
        
        if ep in checkpoints:
            checkpoint_path = os.path.join(results_dir, args.algo, f'checkpoint_ep{ep}.pth')
            agent.save(checkpoint_path)

if __name__ == '__main__':
    main()
