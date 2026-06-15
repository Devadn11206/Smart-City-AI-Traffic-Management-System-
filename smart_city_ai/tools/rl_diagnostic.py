#!/usr/bin/env python3
"""
RL Agent Diagnostic Tool

Analyzes:
1. Model decision-making patterns
2. Training convergence
3. Checkpoint integrity
4. Inference quality
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent
from integration.hybrid_env import HybridSumoEnv
from utils.checkpoint_registry import resolve_active_checkpoint


def diagnose_model_decisions(agent_type='ddqn', num_steps=500):
    """Analyze if model is making diverse decisions or defaulting to random."""
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC: {agent_type.upper()} Decision-Making Analysis")
    print(f"{'='*80}")
    
    # Load model
    config_file = repo_root / "sumo_network/config.sumocfg"
    pinn_model_path = repo_root / "pinn_models/trained_pinn.pth"
    
    if not pinn_model_path.exists():
        print(f"ERROR: PINN model not found at {pinn_model_path}")
        return None
    
    env = HybridSumoEnv(str(config_file), str(pinn_model_path), num_seconds=2000)
    
    try:
        model_path = resolve_active_checkpoint(agent_type)
    except Exception:
        results_dir = repo_root / "results/hybrid_results"
        candidates = [
            results_dir / f"hybrid_{agent_type}_model_best.pth",
            results_dir / f"hybrid_{agent_type}_model_final.pth",
        ]
        model_path = next((p for p in candidates if p.exists()), None)
    
    if model_path is None:
        print(f"ERROR: No checkpoint found for {agent_type}")
        return None
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    # Load agent
    if agent_type == 'dqn':
        agent = DQNAgent(state_dim, action_dim)
    elif agent_type == 'ddqn':
        agent = DDQNAgent(state_dim, action_dim)
    elif agent_type == 'ppo':
        agent = PPOAgent(state_dim, action_dim)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent.load(str(model_path))
    print(f"✓ Loaded {agent_type.upper()} from {model_path}")
    
    # Run inference and collect decision statistics
    state = env.reset(seed=101)
    actions_taken = []
    q_values_list = []
    rewards = []
    done = False
    step = 0
    
    while not done and step < num_steps:
        # Get raw Q-values or policy output
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device if hasattr(agent, 'device') else 'cpu')
        
        if agent_type in ['dqn', 'ddqn']:
            with torch.no_grad():
                q_vals = agent.model(state_tensor).cpu().numpy()[0]
            q_values_list.append(q_vals.tolist())
            action = agent.act(state, eval_mode=True)
        else:  # PPO
            with torch.no_grad():
                policy_dist, value = agent.policy(state_tensor)
            policy = policy_dist.cpu().numpy()[0]
            q_values_list.append(policy.tolist())
            action, _ = agent.act(state, eval_mode=True)
        
        actions_taken.append(int(action))
        next_state, reward, done, _ = env.step(action)
        rewards.append(float(reward))
        state = next_state
        step += 1
    
    env.close()
    
    # Analyze results
    action_counts = Counter(actions_taken)
    q_vals_array = np.array(q_values_list)
    
    report = {
        'agent': agent_type,
        'steps_analyzed': step,
        'checkpoint': str(model_path),
        'decision_diversity': {
            'action_distribution': dict(action_counts),
            'unique_actions': len(action_counts),
            'max_action_frequency': max(action_counts.values()) if action_counts else 0,
            'min_action_frequency': min(action_counts.values()) if action_counts else 0,
        },
        'q_value_statistics': {
            'mean_q_value': float(np.mean(q_vals_array)),
            'std_q_value': float(np.std(q_vals_array)),
            'max_q_value': float(np.max(q_vals_array)),
            'min_q_value': float(np.min(q_vals_array)),
            'q_value_spread': float(np.max(q_vals_array) - np.min(q_vals_array)),
        },
        'reward_statistics': {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'total_reward': float(sum(rewards)),
        },
        'diagnostics': {
            'is_exploring': len(action_counts) > 1,
            'has_diverse_actions': len(action_counts) >= action_dim * 0.5,
            'q_values_are_reasonable': abs(float(np.mean(q_vals_array))) < 100,
            'reward_is_non_zero': float(np.std(rewards)) > 0.01,
        }
    }
    
    print(f"\nAction Distribution: {dict(action_counts)}")
    print(f"Unique Actions: {len(action_counts)}/{action_dim}")
    print(f"Q-Value Mean: {report['q_value_statistics']['mean_q_value']:.4f}")
    print(f"Q-Value Std: {report['q_value_statistics']['std_q_value']:.4f}")
    print(f"Total Reward: {report['reward_statistics']['total_reward']:.2f}")
    print(f"Mean Reward: {report['reward_statistics']['mean_reward']:.4f}")
    
    if report['diagnostics']['is_exploring']:
        print(f"✓ Model IS making diverse decisions")
    else:
        print(f"✗ Model is NOT exploring - might be stuck in local mode")
    
    if not report['diagnostics']['has_diverse_actions']:
        print(f"⚠ WARNING: Model uses < 50% of available actions")
    
    if not report['diagnostics']['q_values_are_reasonable']:
        print(f"⚠ WARNING: Q-values appear unreasonable (too large)")
    
    if not report['diagnostics']['reward_is_non_zero']:
        print(f"⚠ WARNING: Reward signal is nearly zero - might indicate reward function issue")
    
    return report


def compare_model_behaviors(seeds=[101, 202, 303]):
    """Compare decision patterns across multiple seeds."""
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC: Cross-Seed Behavior Analysis")
    print(f"{'='*80}\n")
    
    config_file = repo_root / "sumo_network/config.sumocfg"
    pinn_model_path = repo_root / "pinn_models/trained_pinn.pth"
    
    results = {}
    for agent_type in ['dqn', 'ddqn', 'ppo']:
        results[agent_type] = {}
        
        try:
            model_path = resolve_active_checkpoint(agent_type)
        except Exception:
            results_dir = repo_root / "results/hybrid_results"
            model_path = results_dir / f"hybrid_{agent_type}_model_best.pth"
        
        if not Path(model_path).exists():
            print(f"[{agent_type.upper()}] Model not found")
            continue
        
        env = HybridSumoEnv(str(config_file), str(pinn_model_path), num_seconds=2000)
        state_dim = env.state_dim
        action_dim = env.action_dim
        
        if agent_type == 'dqn':
            agent = DQNAgent(state_dim, action_dim)
        elif agent_type == 'ddqn':
            agent = DDQNAgent(state_dim, action_dim)
        else:
            agent = PPOAgent(state_dim, action_dim)
        
        agent.load(str(model_path))
        
        seed_actions = []
        seed_rewards = []
        
        for seed in seeds:
            state = env.reset(seed=seed)
            actions = []
            rewards = []
            done = False
            
            for _ in range(200):
                if agent_type == 'ppo':
                    action, _ = agent.act(state, eval_mode=True)
                else:
                    action = agent.act(state, eval_mode=True)
                
                next_state, reward, done, _ = env.step(action)
                actions.append(int(action))
                rewards.append(float(reward))
                state = next_state
                
                if done:
                    break
            
            seed_actions.append(actions)
            seed_rewards.append(rewards)
        
        env.close()
        
        # Analyze consistency
        all_actions = [a for seed_list in seed_actions for a in seed_list]
        all_rewards = [r for seed_list in seed_rewards for r in seed_list]
        
        results[agent_type] = {
            'action_uniqueness': len(set(all_actions)) / action_dim,
            'mean_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'reward_per_seed': [float(np.mean(sr)) for sr in seed_rewards],
        }
        
        print(f"[{agent_type.upper()}]")
        print(f"  Action Uniqueness: {results[agent_type]['action_uniqueness']:.2%}")
        print(f"  Mean Reward: {results[agent_type]['mean_reward']:.4f}")
        print(f"  Reward Std Dev: {results[agent_type]['std_reward']:.4f}")
        print()
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', default='ddqn', choices=['dqn', 'ddqn', 'ppo'])
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--compare', action='store_true', help='Compare across seeds')
    args = parser.parse_args()
    
    if args.compare:
        compare_model_behaviors()
    else:
        diagnose_model_decisions(args.agent, args.steps)
