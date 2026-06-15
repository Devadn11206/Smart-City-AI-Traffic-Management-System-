#!/usr/bin/env python3
"""
Final evaluation harness: Compare all algorithms with long-duration eval.
Includes:
- Hybrid DQN
- Hybrid DDQN  
- Hybrid PPO
- Fixed-time baseline
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import logging
from statistics import mean, pstdev

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

from integration.hybrid_env import HybridSumoEnv
from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent
from utils.checkpoint_registry import resolve_active_checkpoint

EVAL_SEEDS = [101, 202, 303]

def run_fixed_baseline(env, steps=10000, seed=101, phase_durations=None):
    """Fixed-time signal: cycle through phases for fixed durations.

    phase_durations: iterable of durations in seconds for each phase index (0..n-1).
    If None, defaults to [30, 15, 30, 15]. Durations are clipped to respect env.min_green.
    """
    if phase_durations is None:
        phase_durations = [30, 15, 30, 15]

    env.reset(seed=seed)
    done = False
    phase_idx = 0
    # run until steps exhausted
    while not done and env.current_step < steps:
        duration = float(phase_durations[phase_idx % len(phase_durations)])
        # ensure phase respects environment's minimum green
        duration = max(duration, getattr(env, 'min_green', 1))
        # simulate for this phase duration (in seconds)
        remaining = min(duration, max(0, steps - env.current_step))
        # number of env.step() calls needed (env.step advances by env.action_repeat seconds)
        calls = int(np.ceil(remaining / max(1, env.action_repeat)))
        for _ in range(calls):
            _, _, done, _ = env.step(phase_idx)
            if done or env.current_step >= steps:
                break
        phase_idx += 1

    metrics = env.get_episode_metrics()
    return metrics

def run_agent(agent_type, env, steps=10000, seed=101):
    """Run trained agent for evaluation."""
    # Find best/final model
    candidates = []
    try:
        candidates.append(resolve_active_checkpoint(agent_type))
    except Exception:
        pass
    results_dir = repo_root / "results/hybrid_results"
    alt_results_dir = repo_root.parent / "results/hybrid_results"
    candidates.extend([
        results_dir / f"hybrid_{agent_type}_model_best.pth",
        results_dir / f"hybrid_{agent_type}_model_final.pth",
        alt_results_dir / f"hybrid_{agent_type}_model_best.pth",
        alt_results_dir / f"hybrid_{agent_type}_model_final.pth",
    ])
    model_path = next((path for path in candidates if path and Path(path).exists()), None)
    if model_path is None:
        print(f"[{agent_type.upper()}] No model found!")
        return None, None
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    if agent_type == 'dqn':
        agent = DQNAgent(state_dim, action_dim)
    elif agent_type == 'ddqn':
        agent = DDQNAgent(state_dim, action_dim)
    elif agent_type == 'ppo':
        agent = PPOAgent(state_dim, action_dim)
    
    agent.load(str(model_path))
    
    state = env.reset(seed=seed)
    done = False
    while not done and env.current_step < steps:
        if agent_type == 'ppo':
            action, _ = agent.act(state, eval_mode=True)
        else:
            action = agent.act(state, eval_mode=True)
        
        state, _, done, _ = env.step(action)
    
    metrics = env.get_episode_metrics()
    return metrics


def aggregate_metrics(metrics_list):
    return {
        'wait_mean': float(mean([m.get('wait', 0.0) for m in metrics_list])) if metrics_list else 0.0,
        'wait_std': float(pstdev([m.get('wait', 0.0) for m in metrics_list])) if len(metrics_list) > 1 else 0.0,
        'queue_mean': float(mean([m.get('queue', 0.0) for m in metrics_list])) if metrics_list else 0.0,
        'queue_std': float(pstdev([m.get('queue', 0.0) for m in metrics_list])) if len(metrics_list) > 1 else 0.0,
        'speed_mean': float(mean([m.get('avg_speed', 0.0) for m in metrics_list])) if metrics_list else 0.0,
        'throughput_mean': float(mean([m.get('throughput', 0.0) for m in metrics_list])) if metrics_list else 0.0,
        'switches_mean': float(mean([m.get('switches', 0.0) for m in metrics_list])) if metrics_list else 0.0,
        'reward_mean': float(mean([m.get('reward_ema', 0.0) for m in metrics_list])) if metrics_list else 0.0,
    }

def main(eval_steps=10000, seeds=None):
    config_file = repo_root / "sumo_network/config.sumocfg"
    pinn_model_path = repo_root / "pinn_models/trained_pinn.pth"

    if seeds is not None:
        EVAL_SEEDS[:] = [int(value) for value in seeds]

    if not pinn_model_path.exists():
        print(f"ERROR: PINN model not found at {pinn_model_path}")
        sys.exit(1)

    print(f"\nRunning Final Evaluation ({eval_steps} steps per agent)...")
    print("=" * 80)

    results = []

    print("\n[FIXED-TIME] Running baseline...")
    fixed_metrics = []
    env = HybridSumoEnv(str(config_file), str(pinn_model_path), num_seconds=eval_steps)
    for seed in EVAL_SEEDS:
        fixed_metrics.append(run_fixed_baseline(env, eval_steps, seed=seed))
    env.close()
    fixed_stats = aggregate_metrics(fixed_metrics)
    print(f"[FIXED-TIME] Wait: {fixed_stats['wait_mean']:.2f}s | Queue: {fixed_stats['queue_mean']:.2f}")
    results.append({'Agent': 'Fixed-Time', 'Waiting Time': fixed_stats['wait_mean'], 'Queue Length': fixed_stats['queue_mean'], 'Type': 'Baseline'})

    print("\n[DQN] Running evaluation...")
    env = HybridSumoEnv(str(config_file), str(pinn_model_path), num_seconds=eval_steps)
    dqn_metrics = [run_agent('dqn', env, eval_steps, seed=seed) for seed in EVAL_SEEDS]
    env.close()
    dqn_stats = aggregate_metrics(dqn_metrics)
    print(f"[DQN] Wait: {dqn_stats['wait_mean']:.2f}s | Queue: {dqn_stats['queue_mean']:.2f}")
    results.append({'Agent': 'Hybrid DQN', 'Waiting Time': dqn_stats['wait_mean'], 'Queue Length': dqn_stats['queue_mean'], 'Type': 'Hybrid'})

    print("\n[DDQN] Running evaluation...")
    env = HybridSumoEnv(str(config_file), str(pinn_model_path), num_seconds=eval_steps)
    ddqn_metrics = [run_agent('ddqn', env, eval_steps, seed=seed) for seed in EVAL_SEEDS]
    env.close()
    ddqn_stats = aggregate_metrics(ddqn_metrics)
    print(f"[DDQN] Wait: {ddqn_stats['wait_mean']:.2f}s | Queue: {ddqn_stats['queue_mean']:.2f}")
    results.append({'Agent': 'Hybrid DDQN', 'Waiting Time': ddqn_stats['wait_mean'], 'Queue Length': ddqn_stats['queue_mean'], 'Type': 'Hybrid'})

    print("\n[PPO] Running evaluation...")
    env = HybridSumoEnv(str(config_file), str(pinn_model_path), num_seconds=eval_steps)
    ppo_metrics = [run_agent('ppo', env, eval_steps, seed=seed) for seed in EVAL_SEEDS]
    env.close()
    ppo_stats = aggregate_metrics(ppo_metrics)
    print(f"[PPO] Wait: {ppo_stats['wait_mean']:.2f}s | Queue: {ppo_stats['queue_mean']:.2f}")
    results.append({'Agent': 'Hybrid PPO', 'Waiting Time': ppo_stats['wait_mean'], 'Queue Length': ppo_stats['queue_mean'], 'Type': 'Hybrid'})

    if len(results) > 0:
        df = pd.DataFrame(results)
        os.makedirs(repo_root / "visualizations/comparison_charts", exist_ok=True)
        output_csv = repo_root / "visualizations/comparison_charts/final_evaluation_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Saved evaluation results to: {output_csv}")
        print("\n" + "=" * 80)
        print("FINAL EVALUATION RESULTS")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        return df

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seeds", type=str, default="101,202,303")
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(',') if value.strip()]
    main(eval_steps=args.steps, seeds=seeds)
