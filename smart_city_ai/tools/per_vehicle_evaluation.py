#!/usr/bin/env python3
"""
Per-vehicle aggregation evaluation using cumulative wait / arrived vehicles

Computes average waiting time per vehicle as: total_incremental_wait / total_arrived
Also reports throughput and averaged queue.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from integration.hybrid_env import HybridSumoEnv
from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent


def run_agent_per_vehicle(agent_type, env, steps=10000):
    # load model
    results_dir = repo_root / "results/hybrid_results"
    model_path_best = results_dir / f"hybrid_{agent_type}_model_best.pth"
    model_path_final = results_dir / f"hybrid_{agent_type}_model_final.pth"
    model_path = None
    if model_path_best.exists():
        model_path = model_path_best
    elif model_path_final.exists():
        model_path = model_path_final

    if model_path is None:
        print(f"[{agent_type.upper()}] No model found!")
        return None

    state_dim = env.state_dim
    action_dim = env.action_dim
    if agent_type == 'dqn':
        agent = DQNAgent(state_dim, action_dim)
    elif agent_type == 'ddqn':
        agent = DDQNAgent(state_dim, action_dim)
    elif agent_type == 'ppo':
        agent = PPOAgent(state_dim, action_dim)

    agent.load(str(model_path))

    # run
    state = env.reset()
    done = False
    while not done and env.current_step < steps:
        if agent_type == 'ppo':
            action, _ = agent.act(state, eval_mode=True)
        else:
            action = agent.act(state, eval_mode=True)
        state, _, done, _ = env.step(action)

    # compute per-vehicle averages using cumulative stats tracked in env
    total_wait = getattr(env, 'cumulative_wait', None)
    total_arrived = getattr(env, 'cumulative_throughput', None)
    avg_wait_per_vehicle = None
    if total_wait is not None and total_arrived and total_arrived > 0:
        avg_wait_per_vehicle = float(total_wait) / float(total_arrived)

    avg_queue = (env.cumulative_queue / env.metric_samples) if env.metric_samples > 0 else 0.0
    throughput = total_arrived if total_arrived is not None else 0

    env.close()
    return {'avg_wait_per_vehicle': avg_wait_per_vehicle, 'avg_queue_per_step': avg_queue, 'throughput': throughput}


def run_fixed_baseline(env, steps=10000, phase_durations=None, seed=None):
    if phase_durations is None:
        phase_durations = [30, 15, 30, 15]

    env.reset(seed=seed)
    done = False
    phase_idx = 0
    while not done and env.current_step < steps:
        duration = float(phase_durations[phase_idx % len(phase_durations)])
        duration = max(duration, getattr(env, 'min_green', 1))
        remaining = min(duration, max(0, steps - env.current_step))
        calls = int(np.ceil(remaining / max(1, env.action_repeat)))
        for _ in range(calls):
            _, _, done, _ = env.step(phase_idx)
            if done or env.current_step >= steps:
                break
        phase_idx += 1

    total_wait = getattr(env, 'cumulative_wait', 0.0)
    total_arrived = getattr(env, 'cumulative_throughput', 0)
    avg_wait_per_vehicle = float(total_wait) / float(total_arrived) if total_arrived > 0 else None
    avg_queue = (env.cumulative_queue / env.metric_samples) if env.metric_samples > 0 else 0.0
    env.close()
    return {'avg_wait_per_vehicle': avg_wait_per_vehicle, 'avg_queue_per_step': avg_queue, 'throughput': total_arrived}


def main():
    config_file = repo_root / "sumo_network/config.sumocfg"
    pinn_model_path = repo_root / "pinn_models/trained_pinn.pth"
    if not pinn_model_path.exists():
        print(f"ERROR: PINN model not found at {pinn_model_path}")
        sys.exit(1)

    steps = 10000
    print(f"Running per-vehicle aggregation evaluation ({steps} steps)")

    results = []

    # Fixed
    env = HybridSumoEnv(str(config_file), str(pinn_model_path), num_seconds=steps)
    print("[FIXED-TIME] Running baseline...")
    res = run_fixed_baseline(env, steps)
    results.append({'Agent': 'Fixed-Time', 'AvgWaitPerVehicle': res['avg_wait_per_vehicle'], 'AvgQueuePerStep': res['avg_queue_per_step'], 'Throughput': res['throughput'], 'Type': 'Baseline'})
    print(f"[FIXED-TIME] Avg wait/vehicle: {res['avg_wait_per_vehicle']}, throughput: {res['throughput']}")

    # DQN
    print("[DQN] Running per-vehicle evaluation...")
    env = HybridSumoEnv(str(config_file), str(pinn_model_path), num_seconds=steps)
    try:
        res = run_agent_per_vehicle('dqn', env, steps)
        if res:
            results.append({'Agent': 'Hybrid DQN', 'AvgWaitPerVehicle': res['avg_wait_per_vehicle'], 'AvgQueuePerStep': res['avg_queue_per_step'], 'Throughput': res['throughput'], 'Type': 'Hybrid'})
            print(f"[DQN] Avg wait/vehicle: {res['avg_wait_per_vehicle']}, throughput: {res['throughput']}")
    except Exception as e:
        print(f"DQN evaluation error: {e}")

    # DDQN
    print("[DDQN] Running per-vehicle evaluation...")
    env = HybridSumoEnv(str(config_file), str(pinn_model_path), num_seconds=steps)
    res = run_agent_per_vehicle('ddqn', env, steps)
    if res:
        results.append({'Agent': 'Hybrid DDQN', 'AvgWaitPerVehicle': res['avg_wait_per_vehicle'], 'AvgQueuePerStep': res['avg_queue_per_step'], 'Throughput': res['throughput'], 'Type': 'Hybrid'})
        print(f"[DDQN] Avg wait/vehicle: {res['avg_wait_per_vehicle']}, throughput: {res['throughput']}")

    # PPO
    print("[PPO] Running per-vehicle evaluation...")
    env = HybridSumoEnv(str(config_file), str(pinn_model_path), num_seconds=steps)
    res = run_agent_per_vehicle('ppo', env, steps)
    if res:
        results.append({'Agent': 'Hybrid PPO', 'AvgWaitPerVehicle': res['avg_wait_per_vehicle'], 'AvgQueuePerStep': res['avg_queue_per_step'], 'Throughput': res['throughput'], 'Type': 'Hybrid'})
        print(f"[PPO] Avg wait/vehicle: {res['avg_wait_per_vehicle']}, throughput: {res['throughput']}")

    # Save
    df = pd.DataFrame(results)
    out_dir = repo_root / 'visualizations' / 'comparison_charts'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'per_vehicle_evaluation_results.csv'
    df.to_csv(out_csv, index=False)
    print(f"Saved per-vehicle evaluation to: {out_csv}")

    print('\nResults:\n')
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
