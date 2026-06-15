import os
import sys
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from environment.sumo_environment import SumoEnvironment
from integration.hybrid_env import HybridSumoEnv
from utils.checkpoint_registry import resolve_active_checkpoint
from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent

DEFAULT_EVAL_SEEDS = [11, 23, 37]


def _resolve_model_path(agent_name: str) -> str | None:
    try:
        return resolve_active_checkpoint(agent_name)
    except Exception:
        pass
    candidates = [
        repo_root / 'models' / agent_name / f'{agent_name}_model_best.pth',
        repo_root / 'models' / agent_name / f'{agent_name}_model_final.pth',
        repo_root / 'results' / 'hybrid_results' / f'hybrid_{agent_name}_model_best.pth',
        repo_root / 'results' / 'hybrid_results' / f'hybrid_{agent_name}_model_final.pth',
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _checkpoint_state_dim(model_path: str) -> int:
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    for tensor in state_dict.values():
        if hasattr(tensor, 'shape') and len(tensor.shape) == 2:
            in_dim = int(tensor.shape[1])
            if in_dim in (7, 9):
                return in_dim
    return 7


def _build_env(config_file: str, state_dim: int):
    pinn_path = repo_root / 'pinn_models' / 'trained_pinn.pth'
    if state_dim >= 9 and pinn_path.exists():
        return HybridSumoEnv(config_file, str(pinn_path), use_gui=False, num_seconds=2000)
    return SumoEnvironment(config_file, use_gui=False, num_seconds=2000)


def evaluate_agent(env, agent, policy_type="RL", seeds=None):
    seeds = list(seeds or DEFAULT_EVAL_SEEDS)
    total_rewards = []
    total_wait_times = []
    total_queue_lengths = []
    total_avg_speeds = []
    total_throughputs = []
    total_switches = []

    for seed in seeds:
        state = env.reset(seed=seed)
        done = False
        reward_sum = 0.0

        while not done:
            if policy_type == "RL":
                action = agent.act(state, eval_mode=True) if hasattr(agent, "act") else agent.act(state)
                if isinstance(action, tuple):
                    action = action[0]
            else:
                action = 0

            next_state, reward, done, _ = env.step(action)
            reward_sum += float(reward)
            state = next_state

        metrics = env.get_episode_metrics()
        total_rewards.append(reward_sum)
        total_wait_times.append(float(metrics.get('wait', 0.0)))
        total_queue_lengths.append(float(metrics.get('queue', 0.0)))
        total_avg_speeds.append(float(metrics.get('avg_speed', 0.0)))
        total_throughputs.append(float(metrics.get('throughput', 0.0)))
        total_switches.append(float(metrics.get('switches', 0.0)))

    return {
        'reward_mean': float(mean(total_rewards)) if total_rewards else 0.0,
        'reward_std': float(pstdev(total_rewards)) if len(total_rewards) > 1 else 0.0,
        'wait_mean': float(mean(total_wait_times)) if total_wait_times else 0.0,
        'wait_std': float(pstdev(total_wait_times)) if len(total_wait_times) > 1 else 0.0,
        'queue_mean': float(mean(total_queue_lengths)) if total_queue_lengths else 0.0,
        'queue_std': float(pstdev(total_queue_lengths)) if len(total_queue_lengths) > 1 else 0.0,
        'speed_mean': float(mean(total_avg_speeds)) if total_avg_speeds else 0.0,
        'throughput_mean': float(mean(total_throughputs)) if total_throughputs else 0.0,
        'switches_mean': float(mean(total_switches)) if total_switches else 0.0,
    }


def _load_agent(agent_cls, model_path: str, state_dim: int, action_dim: int):
    agent = agent_cls(state_dim, action_dim)
    agent.load(model_path)
    return agent


def main():
    config_file = str((repo_root / 'sumo_network' / 'config.sumocfg').resolve())
    eval_seeds = DEFAULT_EVAL_SEEDS

    if not os.path.exists(config_file):
        print('Config file not found. Ensure you are running from the top level dir.')
        return

    print('Evaluating Fixed Time...')
    fixed_env = SumoEnvironment(config_file, use_gui=False, num_seconds=2000)
    fixed_stats = evaluate_agent(fixed_env, None, policy_type='Fixed', seeds=eval_seeds)
    fixed_env.close()

    results = [(
        'Fixed Time',
        fixed_stats['reward_mean'],
        fixed_stats['wait_mean'],
        fixed_stats['queue_mean'],
        fixed_stats['wait_std'],
        fixed_stats['queue_std'],
        fixed_stats['throughput_mean'],
    )]

    agent_specs = [
        ('DQN', 'dqn', DQNAgent),
        ('DDQN', 'ddqn', DDQNAgent),
        ('PPO', 'ppo', PPOAgent),
    ]

    for label, agent_name, agent_cls in agent_specs:
        model_path = _resolve_model_path(agent_name)
        if model_path is None:
            print(f'{label} model not found')
            continue

        state_dim = _checkpoint_state_dim(model_path)
        env = _build_env(config_file, state_dim)
        agent = _load_agent(agent_cls, model_path, env.state_dim, env.action_dim)
        print(f'Evaluating {label}...')
        stats = evaluate_agent(env, agent, seeds=eval_seeds)
        env.close()
        results.append((label, stats['reward_mean'], stats['wait_mean'], stats['queue_mean'], stats['wait_std'], stats['queue_std'], stats['throughput_mean']))

    labels = [row[0] for row in results]
    rewards = [row[1] for row in results]
    waits = [row[2] for row in results]
    queues = [row[3] for row in results]
    wait_stds = [row[4] for row in results]
    queue_stds = [row[5] for row in results]
    throughputs = [row[6] for row in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x - width / 2, waits, width, label='Avg Wait Time', color='b')
    ax1.bar(x + width / 2, queues, width, label='Avg Queue Length', color='r')
    ax1.set_ylabel('Scores')
    ax1.set_title('Performance Comparison of Traffic Signal Optimization Algorithms')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()

    output_dir = repo_root / 'results' / 'comparison_charts'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_dir / 'comparison.png'))
    plt.close()

    metrics_dir = repo_root / 'results' / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / 'evaluation_results.txt', 'w', encoding='utf-8') as f:
        for i, label in enumerate(labels):
            f.write(f'{label} -> Reward: {rewards[i]:.2f}, Wait: {waits[i]:.2f} ± {wait_stds[i]:.2f}, Queue: {queues[i]:.2f} ± {queue_stds[i]:.2f}, Throughput: {throughputs[i]:.2f}\n')

    print('Evaluation Complete. Check results/comparison_charts/ and results/metrics/')


if __name__ == '__main__':
    main()
