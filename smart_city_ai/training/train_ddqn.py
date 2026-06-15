import os
import sys
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path

# Adding parent dir to path to import environment and agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.sumo_environment import SumoEnvironment
from rl_agents.ddqn_agent import DDQNAgent
from scripts.generate_routes import build_routes
from utils.reliability import ensure_parent_dir, load_torch_checkpoint, save_json


logger = logging.getLogger(__name__)


def _get(cfg, key, default):
    # cfg may be argparse.Namespace or dict
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def train_ddqn(cfg=None):
    """Train DDQN with configuration provided as a dict or argparse.Namespace.

    Returns a dict with keys: best_reward, reward_history
    """
    config_file = _get(cfg, 'config_file', "sumo_network/config.sumocfg")
    if not os.path.exists(config_file):
        print("Config file not found. Ensure you are running from the top level dir.")
        return None

    import torch

    episodes = int(_get(cfg, 'episodes', 300))
    num_seconds = int(_get(cfg, 'num_seconds', 3600))
    seed = int(_get(cfg, 'seed', 42))
    checkpoint_every = int(_get(cfg, 'checkpoint_every', 1))
    checkpoint_dir = Path(_get(cfg, 'checkpoint_dir', os.path.join('results', 'checkpoints')))
    resume_from = _get(cfg, 'resume_from', None)

    lr = float(_get(cfg, 'lr', 1e-4))
    gamma = float(_get(cfg, 'gamma', 0.99))
    epsilon = float(_get(cfg, 'epsilon', 1.0))
    epsilon_decay = float(_get(cfg, 'epsilon_decay', 0.995))
    epsilon_min = float(_get(cfg, 'epsilon_min', 0.02))
    replay_size = int(_get(cfg, 'replay_size', 200000))
    batch_size = int(_get(cfg, 'batch_size', 256))
    target_update = int(_get(cfg, 'target_update', 1000))
    use_gui = bool(_get(cfg, 'use_gui', False))

    # device selection
    device_arg = _get(cfg, 'device', None)
    if device_arg is None or device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)

    env = SumoEnvironment(config_file, use_gui=use_gui, num_seconds=num_seconds)
    agent = DDQNAgent(state_dim=env.state_dim, action_dim=env.action_dim,
                      lr=lr, gamma=gamma, epsilon=epsilon,
                      epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                      replay_size=replay_size, batch_size=batch_size, target_update=target_update,
                      device=str(device))

    random.seed(seed)
    np.random.seed(seed)

    reward_history = []
    best_reward = float('-inf')
    start_episode = 0

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'ddqn_latest.pth'
    summary_path = checkpoint_dir / 'ddqn_training_state.json'

    if resume_from:
        try:
            checkpoint = load_torch_checkpoint(resume_from, map_location=device)
            agent.model.load_state_dict(checkpoint['model_state_dict'])
            if 'target_model_state_dict' in checkpoint:
                agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            else:
                agent.update_target_network()
            if 'optimizer_state_dict' in checkpoint:
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.epsilon = float(checkpoint.get('epsilon', agent.epsilon))
            best_reward = float(checkpoint.get('best_reward', best_reward))
            reward_history = list(checkpoint.get('reward_history', []))
            start_episode = int(checkpoint.get('episode', -1)) + 1
            if 'python_random_state' in checkpoint:
                random.setstate(checkpoint['python_random_state'])
            if 'numpy_random_state' in checkpoint:
                np.random.set_state(checkpoint['numpy_random_state'])
            logger.info('Resumed DDQN from %s at episode %s', resume_from, start_episode)
        except Exception:
            logger.exception('Failed to resume DDQN checkpoint from %s', resume_from)

    os.makedirs("models/ddqn", exist_ok=True)
    os.makedirs("results/reward_curves", exist_ok=True)

    def save_checkpoint(episode_index: int, final: bool = False):
        payload = {
            'episode': int(episode_index),
            'model_state_dict': agent.model.state_dict(),
            'target_model_state_dict': agent.target_model.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'state_dim': int(agent.state_dim) if hasattr(agent, 'state_dim') else None,
            'action_dim': int(agent.action_dim) if hasattr(agent, 'action_dim') else None,
            'agent': 'ddqn',
            'version': '1.0',
            'epsilon': float(agent.epsilon),
            'best_reward': float(best_reward),
            'reward_history': list(reward_history),
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'config': {
                'episodes': episodes,
                'num_seconds': num_seconds,
                'seed': seed,
                'lr': lr,
                'gamma': gamma,
                'epsilon_decay': epsilon_decay,
                'epsilon_min': epsilon_min,
                'replay_size': replay_size,
                'batch_size': batch_size,
                'target_update': target_update,
            },
            'final': bool(final),
        }
        torch_path = checkpoint_path if not final else (checkpoint_dir / 'ddqn_final.pth')
        import torch
        torch.save(payload, str(torch_path))
        save_json(summary_path, {
            'episode': int(episode_index),
            'best_reward': float(best_reward),
            'epsilon': float(agent.epsilon),
            'reward_history_len': len(reward_history),
            'checkpoint': str(torch_path),
            'final': bool(final),
        })
        return torch_path

    try:
        for e in range(start_episode, episodes):
            episode_seed = seed + 2000 + e
            build_routes(seed=episode_seed)

            state = env.reset(seed=episode_seed)
            total_reward = 0
            done = False
            action_counts = {i: 0 for i in range(env.action_dim)}

            while not done:
                action = agent.act(state)
                action_counts[action] += 1
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()

                state = next_state
                total_reward += reward

            # decay epsilon slowly after each episode
            try:
                agent.decay_epsilon()
            except Exception:
                pass

            action_dist = {k: round(v/max(1, sum(action_counts.values())), 2) for k, v in action_counts.items()}
            print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}, Actions: {action_dist}")
            reward_history.append(total_reward)

            try:
                metrics = env.get_episode_metrics()
                metrics_file = 'results/reward_curves/ddqn_metrics.csv'
                with open(metrics_file, 'a') as f:
                    if e == 0 and os.path.getsize(metrics_file) == 0:
                        f.write('episode,reward,wait,queue,avg_speed,switches\n')
                    f.write(f"{e+1},{total_reward},{metrics.get('wait','')},{metrics.get('queue','')},{metrics.get('avg_speed','')},{metrics.get('switches','')}\n")
            except Exception:
                pass

            if total_reward > best_reward:
                best_reward = total_reward
                agent.save("models/ddqn/ddqn_model_best.pth")
                print(f"  New best reward: {best_reward:.2f}! Saved best model.")

            if checkpoint_every > 0 and ((e + 1) % checkpoint_every == 0 or e == episodes - 1):
                save_checkpoint(e)
            
            if (e + 1) % 25 == 0:
                agent.save(f"models/ddqn/ddqn_model_ep_{e+1}.pth")
                print(f"  Saved periodic checkpoint at episode {e+1}")

        agent.save("models/ddqn/ddqn_model_final.pth")
        save_checkpoint(max(start_episode, episodes - 1), final=True)
    except KeyboardInterrupt:
        logger.warning('DDQN training interrupted at episode %s; saving checkpoint and shutting down.', e if 'e' in locals() else start_episode)
        save_checkpoint(int(e if 'e' in locals() else start_episode), final=False)
    finally:
        env.close()

    # optional plotting
    try:
        plt.plot(reward_history)
        plt.title('DDQN Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig('results/reward_curves/ddqn_reward.png')
        plt.close()
    except Exception:
        pass

    return {'best_reward': best_reward, 'reward_history': reward_history}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", dest='config_file', default="sumo_network/config.sumocfg")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--num-seconds", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--replay-size", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default=os.path.join('results', 'checkpoints'))
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--use-gui", action='store_true')
    parser.add_argument("--device", type=str, default='auto', help="Device to use: 'cpu', 'cuda' or 'auto'")

    args = parser.parse_args()

    # Normalize arg names to match our train function
    cfg = {
        'config_file': args.config_file,
        'episodes': args.episodes,
        'num_seconds': args.num_seconds,
        'seed': args.seed,
        'lr': args.lr,
        'gamma': args.gamma,
        'epsilon': args.epsilon,
        'epsilon_decay': args.epsilon_decay,
        'epsilon_min': args.epsilon_min,
        'replay_size': args.replay_size,
        'batch_size': args.batch_size,
        'target_update': args.target_update,
        'checkpoint_every': args.checkpoint_every,
        'checkpoint_dir': args.checkpoint_dir,
        'resume_from': args.resume_from,
        'use_gui': args.use_gui,
        'device': args.device,
    }

    train_ddqn(cfg)
