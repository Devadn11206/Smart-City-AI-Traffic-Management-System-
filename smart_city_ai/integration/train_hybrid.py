import os
import sys
import argparse
import random
import logging
import time

import numpy as np
import torch

# Add parent path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.hybrid_env import HybridSumoEnv
from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent
from utils.reliability import load_torch_checkpoint, save_json
from utils.checkpoint_registry import build_checkpoint_metadata


logger = logging.getLogger(__name__)

def train_hybrid_agent(agent_type, episodes=300, num_seconds=10000, config_file=None, pinn_model_path=None):
    # Resolve defaults relative to the repository smart_city_ai root (this file lives in integration/)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if config_file is None:
        config_file = os.path.join(repo_root, "sumo_network", "config.sumocfg")
    if pinn_model_path is None:
        pinn_model_path = os.path.join(repo_root, "pinn_models", "trained_pinn.pth")
    
    if not os.path.exists(pinn_model_path):
        print(f"Error: PINN model not found at {pinn_model_path}. Please train it first.")
        sys.exit(1)
        
    env = HybridSumoEnv(config_file, pinn_model_path, use_gui=False, num_seconds=num_seconds)

    random.seed(42)
    np.random.seed(42)
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    # device selection: allow environment or caller to set CUDA/CPU preference via env var or later arg
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if agent_type == 'dqn':
        agent = DQNAgent(state_dim, action_dim, device=str(device))
    elif agent_type == 'ddqn':
        agent = DDQNAgent(state_dim, action_dim, device=str(device))
    elif agent_type == 'ppo':
        agent = PPOAgent(state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-4, gae_lambda=0.95, entropy_coef=0.01, device=str(device))
    else:
        print(f"Unknown agent type: {agent_type}")
        return

    print(f"Training Hybrid {agent_type.upper()} agent...")
    
    best_reward = float('-inf')
    results_dir = os.path.join(repo_root, "results", "hybrid_results")
    log_dir = os.path.join(repo_root, "results", "hybrid_logs")
    checkpoint_dir = os.path.join(repo_root, "results", "checkpoints")
    canonical_checkpoint_dir = os.path.join(os.path.dirname(repo_root), "results", "checkpoints", "active")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(canonical_checkpoint_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "hybrid_training_log.csv")
    checkpoint_file = os.path.join(canonical_checkpoint_dir, f"{agent_type}.pth")

    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('episode,agent,reward,wait,queue,avg_speed,switches,pinn_rho,pinn_velocity,pinn_confidence,pinn_weight\n')

    start_episode = 0
    if os.path.exists(checkpoint_file):
        try:
            checkpoint = load_torch_checkpoint(checkpoint_file, map_location=device)
            agent_state = checkpoint.get('agent_state_dict') or checkpoint.get('model_state_dict')
            if agent_state:
                if hasattr(agent, 'policy'):
                    agent.policy.load_state_dict(agent_state)
                    agent.policy_old.load_state_dict(agent.policy.state_dict())
                elif hasattr(agent, 'model'):
                    agent.model.load_state_dict(agent_state)
                    agent.target_model.load_state_dict(checkpoint.get('target_state_dict', agent.model.state_dict()))
            if hasattr(agent, 'optimizer') and 'optimizer_state_dict' in checkpoint:
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epsilon' in checkpoint and hasattr(agent, 'epsilon'):
                agent.epsilon = float(checkpoint['epsilon'])
            start_episode = int(checkpoint.get('episode', -1)) + 1
            logger.info('Resumed hybrid %s from episode %s', agent_type, start_episode)
        except Exception:
            logger.exception('Could not resume hybrid checkpoint from %s', checkpoint_file)

    def save_checkpoint(episode_index: int, final: bool = False):
        metadata = build_checkpoint_metadata(
            algorithm=agent_type,
            state_dim=int(getattr(agent, 'state_dim', 0)),
            action_dim=int(getattr(agent, 'action_dim', 0)),
            extra={
                'training_date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'git_hash': 'unknown',
                'benchmark_score': float(best_reward),
            },
        )
        payload = {
            'episode': int(episode_index),
            'agent_type': agent_type,
            'epsilon': float(getattr(agent, 'epsilon', 0.0)),
            'agent_state_dict': agent.policy.state_dict() if hasattr(agent, 'policy') else agent.model.state_dict(),
            'target_state_dict': agent.policy_old.state_dict() if hasattr(agent, 'policy_old') else agent.target_model.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict() if hasattr(agent, 'optimizer') else {},
            'final': bool(final),
            'metadata': metadata,
        }
        torch.save(payload, checkpoint_file if not final else os.path.join(checkpoint_dir, f'hybrid_{agent_type}_final.pth'))
        save_json(os.path.join(checkpoint_dir, f'hybrid_{agent_type}_state.json'), {
            'episode': int(episode_index),
            'final': bool(final),
            'checkpoint': checkpoint_file if not final else os.path.join(checkpoint_dir, f'hybrid_{agent_type}_final.pth'),
        })

    try:
        for e in range(start_episode, episodes):
            episode_seed = 4000 + e
            state = env.reset(seed=episode_seed)
            total_reward = 0
            done = False

            while not done:
                if agent_type == 'ppo':
                    action, logprob = agent.act(state)
                else:
                    action = agent.act(state)

                next_state, reward, done, _ = env.step(action)

                if agent_type in ['dqn', 'ddqn']:
                    agent.remember(state, action, reward, next_state, done)
                    agent.replay()
                else:  # PPO
                    agent.remember(state, action, logprob, reward, done)

                state = next_state
                total_reward += reward

            # PPO target update or agent specific End-of-Episode ops
            if agent_type in ['dqn', 'ddqn']:
                try:
                    agent.update_target_network()
                except Exception:
                    logger.exception('Failed to update target network for %s agent', agent_type)
                if hasattr(agent, 'decay_epsilon'):
                    agent.decay_epsilon()
            elif agent_type == 'ppo':
                agent.replay()

            # log episode metrics averaged over the full run
            metrics = env.get_episode_metrics()
            pinn_info = getattr(env, 'last_pinn_prediction', {}) or {}
            # compute average per-action reward to avoid large cumulative magnitudes
            avg_reward = float(total_reward) / max(1, float(env.metric_samples))
            with open(log_file, 'a') as f:
                f.write(f"{e+1},{agent_type},{avg_reward},{metrics['wait']},{metrics['queue']},{metrics['avg_speed']},{metrics['switches']},{pinn_info.get('rho', 0.0)},{pinn_info.get('velocity', 0.0)},{pinn_info.get('confidence', 0.0)},{pinn_info.get('effective_weight', 0.0)}\n")

            print(f"Episode: {e+1}/{episodes}, Avg Reward: {avg_reward:.4f}, Total Reward: {total_reward:.2f}")

            # Save best model by average per-action reward
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_path = os.path.join(results_dir, f"hybrid_{agent_type}_model_best.pth")
                try:
                    agent.save(best_path)
                    print(f"  New best avg reward: {best_reward:.4f}! Saved best model to {best_path}.")
                except Exception:
                    logger.exception('Failed to save best hybrid model to %s', best_path)
            save_checkpoint(e)

    except KeyboardInterrupt:
        logger.warning('Hybrid %s training interrupted at episode %s; saving checkpoint.', agent_type, e if 'e' in locals() else start_episode)
        save_checkpoint(int(e if 'e' in locals() else start_episode), final=False)
    finally:
        try:
            env.close()
        except Exception:
            logger.exception('Failed to close hybrid environment cleanly')

    # Save the final model
    save_path = os.path.join(results_dir, f"hybrid_{agent_type}_model_final.pth")
    try:
        agent.save(save_path)
    except Exception:
        logger.exception('Failed to save final hybrid model to %s', save_path)
    else:
        save_checkpoint(max(start_episode, episodes - 1), final=True)
    print(f"Saved Hybrid {agent_type.upper()} final model to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=['dqn', 'ddqn', 'ppo'], default='dqn')
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--num-seconds", type=int, default=10000, help="Simulation horizon per episode")
    parser.add_argument("--config", type=str, default=None, help="Path to sumo config file")
    parser.add_argument("--pinn", type=str, default=None, help="Path to trained PINN model .pth")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to hybrid training checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory for hybrid checkpoints")
    args = parser.parse_args()

    train_hybrid_agent(args.agent, args.episodes, num_seconds=args.num_seconds, config_file=args.config, pinn_model_path=args.pinn)
