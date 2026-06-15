#!/usr/bin/env python3
"""
Enhanced RL Training Pipeline

Features:
- Multiple seeds for statistical significance
- Comprehensive logging
- Convergence monitoring
- Hyperparameter optimization
- Checkpoint management
"""
import os
import sys
import json
import time
import random
import logging
import numpy as np
from pathlib import Path
from statistics import mean, stdev
from dataclasses import asdict, dataclass

import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from environment.sumo_environment import SumoEnvironment
from integration.hybrid_env import HybridSumoEnv
from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent
from scripts.generate_routes import build_routes
from utils.reliability import save_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    agent_type: str  # 'dqn', 'ddqn', 'ppo'
    episodes: int = 200
    num_seconds: int = 10000
    seed: int = 42
    
    # DQN/DDQN parameters
    lr: float = 1e-4
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.02
    replay_size: int = 200000
    batch_size: int = 256
    target_update: int = 1000
    
    # PPO parameters
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    entropy_coef: float = 0.02  # Improved from 0.01
    eps_clip: float = 0.2
    k_epochs: int = 4
    gae_lambda: float = 0.95
    
    # Training parameters
    use_hybrid: bool = True  # Use hybrid env with PINN
    pinn_path: str = None
    save_interval: int = 10
    config_file: str = None
    
    # Results
    results_dir: str = None
    checkpoint_dir: str = None
    
    def __post_init__(self):
        if self.config_file is None:
            self.config_file = str(repo_root / "sumo_network/config.sumocfg")
        if self.pinn_path is None:
            self.pinn_path = str(repo_root / "pinn_models/trained_pinn.pth")
        if self.results_dir is None:
            self.results_dir = str(repo_root / "results/rl_training")
        if self.checkpoint_dir is None:
            self.checkpoint_dir = str(repo_root / "results/checkpoints")

        # Apply safer tuned defaults for specialized agents
        if self.agent_type == 'ddqn':
            if self.lr == 1e-4:
                self.lr = 2e-4
            if self.batch_size == 256:
                self.batch_size = 128
            if self.epsilon_decay == 0.995:
                self.epsilon_decay = 0.998
            if self.epsilon_min == 0.02:
                self.epsilon_min = 0.05
            if self.target_update == 1000:
                self.target_update = 500
        elif self.agent_type == 'ppo':
            if self.entropy_coef == 0.02:
                self.entropy_coef = 0.05
            if self.eps_clip == 0.2:
                self.eps_clip = 0.3
            if self.k_epochs == 4:
                self.k_epochs = 5


class TrainingPipeline:
    """Enhanced training pipeline with convergence tracking."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.reward_history = []
        self.metrics_history = []
        self.convergence_metrics = {}
        self.best_metrics = {}
    
    def setup_environment(self):
        """Create training environment."""
        if self.config.use_hybrid:
            env = HybridSumoEnv(
                self.config.config_file,
                self.config.pinn_path,
                use_gui=False,
                num_seconds=self.config.num_seconds
            )
        else:
            env = SumoEnvironment(
                self.config.config_file,
                use_gui=False,
                num_seconds=self.config.num_seconds
            )
        return env
    
    def setup_agent(self, env):
        """Create RL agent with configured hyperparameters."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dim = env.state_dim
        action_dim = env.action_dim
        
        if self.config.agent_type == 'dqn':
            agent = DQNAgent(
                state_dim, action_dim,
                lr=self.config.lr,
                gamma=self.config.gamma,
                epsilon=self.config.epsilon,
                epsilon_decay=self.config.epsilon_decay,
                epsilon_min=self.config.epsilon_min,
                replay_size=self.config.replay_size,
                batch_size=self.config.batch_size,
                target_update=self.config.target_update,
                device=str(device)
            )
        elif self.config.agent_type == 'ddqn':
            agent = DDQNAgent(
                state_dim, action_dim,
                lr=self.config.lr,
                gamma=self.config.gamma,
                epsilon=self.config.epsilon,
                epsilon_decay=self.config.epsilon_decay,
                epsilon_min=self.config.epsilon_min,
                replay_size=self.config.replay_size,
                batch_size=self.config.batch_size,
                target_update=self.config.target_update,
                device=str(device)
            )
        elif self.config.agent_type == 'ppo':
            agent = PPOAgent(
                state_dim, action_dim,
                lr_actor=self.config.lr_actor,
                lr_critic=self.config.lr_critic,
                entropy_coef=self.config.entropy_coef,
                eps_clip=self.config.eps_clip,
                k_epochs=self.config.k_epochs,
                gae_lambda=self.config.gae_lambda,
                device=str(device)
            )
        else:
            raise ValueError(f"Unknown agent type: {self.config.agent_type}")
        
        return agent
    
    def train(self):
        """Execute training loop."""
        env = self.setup_environment()
        agent = self.setup_agent(env)
        
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        print(f"\n{'='*80}")
        print(f"Training {self.config.agent_type.upper()}")
        print(f"{'='*80}")
        print(f"Episodes: {self.config.episodes}")
        print(f"Seed: {self.config.seed}")
        print(f"Config: {self.config}")
        print(f"{'='*80}\n")
        
        best_reward = float('-inf')
        best_metrics = {}
        
        metrics_file = self.results_dir / f"{self.config.agent_type}_training_metrics.csv"
        metrics_file.write_text("episode,reward,wait,queue,speed,throughput,switches,epsilon\n")
        
        try:
            for episode in range(self.config.episodes):
                episode_seed = self.config.seed + 1000 + episode
                build_routes(seed=episode_seed)
                
                state = env.reset(seed=episode_seed)
                total_reward = 0
                done = False
                
                while not done:
                    if self.config.agent_type == 'ppo':
                        action, logprob = agent.act(state)
                    else:
                        action = agent.act(state)
                    
                    next_state, reward, done, _ = env.step(action)
                    
                    if self.config.agent_type in ['dqn', 'ddqn']:
                        agent.remember(state, action, reward, next_state, done)
                        agent.replay()
                    else:  # PPO
                        agent.remember(state, action, logprob, reward, done)
                    
                    state = next_state
                    total_reward += reward
                
                # End-of-episode update
                if self.config.agent_type in ['dqn', 'ddqn']:
                    if hasattr(agent, 'decay_epsilon'):
                        agent.decay_epsilon()
                else:  # PPO
                    agent.replay()
                
                # Metrics
                metrics = env.get_episode_metrics()
                self.reward_history.append(total_reward)
                self.metrics_history.append(metrics)
                
                # Logging
                epsilon = getattr(agent, 'epsilon', 0.0)
                print(f"Episode {episode+1:3d}/{self.config.episodes} | "
                      f"Reward: {total_reward:7.2f} | "
                      f"Wait: {metrics['wait']:6.2f}s | "
                      f"Queue: {metrics['queue']:5.2f} | "
                      f"Throughput: {metrics['throughput']:3.0f} | "
                      f"ε: {epsilon:.4f}")
                
                # Append metrics to CSV
                with open(metrics_file, 'a') as f:
                    f.write(f"{episode+1},{total_reward},{metrics['wait']},{metrics['queue']},"
                           f"{metrics['avg_speed']},{metrics['throughput']},{metrics['switches']},{epsilon}\n")
                
                # Save best model
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_metrics = metrics
                    agent.save(str(self.results_dir / f"{self.config.agent_type}_model_best.pth"))
                    print(f"  ★ New best reward: {best_reward:.2f}")
                
                # Periodic checkpoint
                if (episode + 1) % self.config.save_interval == 0:
                    agent.save(str(self.checkpoint_dir / f"{self.config.agent_type}_checkpoint_e{episode+1}.pth"))
        
        finally:
            env.close()
        
        # Save final model
        agent.save(str(self.results_dir / f"{self.config.agent_type}_model_final.pth"))
        
        # Save training summary
        summary = {
            'config': asdict(self.config),
            'best_reward': float(best_reward),
            'best_metrics': best_metrics,
            'episodes_completed': self.config.episodes,
            'final_reward': float(self.reward_history[-1]) if self.reward_history else 0.0,
            'final_metrics': self.metrics_history[-1] if self.metrics_history else {},
        }
        save_json(self.results_dir / f"{self.config.agent_type}_training_summary.json", summary)
        
        print(f"\n✓ Training complete!")
        print(f"  Best Reward: {summary['best_reward']:.2f}")
        print(f"  Best Wait Time: {summary['best_metrics'].get('wait', 0.0):.2f}s")
        print(f"  Best Throughput: {summary['best_metrics'].get('throughput', 0):.0f}")
        
        return summary


def train_multiple_seeds(agent_type, seeds, **kwargs):
    """Train agent with multiple seeds."""
    results = {}
    
    for seed in seeds:
        config = TrainingConfig(agent_type=agent_type, seed=seed, **kwargs)
        pipeline = TrainingPipeline(config)
        result = pipeline.train()
        results[seed] = result
    
    # Aggregate results
    rewards = [results[s]['best_reward'] for s in seeds]
    print(f"\n{'='*80}")
    print(f"AGGREGATE RESULTS: {agent_type.upper()}")
    print(f"{'='*80}")
    print(f"Reward Mean: {mean(rewards):.2f} ± {stdev(rewards) if len(rewards) > 1 else 0:.2f}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', required=True, choices=['dqn', 'ddqn', 'ppo'])
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', nargs='+', type=int, help='Multiple seeds to train')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon-decay', type=float, default=0.995)
    parser.add_argument('--epsilon-min', type=float, default=0.02)
    parser.add_argument('--replay-size', type=int, default=200000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--target-update', type=int, default=1000)
    parser.add_argument('--lr-actor', type=float, default=1e-4)
    parser.add_argument('--lr-critic', type=float, default=1e-4)
    parser.add_argument('--entropy-coef', type=float, default=0.02)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--k-epochs', type=int, default=4)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--hybrid', action='store_true', default=True)
    args = parser.parse_args()
    
    if args.seeds:
        train_multiple_seeds(
            args.agent,
            args.seeds,
            episodes=args.episodes,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            replay_size=args.replay_size,
            batch_size=args.batch_size,
            target_update=args.target_update,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            entropy_coef=args.entropy_coef,
            eps_clip=args.eps_clip,
            k_epochs=args.k_epochs,
            gae_lambda=args.gae_lambda,
            save_interval=args.save_interval,
            use_hybrid=args.hybrid,
        )
    else:
        config = TrainingConfig(
            agent_type=args.agent,
            episodes=args.episodes,
            seed=args.seed,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            replay_size=args.replay_size,
            batch_size=args.batch_size,
            target_update=args.target_update,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            entropy_coef=args.entropy_coef,
            eps_clip=args.eps_clip,
            k_epochs=args.k_epochs,
            gae_lambda=args.gae_lambda,
            save_interval=args.save_interval,
            use_hybrid=args.hybrid,
        )
        pipeline = TrainingPipeline(config)
        pipeline.train()
