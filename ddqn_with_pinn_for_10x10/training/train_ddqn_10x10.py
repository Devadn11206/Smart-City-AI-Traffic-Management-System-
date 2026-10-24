import os
import sys
import logging
import time
import numpy as np
import random
import torch

# Add the project root to sys.path to import from existing environments
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dqn_pinn_package.environment.sumo_environment_6x6 import SumoEnvironment6x6
from ddqn_pinn_package.rl_agents.ddqn_agent import DDQNAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints_baseline_10x10')

def save_checkpoint(agents, ts_ids, episode, epsilon_dict):
    """Save all agent model weights + epsilon values for a given episode."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt = {
        'episode': episode,
        'epsilon': epsilon_dict,
    }
    for ts in ts_ids:
        ckpt[f'model_{ts}'] = agents[ts].model.state_dict()
        ckpt[f'target_{ts}'] = agents[ts].target_model.state_dict()
    save_path = os.path.join(CHECKPOINT_DIR, f'ckpt_ep{episode}.pt')
    torch.save(ckpt, save_path)
    # Keep only the last 2 checkpoints to save disk space while avoiding deletion of the just-written file.
    existing = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('ckpt_ep') and f.endswith('.pt')],
        key=lambda x: int(x.replace('ckpt_ep', '').replace('.pt', '')),
    )
    for old in existing[:-2]:
        old_path = os.path.join(CHECKPOINT_DIR, old)
        if os.path.exists(old_path) and os.path.abspath(old_path) != os.path.abspath(save_path):
            os.remove(old_path)
    logger.info(f"Checkpoint saved at episode {episode} -> {save_path}")

def load_latest_checkpoint(agents, ts_ids):
    """Load the most recent checkpoint. Returns the episode number to resume from."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return 0
    files = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('ckpt_ep') and f.endswith('.pt')],
        key=lambda x: int(x.replace('ckpt_ep', '').replace('.pt', '')),
    )
    if not files:
        return 0
    latest = os.path.join(CHECKPOINT_DIR, files[-1])
    ckpt = torch.load(latest, map_location='cpu', weights_only=False)
    resume_episode = ckpt['episode']
    for ts in ts_ids:
        if f'model_{ts}' in ckpt:
            agents[ts].model.load_state_dict(ckpt[f'model_{ts}'])
            agents[ts].target_model.load_state_dict(ckpt[f'target_{ts}'])
            agents[ts].epsilon = ckpt['epsilon'].get(ts, agents[ts].epsilon)
    logger.info(f"Resumed from checkpoint: episode {resume_episode} ({latest})")
    return resume_episode

def train_ddqn_10x10():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_file = os.path.join(base_dir, "dqn_pinn_package", "sumo_network_10x10", "config.sumocfg")
    
    episodes = 200
    num_seconds = 3600
    seed = 42
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Initialize Environment to discover traffic lights
    env = SumoEnvironment6x6(config_file, use_gui=False, num_seconds=num_seconds)
    env.start_simulation()
    ts_ids = env.ts_ids
    env.close()
    time.sleep(2)  # give OS time to release TCP port
    
    num_agents = len(ts_ids)
    logger.info(f"Detected {num_agents} traffic lights. Training Baseline DDQN on 10x10.")
    
    # 2. Initialize DDQN Agents
    agents = {}
    for ts in ts_ids:
        agents[ts] = DDQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            device=str(device)
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 3. Load checkpoint if available (resume training)
    start_episode = load_latest_checkpoint(agents, ts_ids)

    metrics_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ddqn_baseline_10x10_metrics.csv')

    # Only write header if starting fresh
    if start_episode == 0:
        with open(metrics_file, 'w') as f:
            f.write('episode,reward,avg_waiting_time,avg_queue_length,throughput,avg_travel_time,fuel_consumption,avg_speed,loss\n')
        logger.info("Starting fresh training from Episode 1.")
    else:
        logger.info(f"Resuming training from Episode {start_episode + 1}. Appending to existing metrics.")

    logger.info("Starting Multi-Agent Baseline DDQN Training Loop (10x10)...")
    
    try:
        for e in range(start_episode, episodes):
            episode_seed = seed + 4000 + e
            
            states_dict = env.reset(seed=episode_seed)
            total_rewards = {ts: 0.0 for ts in ts_ids}
            done = False
            
            step_count = 0
            episode_loss = 0.0
            
            while not done:
                action_dict = {}
                for ts in ts_ids:
                    action_dict[ts] = agents[ts].act(states_dict[ts])
                    
                next_states_dict, rewards_dict, done, _ = env.step(action_dict)
                
                step_loss = 0.0
                for ts in ts_ids:
                    state = states_dict[ts]
                    action = action_dict[ts]
                    reward = rewards_dict[ts]
                    next_state = next_states_dict[ts]
                    
                    agents[ts].remember(state, action, reward, next_state, done)
                    agents[ts].replay()
                    
                    step_loss += getattr(agents[ts], 'last_loss', 0.0)
                    total_rewards[ts] += reward
                    
                episode_loss += step_loss / max(1, len(ts_ids))
                states_dict = next_states_dict
                step_count += 1
                
            for ts in ts_ids:
                try: agents[ts].decay_epsilon()
                except Exception: pass

            mean_reward = np.mean(list(total_rewards.values()))
            metrics = env.get_episode_metrics()
            wait = metrics.get('wait', 0.0)
            queue = metrics.get('queue', 0.0)
            throughput = metrics.get('throughput', 0.0)
            speed = metrics.get('avg_speed', 0.0)
            fuel = metrics.get('fuel', 0.0)
            travel_time = metrics.get('travel_time', 0.0)
            avg_loss = episode_loss / max(1, step_count)
            
            logger.info(f"DDQN Baseline (10x10) | Episode {e+1}/{episodes} - Reward: {mean_reward:.2f} - Wait: {wait:.2f} - Queue: {queue:.2f}")
            
            with open(metrics_file, 'a') as f:
                f.write(f"{e+1},{mean_reward},{wait},{queue},{throughput},{travel_time},{fuel},{speed},{avg_loss}\n")

            # Save checkpoint every 5 episodes
            if (e + 1) % 5 == 0:
                epsilon_dict = {ts: agents[ts].epsilon for ts in ts_ids}
                save_checkpoint(agents, ts_ids, e + 1, epsilon_dict)

            # Sleep between episodes to let Windows release the TCP port (prevents socket crash)
            time.sleep(1)

    except KeyboardInterrupt:
        logger.warning('Training interrupted manually. Saving checkpoint...')
        epsilon_dict = {ts: agents[ts].epsilon for ts in ts_ids}
        save_checkpoint(agents, ts_ids, e + 1, epsilon_dict)
    finally:
        env.close()
        
    logger.info(f"Baseline DDQN 10x10 Training completed. Metrics saved to {metrics_file}")

if __name__ == "__main__":
    train_ddqn_10x10()
