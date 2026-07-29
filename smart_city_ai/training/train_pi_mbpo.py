import os
import sys
import json
import logging
import numpy as np
import random
from pathlib import Path
import torch

# Adding parent dir to path to import environment and agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.sumo_environment import SumoEnvironment
from pinn_models.pinn_ensemble import PINNEnsemble
from rl_agents.pi_mbpo_agent import PIMBPOAgent
from scripts.generate_routes import build_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def dummy_reward_func(state, action, next_state):
    """Fallback proxy reward if environment reward func is not accessible."""
    return -0.1

def train_pi_mbpo():
    # Dynamically locate the sumo_network config file relative to this script
    base_dir = os.path.dirname(os.path.dirname(__file__)) # gets you to smart_city_ai/
    config_file = os.path.join(base_dir, "sumo_network", "config.sumocfg")
    episodes = 200  # Strictly 200 episodes as requested
    num_seconds = 3600
    seed = 42
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Initialize Environment
    env = SumoEnvironment(config_file, use_gui=False, num_seconds=num_seconds)
    
    # 2. Initialize PINN Ensemble World Model (Offline Phase Placeholder)
    logger.info("Initializing PINN Ensemble World Model...")
    pinn_ensemble = PINNEnsemble(state_dim=env.state_dim, action_dim=env.action_dim, num_models=3, device=device)
    
    # Note: In a full pipeline, you would load pre-trained weights here:
    pretrained_path = "pinn_ensemble_pretrained.pth"
    if os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained PINN ensemble weights from {pretrained_path}...")
        pinn_ensemble.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        logger.warning(f"No pretrained weights found at {pretrained_path}. Using random initialization!")
    
    # 3. Initialize PI-MBPO Agent
    agent = PIMBPOAgent(
        state_dim=env.state_dim, 
        action_dim=env.action_dim,
        lr=1e-4, 
        gamma=0.99, 
        epsilon=1.0, 
        epsilon_decay=0.995, 
        epsilon_min=0.02,
        replay_size=200000, 
        batch_size=256, 
        target_update=1000, 
        device=str(device),
        pinn_ensemble=pinn_ensemble,
        variance_threshold=0.5, # adjust based on state scale
        real_ratio=0.25
    )
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    metrics_file = 'pi_mbpo_metrics.csv'
    # Requested metrics: Episode, Reward, AvgWait, Queue, Throughput, Fuel, Speed, TravelTime
    with open(metrics_file, 'w') as f:
        f.write('Episode,Reward,AvgWait,Queue,Throughput,Fuel,Speed,TravelTime\n')

    logger.info("Starting PI-MBPO Training Loop...")
    
    try:
        for e in range(episodes):
            episode_seed = seed + 3000 + e
            build_routes(seed=episode_seed)

            state = env.reset(seed=episode_seed)
            total_reward = 0
            done = False
            
            while not done:
                # Agent acts
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                
                # Save to real buffer
                agent.remember(state, action, reward, next_state, done)
                
                # Imagination Step (Dyna rollout)
                agent.generate_synthetic_data(num_samples=10, env_reward_func=dummy_reward_func)
                
                # Update Q-Network using dual buffers
                agent.replay()

                state = next_state
                total_reward += reward

            try:
                agent.decay_epsilon()
            except Exception:
                pass

            metrics = env.get_episode_metrics()
            wait = metrics.get('wait', 0.0)
            queue = metrics.get('queue', 0.0)
            throughput = metrics.get('throughput', 0.0)
            speed = metrics.get('avg_speed', 0.0)
            fuel = 0.0        # Proxy/Not available natively in basic metrics
            travel_time = 0.0 # Proxy/Not available natively in basic metrics
            
            logger.info(f"Episode {e+1}/{episodes} - Reward: {total_reward:.2f} - Wait: {wait:.2f} - Queue: {queue:.2f}")
            
            with open(metrics_file, 'a') as f:
                f.write(f"{e+1},{total_reward},{wait},{queue},{throughput},{fuel},{speed},{travel_time}\n")
                
            if (e + 1) % 5 == 0:
                agent.save(f"models/pi_mbpo/pi_mbpo_model_ep_{e+1}.pth")
                logger.info(f"  Saved periodic checkpoint at episode {e+1}")

    except KeyboardInterrupt:
        logger.warning('Training interrupted manually.')
    finally:
        env.close()
        
    logger.info(f"Training completed. Metrics saved to {metrics_file}")

if __name__ == "__main__":
    train_pi_mbpo()
