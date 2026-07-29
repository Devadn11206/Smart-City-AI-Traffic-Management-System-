import os
import sys
import numpy as np
import random
import logging
from pathlib import Path
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.sumo_environment import SumoEnvironment
from scripts.generate_routes import build_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_dataset(episodes=20, num_seconds=3600, seed=42, output_file='pinn_dataset.pt'):
    """
    Generates a dataset of (state, action, next_state) by running random actions in SUMO.
    """
    # Dynamically locate the sumo_network config file relative to this script
    base_dir = os.path.dirname(os.path.dirname(__file__)) # gets you to smart_city_ai/
    config_file = os.path.join(base_dir, "sumo_network", "config.sumocfg")
    
    env = SumoEnvironment(config_file, use_gui=False, num_seconds=num_seconds)
    
    random.seed(seed)
    np.random.seed(seed)
    
    dataset = []
    
    logger.info(f"Generating dataset for {episodes} episodes. This will take a few minutes...")
    
    try:
        for e in range(episodes):
            episode_seed = seed + 1000 + e
            build_routes(seed=episode_seed)
            
            state = env.reset(seed=episode_seed)
            done = False
            
            step_count = 0
            while not done:
                # Use a random policy to explore the state space
                action = random.randrange(env.action_dim)
                next_state, reward, done, _ = env.step(action)
                
                dataset.append({
                    'state': state,
                    'action': action,
                    'next_state': next_state
                })
                
                state = next_state
                step_count += 1
                
            logger.info(f"Episode {e+1}/{episodes} completed. Collected {step_count} transitions.")
    finally:
        env.close()
        
    # Save to file
    logger.info(f"Saving dataset with {len(dataset)} transitions to {output_file}...")
    torch.save(dataset, output_file)
    logger.info("Done!")

if __name__ == "__main__":
    generate_dataset()
