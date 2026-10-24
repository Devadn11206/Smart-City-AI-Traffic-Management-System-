import os
import sys
import numpy as np
import random
import logging
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dqn_pinn_package.environment.sumo_environment_6x6 import SumoEnvironment6x6

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_dataset_10x10(episodes=10, num_seconds=3600, seed=42, output_file='pinn_dataset_10x10.pt'):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_file = os.path.join(base_dir, "dqn_pinn_package", "sumo_network_10x10", "config.sumocfg")
    
    env = SumoEnvironment6x6(config_file, use_gui=False, num_seconds=num_seconds)
    
    random.seed(seed)
    np.random.seed(seed)
    
    dataset = []
    
    logger.info(f"Generating 10x10 dataset for {episodes} episodes. This will take a few minutes...")
    
    try:
        for e in range(episodes):
            episode_seed = seed + 1000 + e
            
            states_dict = env.reset(seed=episode_seed)
            ts_ids = env.ts_ids
            done = False
            
            step_count = 0
            while not done:
                action_dict = {}
                for ts in ts_ids:
                    action_dict[ts] = random.randrange(env.action_dim)
                    
                next_states_dict, rewards_dict, done, _ = env.step(action_dict)
                
                # Add all 96 transitions to the global physics dataset
                for ts in ts_ids:
                    dataset.append({
                        'state': states_dict[ts],
                        'action': action_dict[ts],
                        'next_state': next_states_dict[ts]
                    })
                
                states_dict = next_states_dict
                step_count += 1
                
            logger.info(f"Episode {e+1}/{episodes} completed. Collected {step_count * len(ts_ids)} transitions.")
    finally:
        env.close()
        
    logger.info(f"Saving dataset with {len(dataset)} transitions to {output_file}...")
    torch.save(dataset, output_file)
    logger.info("Done!")

if __name__ == "__main__":
    generate_dataset_10x10()
