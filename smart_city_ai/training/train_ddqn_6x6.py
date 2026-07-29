import os
import sys
import logging
import numpy as np
import random
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.sumo_environment_6x6 import SumoEnvironment6x6
from rl_agents.ddqn_agent import DDQNAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_ddqn_6x6():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_file = os.path.join(base_dir, "sumo_network_6x6", "config.sumocfg")
    
    episodes = 200
    num_seconds = 3600
    seed = 42
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Initialize 6x6 Environment
    env = SumoEnvironment6x6(config_file, use_gui=False, num_seconds=num_seconds)
    
    # Initialize SUMO briefly to get ts_ids
    env.start_simulation()
    ts_ids = env.ts_ids
    env.close()
    
    num_agents = len(ts_ids)
    logger.info(f"Detected {num_agents} traffic lights in the 6x6 network.")
    
    # 2. Initialize 36 standard DDQN Agents (Baseline)
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

    metrics_file = 'ddqn_6x6_metrics.csv'
    with open(metrics_file, 'w') as f:
        f.write('Episode,MeanReward\n')

    logger.info("Starting Multi-Agent DDQN Training Loop (Baseline)...")
    
    try:
        for e in range(episodes):
            episode_seed = seed + 4000 + e
            
            states_dict = env.reset(seed=episode_seed)
            total_rewards = {ts: 0.0 for ts in ts_ids}
            done = False
            
            step_count = 0
            while not done:
                # All agents act
                action_dict = {}
                for ts in ts_ids:
                    action_dict[ts] = agents[ts].act(states_dict[ts])
                    
                # Environment step
                next_states_dict, rewards_dict, done, _ = env.step(action_dict)
                
                # All agents learn
                for ts in ts_ids:
                    state = states_dict[ts]
                    action = action_dict[ts]
                    reward = rewards_dict[ts]
                    next_state = next_states_dict[ts]
                    
                    agents[ts].remember(state, action, reward, next_state, done)
                    agents[ts].replay()
                    total_rewards[ts] += reward

                states_dict = next_states_dict
                step_count += 1
                
            for ts in ts_ids:
                try: agents[ts].decay_epsilon()
                except Exception: pass

            mean_reward = np.mean(list(total_rewards.values()))
            logger.info(f"Episode {e+1}/{episodes} - Mean Reward across {num_agents} agents: {mean_reward:.2f}")
            
            with open(metrics_file, 'a') as f:
                f.write(f"{e+1},{mean_reward}\n")
                
            if (e + 1) % 5 == 0:
                checkpoint_dir = os.path.join(base_dir, "checkpoints", f"episode_{e+1}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                for ts in ts_ids:
                    agents[ts].save(os.path.join(checkpoint_dir, f"{ts}_ddqn.pth"))
                logger.info(f"Checkpoints saved at episode {e+1} in {checkpoint_dir}")

    except KeyboardInterrupt:
        logger.warning('Training interrupted manually.')
    finally:
        env.close()
        
    logger.info(f"Training completed. Metrics saved to {metrics_file}")

if __name__ == "__main__":
    train_ddqn_6x6()
