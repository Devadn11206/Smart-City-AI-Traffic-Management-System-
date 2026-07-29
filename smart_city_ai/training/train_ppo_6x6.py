import os
import sys
import logging
import numpy as np
import random
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.sumo_environment_6x6 import SumoEnvironment6x6
from rl_agents.ppo_agent import PPOAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_ppo_6x6():
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
    
    # 2. Initialize 36 standard PPO Agents (Baseline)
    agents = {}
    for ts in ts_ids:
        agents[ts] = PPOAgent(
            state_dim=env.state_dim, 
            action_dim=env.action_dim,
            device=str(device)
        )
        
    start_episode = 0
    model_dir = "models/ppo_6x6"
    if os.path.exists(model_dir):
        ep_dirs = [d for d in os.listdir(model_dir) if d.startswith('ep_')]
        if ep_dirs:
            max_ep = max([int(d.split('_')[1]) for d in ep_dirs])
            logger.info(f"Found checkpoint for episode {max_ep}. Loading...")
            for ts in ts_ids:
                checkpoint_path = os.path.join(model_dir, f"ep_{max_ep}", f"ts_{ts}.pth")
                if os.path.exists(checkpoint_path):
                    try:
                        agents[ts].load(checkpoint_path)
                    except Exception as ex:
                        logger.warning(f"Could not load checkpoint for {ts}: {ex}")
            start_episode = max_ep
            logger.info(f"Resuming training from episode {start_episode + 1}")
            
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    metrics_file = 'ppo_6x6_metrics.csv'
    if start_episode == 0:
        with open(metrics_file, 'w') as f:
            f.write('episode,reward,avg_waiting_time,avg_queue_length,throughput,avg_travel_time,fuel_consumption,avg_speed,loss\n')

    logger.info("Starting Multi-Agent Standard PPO Training Loop (Baseline)...")
    
    try:
        for e in range(start_episode, episodes):
            episode_seed = seed + 5000 + e
            
            states_dict = env.reset(seed=episode_seed)
            total_rewards = {ts: 0.0 for ts in ts_ids}
            done = False
            
            step_count = 0
            
            # Phase 1: Rollout Collection
            while not done:
                # All agents act
                action_dict = {}
                logprob_dict = {}
                for ts in ts_ids:
                    action, logprob = agents[ts].act(states_dict[ts])
                    action_dict[ts] = action
                    logprob_dict[ts] = logprob
                    
                # Environment step
                next_states_dict, rewards_dict, done, _ = env.step(action_dict)
                
                # All agents remember
                for ts in ts_ids:
                    state = states_dict[ts]
                    action = action_dict[ts]
                    logprob = logprob_dict[ts]
                    reward = rewards_dict[ts]
                    
                    agents[ts].remember(state, action, logprob, reward, done)
                    total_rewards[ts] += reward

                states_dict = next_states_dict
                step_count += 1
                
            # Phase 2: PPO Update (End of Episode)
            episode_loss = 0.0
            for ts in ts_ids:
                agents[ts].replay()
                episode_loss += getattr(agents[ts], 'last_loss', 0.0)
            
            mean_reward = np.mean(list(total_rewards.values()))
            avg_loss = episode_loss / max(1, len(ts_ids))
            
            metrics = env.get_episode_metrics()
            wait = metrics.get('wait', 0.0)
            queue = metrics.get('queue', 0.0)
            throughput = metrics.get('throughput', 0.0)
            speed = metrics.get('avg_speed', 0.0)
            fuel = metrics.get('fuel', 0.0)
            travel_time = metrics.get('travel_time', 0.0)
            
            logger.info(f"Episode {e+1}/{episodes} - Reward: {mean_reward:.2f} - Wait: {wait:.2f} - Queue: {queue:.2f}")
            
            with open(metrics_file, 'a') as f:
                f.write(f"{e+1},{mean_reward},{wait},{queue},{throughput},{travel_time},{fuel},{speed},{avg_loss}\n")

            if (e + 1) % 5 == 0:
                for ts in ts_ids:
                    agents[ts].save(f"models/ppo_6x6/ep_{e+1}/ts_{ts}.pth")
                logger.info(f"  Saved periodic checkpoints at episode {e+1} for all {num_agents} agents.")

    except KeyboardInterrupt:
        logger.warning('Training interrupted manually.')
    finally:
        env.close()
        
    logger.info(f"Training completed. Metrics saved to {metrics_file}")

if __name__ == "__main__":
    train_ppo_6x6()
