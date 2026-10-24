import os
import sys
import logging
import numpy as np
import random
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'smart_city_ai')))

from environment.sumo_environment_6x6 import SumoEnvironment6x6
from pinn_models.pinn_ensemble import PINNEnsemble
from rl_agents.pi_mbpo_agent import PIMBPOAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def dummy_reward_func(state, action, next_state):
    return -0.1

def train_pi_mbpo_6x6():
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
    
    # 2. Global PINN Ensemble (Shared across all intersections to learn global traffic physics)
    pinn_ensemble = PINNEnsemble(state_dim=env.state_dim, action_dim=env.action_dim, num_models=3, device=device)
    pretrained_path = "pinn_ensemble_pretrained_6x6.pth"
    if os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained 6x6 PINN weights from {pretrained_path}...")
        pinn_ensemble.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        logger.warning(f"No pretrained weights found at {pretrained_path}. Using random initialization!")
    
    # 3. Initialize 36 PI-MBPO Agents
    agents = {}
    for ts in ts_ids:
        agents[ts] = PIMBPOAgent(
            state_dim=env.state_dim, 
            action_dim=env.action_dim,
            device=str(device),
            pinn_ensemble=pinn_ensemble, # Share the global PINN
            variance_threshold=0.5,
            real_ratio=0.25
        )
        
    start_episode = 0
    model_dir = "models/pi_mbpo_6x6"
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

    metrics_file = 'pi_mbpo_6x6_metrics.csv'
    if start_episode == 0:
        with open(metrics_file, 'w') as f:
            f.write('episode,reward,avg_waiting_time,avg_queue_length,throughput,avg_travel_time,fuel_consumption,avg_speed,loss\n')

    logger.info("Starting Multi-Agent PI-MBPO Training Loop...")
    
    try:
        for e in range(start_episode, episodes):
            episode_seed = seed + 3000 + e
            
            states_dict = env.reset(seed=episode_seed)
            total_rewards = {ts: 0.0 for ts in ts_ids}
            done = False
            
            step_count = 0
            episode_loss = 0.0
            while not done:
                # All agents act
                action_dict = {}
                for ts in ts_ids:
                    action_dict[ts] = agents[ts].act(states_dict[ts])
                    
                # Environment step
                next_states_dict, rewards_dict, done, _ = env.step(action_dict)
                
                # All agents learn
                step_loss = 0.0
                for ts in ts_ids:
                    state = states_dict[ts]
                    action = action_dict[ts]
                    reward = rewards_dict[ts]
                    next_state = next_states_dict[ts]
                    
                    agents[ts].remember(state, action, reward, next_state, done)
                    
                    # Imagination Step (1-step rollout using global PINN)
                    agents[ts].generate_synthetic_data(num_samples=5, env_reward_func=dummy_reward_func)
                    
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
            
            logger.info(f"Episode {e+1}/{episodes} - Reward: {mean_reward:.2f} - Wait: {wait:.2f} - Queue: {queue:.2f}")
            
            with open(metrics_file, 'a') as f:
                f.write(f"{e+1},{mean_reward},{wait},{queue},{throughput},{travel_time},{fuel},{speed},{avg_loss}\n")

            if (e + 1) % 5 == 0:
                for ts in ts_ids:
                    agents[ts].save(f"models/pi_mbpo_6x6/ep_{e+1}/ts_{ts}.pth")
                logger.info(f"  Saved periodic checkpoints at episode {e+1} for all {num_agents} agents.")

    except KeyboardInterrupt:
        logger.warning('Training interrupted manually.')
    finally:
        env.close()
        
    logger.info(f"Training completed. Metrics saved to {metrics_file}")

if __name__ == "__main__":
    train_pi_mbpo_6x6()
