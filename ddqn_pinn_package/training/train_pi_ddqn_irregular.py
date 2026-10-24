import os
import sys
import logging
import numpy as np
import random
import torch

# Add the project root to sys.path to import from existing environments
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# We can reuse SumoEnvironment6x6 because it dynamically fetches all traffic lights!
from dqn_pinn_package.environment.sumo_environment_6x6 import SumoEnvironment6x6
from dqn_pinn_package.pinn_models.pinn_ensemble import PINNEnsemble
from ddqn_pinn_package.rl_agents.pi_ddqn_agent import PIDDQNAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def dummy_reward_func(state, action, next_state):
    return -0.1

def train_pi_ddqn_irregular():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_file = os.path.join(base_dir, "dqn_pinn_package", "sumo_network_irregular", "config.sumocfg")
    
    episodes = 200 # Set to 200 episodes for full training
    num_seconds = 3600
    seed = 42
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Initialize Environment
    env = SumoEnvironment6x6(config_file, use_gui=False, num_seconds=num_seconds)
    
    env.start_simulation()
    ts_ids = env.ts_ids
    env.close()
    
    num_agents = len(ts_ids)
    logger.info(f"Detected {num_agents} traffic lights. Training Advanced PI-DDQN on Irregular Network.")
    
    # 2. Global PINN Ensemble (Shared across all intersections)
    pinn_ensemble = PINNEnsemble(state_dim=env.state_dim, action_dim=env.action_dim, num_models=3, device=device)
    logger.warning("No pretrained weights for Irregular Network. Using random initialization for PINN!")
        
    # 3. Initialize PI-DDQN Agents
    agents = {}
    for ts in ts_ids:
        agents[ts] = PIDDQNAgent(
            state_dim=env.state_dim, 
            action_dim=env.action_dim,
            device=str(device),
            pinn_ensemble=pinn_ensemble,
            variance_threshold=0.5,
            real_ratio=0.25,
            lambda_physics=0.1 # Weight of the physics penalty
        )
            
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    metrics_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pi_ddqn_irregular_metrics.csv')
    with open(metrics_file, 'w') as f:
        f.write('episode,reward,avg_waiting_time,avg_queue_length,throughput,avg_travel_time,fuel_consumption,avg_speed,loss\n')

    logger.info("Starting Multi-Agent PI-DDQN Training Loop (Irregular)...")
    
    try:
        for e in range(episodes):
            episode_seed = seed + 5000 + e
            
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
                    
                    # PINN Synthetic Rollouts
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
            
            logger.info(f"PI-DDQN (Irregular) | Episode {e+1}/{episodes} - Reward: {mean_reward:.2f} - Wait: {wait:.2f} - Queue: {queue:.2f}")
            
            with open(metrics_file, 'a') as f:
                f.write(f"{e+1},{mean_reward},{wait},{queue},{throughput},{travel_time},{fuel},{speed},{avg_loss}\n")

    except KeyboardInterrupt:
        logger.warning('Training interrupted manually.')
    finally:
        env.close()
        
    logger.info(f"Advanced PI-DDQN Irregular Training completed. Metrics saved to {metrics_file}")

if __name__ == "__main__":
    train_pi_ddqn_irregular()
