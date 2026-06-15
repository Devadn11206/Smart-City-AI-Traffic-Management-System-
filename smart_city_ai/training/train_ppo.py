import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import logging

logger = logging.getLogger(__name__)

# Adding parent dir to path to import environment and agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.sumo_environment import SumoEnvironment
from rl_agents.ppo_agent import PPOAgent
from scripts.generate_routes import build_routes

def main():
    config_file = "sumo_network/config.sumocfg"
    
    if not os.path.exists(config_file):
        print("Config file not found. Ensure you are running from the top level dir.")
        return

    env = SumoEnvironment(config_file, use_gui=False, num_seconds=3600)
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim,
                     lr_actor=3e-4, lr_critic=3e-4, gae_lambda=0.95, entropy_coef=0.01)

    random.seed(42)
    np.random.seed(42)
    
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    
    reward_history = []
    best_reward = float('-inf')
    
    os.makedirs("models/ppo", exist_ok=True)
    os.makedirs("results/reward_curves", exist_ok=True)
    
    try:
        for e in range(episodes):
            # Multi-Scenario Training: Vary traffic flows every episode!
            episode_seed = 3000 + e
            build_routes(seed=episode_seed)
            
            state = env.reset(seed=episode_seed)
            total_reward = 0
            done = False
            
            step_counter = 0
            while not done:
                action, logprob = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, logprob, reward, done)
                
                state = next_state
                total_reward += reward
                step_counter += 1
                
                if step_counter % 2000 == 0:
                    agent.replay()
            
            # We might have left over memory at the end of the episode
            if len(agent.memory) > 0:
                agent.replay()
                    
            if e % 5 == 0:
                print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward:.2f}")
            reward_history.append(total_reward)
            # Save per-episode metrics for monitoring
            try:
                metrics = env.get_episode_metrics()
                with open('results/reward_curves/ppo_metrics.csv', 'a') as f:
                    if e == 0:
                        f.write('episode,reward,wait,queue,avg_speed,switches\n')
                    f.write(f"{e+1},{total_reward},{metrics['wait']},{metrics['queue']},{metrics['avg_speed']},{metrics['switches']}\n")
            except Exception:
                logger.exception('Failed to write PPO episode metrics for episode %s', e+1)
            
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save("models/ppo/ppo_model_best.pth")
                print(f"  Episode {e+1}: New best reward: {best_reward:.2f}! Saved best model.")
                
            if (e + 1) % 25 == 0:
                agent.save(f"models/ppo/ppo_model_ep_{e+1}.pth")
                print(f"  Saved periodic checkpoint at episode {e+1}")
                
        agent.save("models/ppo/ppo_model_final.pth")
    except Exception as e:
        print(f"Training crashed! Saving backup checkpoint... Error: {e}")
        agent.save("models/ppo/ppo_model_crash_backup.pth")
        raise e
    finally:
        env.close()
    
    plt.plot(reward_history)
    plt.title('PPO Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('results/reward_curves/ppo_reward.png')
    plt.close()

if __name__ == "__main__":
    main()
