import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random

# Adding parent dir to path to import environment and agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.sumo_environment import SumoEnvironment
from rl_agents.dqn_agent import DQNAgent
from scripts.generate_routes import build_routes

def main():
    # Adjust the network and route files paths according to step 1
    config_file = "sumo_network/config.sumocfg"
    
    if not os.path.exists(config_file):
        print("Config file not found. Ensure you are running from the top level dir.")
        return

    env = SumoEnvironment(config_file, use_gui=False, num_seconds=3600)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    random.seed(42)
    np.random.seed(42)
    
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    reward_history = []
    best_reward = float('-inf')
    
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("results/reward_curves", exist_ok=True)
    
    try:
        for e in range(episodes):
            # Multi-Scenario Training: Vary traffic flows every episode!
            episode_seed = 1000 + e
            build_routes(seed=episode_seed)
            
            state = env.reset(seed=episode_seed)
            total_reward = 0
            done = False
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                
                state = next_state
                total_reward += reward
                
            if e % 5 == 0:
                agent.update_target_network()
                
            agent.decay_epsilon()
                
            print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            reward_history.append(total_reward)
            
            if total_reward > best_reward:
                best_reward = total_reward
                agent.save("models/dqn/dqn_model_best.pth")
                print(f"  New best reward: {best_reward:.2f}! Saved best model.")
            
            if (e + 1) % 5 == 0:
                agent.save(f"models/dqn/dqn_model_ep_{e+1}.pth")
                print(f"  Saved periodic checkpoint at episode {e+1}")
            
        agent.save("models/dqn/dqn_model_final.pth")
    except Exception as e:
        print(f"Training crashed! Saving backup checkpoint... Error: {e}")
        agent.save("models/dqn/dqn_model_crash_backup.pth")
        raise e
    finally:
        env.close()
    
    # Plotting
    plt.plot(reward_history)
    plt.title('DQN Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('results/reward_curves/dqn_reward.png')
    plt.close()

if __name__ == "__main__":
    main()
