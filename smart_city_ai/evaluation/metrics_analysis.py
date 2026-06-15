import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(dqn_rewards, ddqn_rewards, ppo_rewards, filename="results/comparison_charts/learning_curves.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(dqn_rewards, label='DQN')
    plt.plot(ddqn_rewards, label='DDQN')
    plt.plot(ppo_rewards, label='PPO')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('RL Training Convergence Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved learning curves to {filename}")

def plot_performance_metrics(labels, metrics_data, title, ylabel, filename):
    x = np.arange(len(labels))
    plt.figure(figsize=(8, 5))
    plt.bar(x, metrics_data, color=['gray', 'blue', 'orange', 'green'])
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    print(f"Saved metric plot to {filename}")
