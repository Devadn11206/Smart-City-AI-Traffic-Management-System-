import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_diagnostics():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ppo_dir = os.path.join(base_dir, 'results', 'ppo')
    
    ep_file = os.path.join(ppo_dir, '2x2_episode_metrics.csv')
    ts_file = os.path.join(ppo_dir, '2x2_intersection_metrics.csv')
    
    if not os.path.exists(ep_file) or not os.path.exists(ts_file):
        print("Log files not found!")
        return
        
    ep_df = pd.read_csv(ep_file)
    ts_df = pd.read_csv(ts_file)
    
    os.makedirs(os.path.join(ppo_dir, 'plots'), exist_ok=True)
    
    # Separate eval and training episodes
    if 'eval_mode' in ep_df.columns:
        train_df = ep_df[ep_df['eval_mode'] == False].copy()
        eval_df = ep_df[ep_df['eval_mode'] == True].copy()
    else:
        train_df = ep_df.copy()
        eval_df = pd.DataFrame(columns=ep_df.columns)

    # Convert columns to numeric if they are not
    metrics_to_convert = ['actor_loss', 'critic_loss', 'entropy', 'kl_div', 'clip_frac', 'explained_var']
    for metric in metrics_to_convert:
        if metric in train_df.columns:
            train_df[metric] = pd.to_numeric(train_df[metric], errors='coerce')
    
    # 1. Reward Trend
    plt.figure(figsize=(10, 5))
    plt.plot(train_df['episode'], train_df['reward'], label='Training Reward')
    if not eval_df.empty:
        plt.scatter(eval_df['episode'], eval_df['reward'], color='red', label='Eval Reward')
    plt.title('PPO Reward Trend')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ppo_dir, 'plots', 'reward_trend.png'))
    plt.close()
    
    # 2. Actor and Critic Loss Trend
    if 'actor_loss' in train_df.columns and 'critic_loss' in train_df.columns:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Actor Loss', color='tab:blue')
        ax1.plot(train_df['episode'], train_df['actor_loss'], color='tab:blue', label='Actor Loss')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Critic Loss', color='tab:orange')
        ax2.plot(train_df['episode'], train_df['critic_loss'], color='tab:orange', label='Critic Loss')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        
        plt.title('PPO Loss Trends')
        fig.tight_layout()
        plt.savefig(os.path.join(ppo_dir, 'plots', 'loss_trend.png'))
        plt.close()
        
    # 3. Entropy Trend
    if 'entropy' in train_df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(train_df['episode'], train_df['entropy'], color='green')
        plt.title('PPO Entropy Trend')
        plt.xlabel('Episode')
        plt.ylabel('Entropy')
        plt.grid(True)
        plt.savefig(os.path.join(ppo_dir, 'plots', 'entropy_trend.png'))
        plt.close()
        
    # 4. Action Distribution per phase
    plt.figure(figsize=(10, 5))
    action_counts = ts_df['selected_action'].value_counts().sort_index()
    action_counts.plot(kind='bar', color='purple')
    plt.title('Overall Action Distribution')
    plt.xlabel('Action (Phase)')
    plt.ylabel('Count')
    plt.grid(axis='y')
    plt.savefig(os.path.join(ppo_dir, 'plots', 'action_distribution.png'))
    plt.close()
    
    print("Plots generated successfully in results/2x2/ppo/plots/")

if __name__ == '__main__':
    plot_diagnostics()
