import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def plot_comparison():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ppo_metrics_path = os.path.join(base_dir, 'ppo_6x6_metrics.csv')
    pinn_metrics_path = os.path.join(base_dir, 'ppo_pinn_6x6_metrics.csv')
    
    if not os.path.exists(ppo_metrics_path) or not os.path.exists(pinn_metrics_path):
        print("Error: Metrics CSV files not found. Please ensure both train_ppo_6x6.py and train_ppo_pinn_6x6.py have generated data.")
        sys.exit(1)
        
    df_ppo = pd.read_csv(ppo_metrics_path)
    df_pinn = pd.read_csv(pinn_metrics_path)
    
    metrics = [
        ('reward', 'Training Reward Convergence', 'Reward'),
        ('avg_waiting_time', 'Average Waiting Time', 'Wait Time (s)'),
        ('avg_queue_length', 'Average Queue Length', 'Queue Length (Vehicles)'),
        ('throughput', 'Throughput', 'Throughput (Vehicles)'),
        ('avg_travel_time', 'Average Travel Time', 'Travel Time (s)'),
        ('fuel_consumption', 'Fuel Consumption', 'Fuel (mg/s)'),
        ('avg_speed', 'Average Speed', 'Speed (m/s)'),
        ('loss', 'Neural Network Loss', 'Loss')
    ]
    
    # Ensure plots directory exists
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    for metric_col, title, ylabel in metrics:
        if metric_col not in df_ppo.columns or metric_col not in df_pinn.columns:
            print(f"Warning: Metric '{metric_col}' not found in one of the CSV files. Skipping.")
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Plot standard PPO
        plt.plot(df_ppo['episode'], df_ppo[metric_col], label='Standard PPO', color='tab:blue', alpha=0.8, linewidth=2)
        
        # Plot PINN-Guided PPO
        plt.plot(df_pinn['episode'], df_pinn[metric_col], label='PINN-Guided PPO (Ours)', color='tab:orange', alpha=0.8, linewidth=2)
        
        # Add rolling averages for smoothing if we have enough episodes
        if len(df_ppo) >= 10:
            ppo_rolling = df_ppo[metric_col].rolling(window=10).mean()
            pinn_rolling = df_pinn[metric_col].rolling(window=10).mean()
            plt.plot(df_ppo['episode'], ppo_rolling, color='darkblue', linestyle='--', linewidth=2.5, alpha=0.9, label='PPO Trend (MA-10)')
            plt.plot(df_pinn['episode'], pinn_rolling, color='darkorange', linestyle='--', linewidth=2.5, alpha=0.9, label='PINN Trend (MA-10)')
        
        plt.title(f'Comparison: {title}', fontsize=16, fontweight='bold')
        plt.xlabel('Training Episode', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        
        save_path = os.path.join(plots_dir, f'ppo_vs_pinn_{metric_col}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {save_path}")
        
    print(f"\nAll comparative plots have been generated and saved to {plots_dir}")

if __name__ == '__main__':
    plot_comparison()
