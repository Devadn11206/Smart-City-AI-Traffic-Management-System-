import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def plot_comparison():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dqn_metrics_path = os.path.join(base_dir, 'dqn_6x6_metrics.csv')
    pinn_metrics_path = os.path.join(base_dir, 'pi_mbpo_6x6_metrics.csv')
    
    if not os.path.exists(dqn_metrics_path) or not os.path.exists(pinn_metrics_path):
        print("Error: Metrics CSV files not found.")
        sys.exit(1)
        
    df_dqn = pd.read_csv(dqn_metrics_path)
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
    
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    for metric_col, title, ylabel in metrics:
        if metric_col not in df_dqn.columns or metric_col not in df_pinn.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        
        plt.plot(df_dqn['episode'], df_dqn[metric_col], label='Standard DQN', color='tab:blue', alpha=0.8, linewidth=2)
        plt.plot(df_pinn['episode'], df_pinn[metric_col], label='PINN-Guided DQN (Ours)', color='tab:orange', alpha=0.8, linewidth=2)
        
        if len(df_dqn) >= 10:
            dqn_rolling = df_dqn[metric_col].rolling(window=10).mean()
            pinn_rolling = df_pinn[metric_col].rolling(window=10).mean()
            plt.plot(df_dqn['episode'], dqn_rolling, color='darkblue', linestyle='--', linewidth=2.5, alpha=0.9, label='DQN Trend (MA-10)')
            plt.plot(df_pinn['episode'], pinn_rolling, color='darkorange', linestyle='--', linewidth=2.5, alpha=0.9, label='PINN Trend (MA-10)')
        
        plt.title(f'Comparison: {title}', fontsize=16, fontweight='bold')
        plt.xlabel('Training Episode', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        
        save_path = os.path.join(plots_dir, f'dqn_vs_pinn_{metric_col}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {save_path}")
        
    print(f"\nAll comparative plots have been generated and saved to {plots_dir}")

if __name__ == '__main__':
    plot_comparison()
