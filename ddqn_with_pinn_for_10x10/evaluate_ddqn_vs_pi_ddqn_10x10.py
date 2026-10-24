import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def load_metrics(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: Could not find {filepath}")
        return None
    return pd.read_csv(filepath)

def plot_comparison():
    base_dir = os.path.dirname(__file__)
    baseline_path = os.path.join(base_dir, 'ddqn_baseline_10x10_metrics.csv')
    pi_ddqn_path = os.path.join(base_dir, 'pi_ddqn_10x10_metrics.csv')
    
    baseline_df = load_metrics(baseline_path)
    pi_df = load_metrics(pi_ddqn_path)
    
    if baseline_df is None or pi_df is None:
        print("Cannot plot comparison: Missing one or both metrics files. Please run both training scripts first.")
        return
        
    sns.set_theme(style="darkgrid")
    
    metrics = [
        ('reward', 'Total Reward (10x10)', 'Higher is better'),
        ('avg_waiting_time', 'Average Waiting Time (s) (10x10)', 'Lower is better'),
        ('avg_queue_length', 'Average Queue Length (vehicles) (10x10)', 'Lower is better'),
        ('throughput', 'Throughput (vehicles completed) (10x10)', 'Higher is better'),
        ('avg_travel_time', 'Average Travel Time (s) (10x10)', 'Lower is better'),
        ('loss', 'Neural Network Loss (10x10)', 'Convergence indicator')
    ]
    
    for metric, title, ylabel in metrics:
        if metric not in baseline_df.columns or metric not in pi_df.columns:
            print(f"Skipping {metric} because it's not present in the CSV files.")
            continue

        plt.figure(figsize=(10, 6))
        
        plt.plot(baseline_df['episode'], baseline_df[metric], label='Baseline DDQN (10x10)', linewidth=2.5, color='coral', marker='o', markersize=5)
        plt.plot(pi_df['episode'], pi_df[metric], label='Physics-Informed DDQN (PI-DDQN) (10x10)', linewidth=2.5, color='dodgerblue', marker='s', markersize=5)
        
        plt.title(f'Performance Comparison: {title}', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save each plot separately
        save_path = os.path.join(base_dir, f'ddqn_vs_pi_ddqn_10x10_{metric}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() # Close the figure to free up memory
        print(f"Saved {title} plot to {save_path}")
    
if __name__ == '__main__':
    plot_comparison()
