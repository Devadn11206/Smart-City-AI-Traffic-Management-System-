import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Paths setup
current_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.dirname(current_dir)

def plot_results(pg_dqn_csv, standard_dqn_csv=None, output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_pg = pd.read_csv(pg_dqn_csv)
    
    metrics = ['Reward', 'AvgWait', 'Queue', 'Throughput', 'Fuel', 'Speed', 'TravelTime', 'Loss', 'AvgQ', 'TargetQ', 'TDError', 'GradNorm']
    
    for metric in metrics:
        if metric not in df_pg.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        plt.plot(df_pg['Episode'], df_pg[metric], label='AC-PG-DQN', alpha=0.8)
        
        # Plot smoothed
        plt.plot(df_pg['Episode'], df_pg[metric].rolling(window=10).mean(), color='blue', linewidth=2)
        
        if standard_dqn_csv and os.path.exists(standard_dqn_csv):
            df_std = pd.read_csv(standard_dqn_csv)
            if metric in df_std.columns:
                plt.plot(df_std['Episode'], df_std[metric], label='Standard DQN', alpha=0.8)
                plt.plot(df_std['Episode'], df_std[metric].rolling(window=10).mean(), color='orange', linewidth=2)
                
        plt.title(f'Training Comparison: {metric}')
        plt.xlabel('Episode')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_path = os.path.join(output_dir, f"{metric}_comparison.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_pg_dqn_results.py <path_to_pg_dqn_metrics.csv> [path_to_standard_dqn_metrics.csv]")
        sys.exit(1)
        
    pg_csv = sys.argv[1]
    std_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_results(pg_csv, std_csv)
