#!/usr/bin/env python3
"""
Plot hybrid training curves and comparison charts.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("darkgrid")

def plot_hybrid_training_curves():
    repo_root = Path(__file__).parent.parent
    log_file = repo_root / "results/hybrid_logs/hybrid_training_log.csv"
    output_dir = repo_root / "results/hybrid_logs"
    
    if not log_file.exists():
        fallback_log = repo_root.parent / "results" / "hybrid_logs" / "hybrid_training_log.csv"
        if fallback_log.exists():
            log_file = fallback_log
            output_dir = repo_root / "results" / "hybrid_logs"
            print(f"Using fallback log file: {fallback_log}")
        else:
            print(f"Log file not found: {log_file}")
            return
    
    try:
        df = pd.read_csv(log_file)
    except Exception as e:
        print(f"Error reading log: {e}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    agents = sorted(df['agent'].unique())
    
    # 1. Reward curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for agent in agents:
        agent_data = df[df['agent'] == agent].sort_values('episode')
        # Smooth with rolling average
        smoothed = agent_data['reward'].rolling(window=10, center=True).mean()
        ax.plot(agent_data['episode'], smoothed, label=f"{agent.upper()}", linewidth=2, marker='o', markersize=2, alpha=0.7)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title("Hybrid Training Reward Curves (10-ep moving avg)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_reward_curves.png", dpi=150)
    print(f"Saved: training_reward_curves.png")
    plt.close()
    
    # 2. Waiting time curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for agent in agents:
        agent_data = df[df['agent'] == agent].sort_values('episode')
        smoothed = agent_data['wait'].rolling(window=10, center=True).mean()
        ax.plot(agent_data['episode'], smoothed, label=f"{agent.upper()}", linewidth=2, marker='o', markersize=2, alpha=0.7)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Avg Waiting Time (s)", fontsize=12)
    ax.set_title("Hybrid Training Waiting Time Curves (10-ep moving avg)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_wait_curves.png", dpi=150)
    print(f"Saved: training_wait_curves.png")
    plt.close()
    
    # 3. Queue length curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for agent in agents:
        agent_data = df[df['agent'] == agent].sort_values('episode')
        smoothed = agent_data['queue'].rolling(window=10, center=True).mean()
        ax.plot(agent_data['episode'], smoothed, label=f"{agent.upper()}", linewidth=2, marker='o', markersize=2, alpha=0.7)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Avg Queue Length (vehicles)", fontsize=12)
    ax.set_title("Hybrid Training Queue Length Curves (10-ep moving avg)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_queue_curves.png", dpi=150)
    print(f"Saved: training_queue_curves.png")
    plt.close()
    
    # 4. Average speed curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for agent in agents:
        agent_data = df[df['agent'] == agent].sort_values('episode')
        smoothed = agent_data['avg_speed'].rolling(window=10, center=True).mean()
        ax.plot(agent_data['episode'], smoothed, label=f"{agent.upper()}", linewidth=2, marker='o', markersize=2, alpha=0.7)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Avg Speed (m/s)", fontsize=12)
    ax.set_title("Hybrid Training Average Speed Curves (10-ep moving avg)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_speed_curves.png", dpi=150)
    print(f"Saved: training_speed_curves.png")
    plt.close()
    
    # 5. Comparison table: final episode metrics
    print("\n" + "=" * 80)
    print("FINAL TRAINING METRICS (Last Episode)")
    print("=" * 80)
    
    final_metrics = []
    for agent in agents:
        agent_data = df[df['agent'] == agent].sort_values('episode')
        last = agent_data.iloc[-1]
        final_metrics.append({
            'Agent': agent.upper(),
            'Reward': last['reward'],
            'Wait (s)': last['wait'],
            'Queue': last['queue'],
            'Speed (m/s)': last['avg_speed']
        })
    
    final_df = pd.DataFrame(final_metrics)
    print(final_df.to_string(index=False))
    
    # 6. Best metrics achieved
    print("\n" + "=" * 80)
    print("BEST METRICS ACHIEVED")
    print("=" * 80)
    
    best_metrics = []
    for agent in agents:
        agent_data = df[df['agent'] == agent]
        
        best_reward_idx = agent_data['reward'].idxmax()
        best_wait_idx = agent_data['wait'].idxmin()
        best_queue_idx = agent_data['queue'].idxmin()
        best_speed_idx = agent_data['avg_speed'].idxmax()
        
        best_metrics.append({
            'Agent': agent.upper(),
            'Best Reward': agent_data.loc[best_reward_idx, 'reward'],
            'Best Wait (s)': agent_data.loc[best_wait_idx, 'wait'],
            'Best Queue': agent_data.loc[best_queue_idx, 'queue'],
            'Best Speed (m/s)': agent_data.loc[best_speed_idx, 'avg_speed']
        })
    
    best_df = pd.DataFrame(best_metrics)
    print(best_df.to_string(index=False))
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    plot_hybrid_training_curves()
