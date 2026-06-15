#!/usr/bin/env python3
"""
Monitor and display real-time training progress from hybrid training logs.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def monitor_training():
    repo_root = Path(__file__).parent.parent
    log_file = repo_root / "results/hybrid_logs/hybrid_training_log.csv"
    
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return
    
    # Read the log
    try:
        df = pd.read_csv(log_file)
    except Exception as e:
        print(f"Error reading log: {e}")
        return
    
    # Group by agent
    agents = df['agent'].unique() if 'agent' in df.columns else []
    
    print("\n" + "=" * 80)
    print("HYBRID TRAINING PROGRESS MONITOR")
    print("=" * 80)
    
    for agent in sorted(agents):
        agent_data = df[df['agent'] == agent]
        if len(agent_data) == 0:
            continue
        
        print(f"\n[{agent.upper()}] Trained {len(agent_data)} episodes")
        
        # Latest metrics
        latest = agent_data.iloc[-1]
        print(f"  Episode {int(latest['episode'])}: Reward={latest['reward']:.2f}")
        print(f"    Wait Time: {latest['wait']:.2f}s | Queue: {latest['queue']:.2f} | Avg Speed: {latest['avg_speed']:.2f} m/s")
        print(f"    Switches: {int(latest['switches'])}")
        
        # Trend (last 10 episodes)
        if len(agent_data) >= 10:
            last_10 = agent_data.tail(10)
            avg_reward = last_10['reward'].mean()
            avg_wait = last_10['wait'].mean()
            print(f"  Last 10-ep avg: Reward={avg_reward:.2f} | Wait={avg_wait:.2f}s")
        
        # Best metrics
        best_reward_idx = agent_data['reward'].idxmax()
        best_reward = agent_data.loc[best_reward_idx]
        print(f"  Best Reward: {best_reward['reward']:.2f} (ep {int(best_reward['episode'])})")
        
        best_wait_idx = agent_data['wait'].idxmin()
        best_wait = agent_data.loc[best_wait_idx]
        print(f"  Best Wait Time: {best_wait['wait']:.2f}s (ep {int(best_wait['episode'])})")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    monitor_training()
