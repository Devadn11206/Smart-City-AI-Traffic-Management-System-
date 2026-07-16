import os
import time
import json
import pandas as pd
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results', '4x4')
    os.makedirs(results_dir, exist_ok=True)
    
    best_file = os.path.join(results_dir, 'best_metrics.json')
    snapshots_file = os.path.join(results_dir, 'checkpoint_snapshots.json')
    
    algos = ['idqn', 'dqn', 'ddqn', 'ppo', 'presslight', 'colight']
    checkpoints = [1, 10, 25, 50, 100, 150, 200]
    
    # Initialize files if they don't exist
    if not os.path.exists(best_file):
        with open(best_file, 'w') as f:
            json.dump({}, f, indent=4)
            
    if not os.path.exists(snapshots_file):
        with open(snapshots_file, 'w') as f:
            json.dump({}, f, indent=4)
            
    print("Starting monitoring script...")
    
    # Loop continuously
    while True:
        try:
            # Load current state
            with open(best_file, 'r') as f:
                try:
                    best_metrics = json.load(f)
                except json.JSONDecodeError:
                    best_metrics = {}
                    
            with open(snapshots_file, 'r') as f:
                try:
                    snapshots = json.load(f)
                except json.JSONDecodeError:
                    snapshots = {}
                    
            updated_best = False
            updated_snapshots = False
            
            all_completed = True
            
            for algo in algos:
                csv_path = os.path.join(results_dir, algo, '4x4_episode_metrics.csv')
                if not os.path.exists(csv_path):
                    all_completed = False
                    continue
                    
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    all_completed = False
                    continue
                    
                if len(df) == 0:
                    all_completed = False
                    continue
                    
                if len(df) < 200:
                    all_completed = False
                
                # Check best metrics
                algo_best = best_metrics.get(algo, {})
                
                max_reward = float(df['reward'].max())
                min_queue = float(df['avg_queue_length'].min())
                max_throughput = float(df['throughput'].max())
                max_speed = float(df['avg_speed'].max())
                
                # Travel time might be 0 in errors or initial, handle carefully
                valid_travel_time = df['avg_travel_time'][df['avg_travel_time'] > 0]
                min_travel = float(valid_travel_time.min()) if len(valid_travel_time) > 0 else 0.0
                
                new_best = {
                    'Best Episode Reward': max_reward,
                    'Best Average Queue Length': min_queue,
                    'Best Throughput': max_throughput,
                    'Best Average Speed': max_speed,
                    'Best Average Travel Time': min_travel
                }
                
                if algo_best != new_best:
                    best_metrics[algo] = new_best
                    updated_best = True
                    
                # Check snapshots
                algo_snapshots = snapshots.get(algo, {})
                for cp in checkpoints:
                    if str(cp) in algo_snapshots:
                        continue # Already recorded
                        
                    # Check if this episode is in the dataframe
                    row = df[df['episode'] == cp]
                    if not row.empty:
                        row_data = row.iloc[0]
                        snapshot_data = {
                            'reward': float(row_data['reward']),
                            'avg_waiting_time': float(row_data['avg_waiting_time']),
                            'avg_queue_length': float(row_data['avg_queue_length']),
                            'throughput': float(row_data['throughput']),
                            'avg_travel_time': float(row_data['avg_travel_time']),
                            'avg_speed': float(row_data['avg_speed']),
                            'fuel_consumption': float(row_data['fuel_consumption'])
                        }
                        algo_snapshots[str(cp)] = snapshot_data
                        snapshots[algo] = algo_snapshots
                        updated_snapshots = True
                        print(f"Recorded snapshot for {algo.upper()} at Episode {cp}")
                        
            # Save if updated
            if updated_best:
                with open(best_file, 'w') as f:
                    json.dump(best_metrics, f, indent=4)
                    
            if updated_snapshots:
                with open(snapshots_file, 'w') as f:
                    json.dump(snapshots, f, indent=4)
                    
            if all_completed:
                print("All algorithms have completed 200 episodes. Exiting monitoring script.")
                break
                
        except Exception as e:
            print(f"Error in monitor loop: {e}")
            
        time.sleep(60) # Wait 60 seconds

if __name__ == '__main__':
    main()
