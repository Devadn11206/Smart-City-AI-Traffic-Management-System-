import os
import pandas as pd
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')
    benchmark_dir = os.path.join(results_dir, 'benchmark_comparison')
    os.makedirs(benchmark_dir, exist_ok=True)
    
    algos = ['idqn', 'dqn', 'ddqn', 'ppo', 'presslight', 'colight']
    metrics = ['reward', 'avg_waiting_time', 'avg_queue_length', 'throughput', 'avg_travel_time', 'fuel_consumption', 'avg_speed']
    
    all_data = {}
    for algo in algos:
        file_path = os.path.join(results_dir, algo, '6x6_episode_metrics.csv')
        if os.path.exists(file_path):
            all_data[algo] = pd.read_csv(file_path)
            
    if not all_data:
        print("No data found for analysis.")
        return
        
    # Generate 6x6_benchmark_results.csv
    summary_data = []
    for algo, df in all_data.items():
        tail = df.tail(50) # Average over last 50 episodes
        summary_data.append({
            'Algorithm': algo.upper(),
            'Total Reward': tail['reward'].mean(),
            'Average Waiting Time': tail['avg_waiting_time'].mean(),
            'Average Queue Length': tail['avg_queue_length'].mean(),
            'Throughput': tail['throughput'].mean(),
            'Average Travel Time': tail['avg_travel_time'].mean(),
            'Fuel Consumption': tail['fuel_consumption'].mean(),
            'Average Speed': tail['avg_speed'].mean()
        })
        
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(benchmark_dir, '6x6_benchmark_results.csv'), index=False)
    
    # Generate 6x6_convergence_report.csv
    conv_data = []
    for algo, df in all_data.items():
        smoothed_reward = df['reward'].rolling(window=10, min_periods=1).mean()
        best_ep = smoothed_reward.idxmax() + 1
        tail = df.tail(50)
        conv_data.append({
            'Algorithm': algo.upper(),
            'Best Episode (Smoothed)': best_ep,
            'Reward Var (50)': tail['reward'].var(),
            'Wait Var (50)': tail['avg_waiting_time'].var(),
            'Queue Var (50)': tail['avg_queue_length'].var()
        })
    conv_df = pd.DataFrame(conv_data)
    conv_df.to_csv(os.path.join(benchmark_dir, '6x6_convergence_report.csv'), index=False)
    
    print("6x6 Basic Analysis generated.")

if __name__ == '__main__':
    main()
