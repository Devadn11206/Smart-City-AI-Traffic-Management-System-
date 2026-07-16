import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

algos = ['idqn', 'dqn', 'ddqn', 'ppo', 'presslight', 'colight']
metrics = ['reward', 'avg_waiting_time', 'avg_queue_length', 'throughput', 'avg_travel_time', 'fuel_consumption', 'avg_speed', 'loss']

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results', '2x2')
    bench_dir = os.path.join(results_dir, 'benchmark_comparison')
    os.makedirs(bench_dir, exist_ok=True)
    
    all_data = {}
    
    # Load data
    for algo in algos:
        file_path = os.path.join(results_dir, algo, '2x2_episode_metrics.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_data[algo] = df
            
    if not all_data:
        print("No training data found. Please run the training pipeline first.")
        return
        
    # Generate metric history CSVs & Plots
    for metric in metrics:
        history_df = pd.DataFrame()
        plt.figure(figsize=(10, 6))
        
        for algo, df in all_data.items():
            if metric in df.columns:
                history_df[algo] = df[metric]
                plt.plot(df['episode'], df[metric], label=algo.upper())
                
        history_df.to_csv(os.path.join(bench_dir, f'2x2_{metric}_history.csv'), index=False)
        
        plt.title(f'{metric.replace("_", " ").title()} vs Episode')
        plt.xlabel('Episode')
        plt.ylabel(metric.replace("_", " ").title())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(bench_dir, f'2x2_{metric}_comparison.png'))
        plt.close()
        
    # Generate convergence report
    conv_data = []
    for algo, df in all_data.items():
        best_ep = df['reward'].idxmax() + 1
        worst_ep = df['reward'].idxmin() + 1
        reward_var = df['reward'].tail(50).var()
        wait_var = df['avg_waiting_time'].tail(50).var()
        queue_var = df['avg_queue_length'].tail(50).var()
        
        conv_data.append({
            'Algorithm': algo.upper(),
            'Best Episode': best_ep,
            'Worst Episode': worst_ep,
            'Reward Variance (last 50)': reward_var,
            'Wait Variance (last 50)': wait_var,
            'Queue Variance (last 50)': queue_var
        })
        
    pd.DataFrame(conv_data).to_csv(os.path.join(bench_dir, '2x2_convergence_report.csv'), index=False)
    
    # Generate Final Benchmark Results (average of last 50 episodes)
    bench_results = []
    for algo, df in all_data.items():
        tail_df = df.tail(50)
        bench_results.append({
            'Algorithm': algo.upper(),
            'Total Reward': tail_df['reward'].mean(),
            'Average Waiting Time': tail_df['avg_waiting_time'].mean(),
            'Average Queue Length': tail_df['avg_queue_length'].mean(),
            'Throughput': tail_df['throughput'].mean(),
            'Average Travel Time': tail_df['avg_travel_time'].mean(),
            'Fuel Consumption': tail_df['fuel_consumption'].mean(),
            'Average Speed': tail_df['avg_speed'].mean()
        })
        
    bench_df = pd.DataFrame(bench_results)
    bench_df.to_csv(os.path.join(bench_dir, '2x2_benchmark_results.csv'), index=False)
    
    print(f"Analysis complete. Reports saved to {bench_dir}")

if __name__ == '__main__':
    main()
