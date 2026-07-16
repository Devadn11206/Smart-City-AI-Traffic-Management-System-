import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')
    benchmark_dir = os.path.join(results_dir, 'benchmark_comparison')
    os.makedirs(benchmark_dir, exist_ok=True)
    
    algos = ['idqn', 'dqn', 'ddqn', 'ppo', 'presslight', 'colight']
    colors = {'idqn': 'tab:purple', 'dqn': 'tab:brown', 'ddqn': 'tab:gray', 'ppo': 'tab:blue', 'presslight': 'tab:green', 'colight': 'tab:red'}
    
    metrics = {
        'reward': 'Total Reward',
        'avg_waiting_time': 'Average Waiting Time (s)',
        'avg_queue_length': 'Average Queue Length',
        'throughput': 'Throughput',
        'avg_speed': 'Average Speed (m/s)',
        'avg_travel_time': 'Average Travel Time (s)',
        'fuel_consumption': 'Fuel Consumption (ml)',
        'loss': 'Loss'
    }
    
    all_data = {}
    for algo in algos:
        file_path = os.path.join(results_dir, algo, '6x6_episode_metrics.csv')
        if os.path.exists(file_path):
            all_data[algo] = pd.read_csv(file_path)
            
    if not all_data:
        print("No data found to plot.")
        return
        
    for metric_col, metric_name in metrics.items():
        plt.figure(figsize=(10, 6))
        for algo, df in all_data.items():
            if metric_col in df.columns:
                # Plot the raw values lightly
                plt.plot(df['episode'], df[metric_col], alpha=0.3, color=colors.get(algo, 'black'))
                # Plot the smoothed values (moving average) clearly
                smoothed = df[metric_col].rolling(window=10, min_periods=1).mean()
                plt.plot(df['episode'], smoothed, label=algo.upper(), linewidth=2, color=colors.get(algo, 'black'))
                
        plt.title(f'6x6 Benchmark: {metric_name} over Episodes')
        plt.xlabel('Episode')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        save_path = os.path.join(benchmark_dir, f'6x6_{metric_col}_plot.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {save_path}")

if __name__ == '__main__':
    main()
