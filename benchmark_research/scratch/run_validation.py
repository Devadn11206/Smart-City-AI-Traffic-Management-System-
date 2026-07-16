import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

def run_experiments():
    base_dir = r"c:\Users\ddnan\intenrship_IIIT\benchmark_research"
    os.chdir(base_dir)
    
    import time
    experiments = [
        ('dqn', 'A'),
        ('dqn', 'B')
    ]
    
    for algo, rev in experiments:
        print(f"Running {algo} with Reward Version {rev}...")
        cmd = f"python scripts\\train.py --algo {algo} --episodes 15 --reward_version {rev}"
        subprocess.run(cmd, shell=True)
        print(f"Completed {algo} Version {rev}. Waiting to clear TraCI sockets...")
        time.sleep(5)
        
def plot_results():
    base_dir = r"c:\Users\ddnan\intenrship_IIIT\benchmark_research"
    results_dir = os.path.join(base_dir, 'results', '2x2')
    
    experiments = [
        ('ppo', 'A'),
        ('ppo', 'B'),
        ('dqn', 'A'),
        ('dqn', 'B')
    ]
    
    data = {}
    for algo, rev in experiments:
        folder = f"{algo}_rev{rev}"
        csv_path = os.path.join(results_dir, folder, '2x2_episode_metrics.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'eval_mode' in df.columns:
                df = df[df['eval_mode'] == False]
            data[f"{algo.upper()}_Version{rev}"] = df
            
    if not data:
        print("No data found!")
        return
        
    os.makedirs(os.path.join(results_dir, 'validation_plots'), exist_ok=True)
    
    metrics = {
        'reward': 'Reward Trend',
        'avg_queue_length': 'Average Queue Length',
        'throughput': 'Throughput',
        'avg_waiting_time': 'Average Waiting Time',
        'avg_speed': 'Average Speed'
    }
    
    for metric, title in metrics.items():
        plt.figure(figsize=(10, 6))
        for exp_name, df in data.items():
            if metric in df.columns:
                plt.plot(df['episode'], df[metric], label=exp_name, marker='o', markersize=4)
                
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'validation_plots', f'{metric}_comparison.png'))
        plt.close()
        
    print("Plots generated in results/2x2/validation_plots")

if __name__ == "__main__":
    run_experiments()
    plot_results()
