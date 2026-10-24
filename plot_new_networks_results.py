import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def plot_results():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_10x10 = os.path.join(base_dir, 'ddqn_pinn_package', 'pi_ddqn_10x10_metrics.csv')
    metrics_baseline_10x10 = os.path.join(base_dir, 'ddqn_pinn_package', 'ddqn_baseline_10x10_metrics.csv')
    metrics_irregular = os.path.join(base_dir, 'ddqn_pinn_package', 'pi_ddqn_irregular_metrics.csv')
    
    sns.set(style="whitegrid")
    
    data = {}
    if os.path.exists(metrics_10x10):
        data['PI-DDQN 10x10'] = pd.read_csv(metrics_10x10)
    if os.path.exists(metrics_baseline_10x10):
        data['Baseline DDQN 10x10'] = pd.read_csv(metrics_baseline_10x10)
    if os.path.exists(metrics_irregular):
        data['PI-DDQN Irregular'] = pd.read_csv(metrics_irregular)
        
    if not data:
        print("Metrics files not found yet. Run the training scripts first!")
        return
        
    metrics_to_plot = [
        ('reward', 'Episode Reward (Higher is better)'),
        ('avg_waiting_time', 'Average Waiting Time (Lower is better)'),
        ('avg_queue_length', 'Average Queue Length (Lower is better)'),
        ('throughput', 'Throughput (Higher is better)'),
        ('avg_travel_time', 'Average Travel Time (Lower is better)')
    ]
    
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(10, 5 * len(metrics_to_plot)))
    if len(metrics_to_plot) == 1: axes = [axes]
    
    colors = {'PI-DDQN 10x10': 'blue', 'Baseline DDQN 10x10': 'red', 'PI-DDQN Irregular': 'orange'}
    
    for i, (metric, ylabel) in enumerate(metrics_to_plot):
        ax = axes[i]
        
        for name, df in data.items():
            if metric in df.columns:
                # Apply moving average for smoother plots
                smoothed = df[metric].rolling(window=10, min_periods=1).mean()
                ax.plot(df['episode'], smoothed, label=f"{name} (SMA 10)", color=colors[name], linewidth=2)
                ax.plot(df['episode'], df[metric], alpha=0.2, color=colors[name])
                
        ax.set_title(ylabel)
        ax.set_xlabel('Episode')
        ax.set_ylabel(metric)
        ax.legend()
        
    plt.tight_layout()
    output_path = os.path.join(base_dir, 'results', 'new_networks_comparison.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plots saved to {output_path}")

if __name__ == "__main__":
    plot_results()
