import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hybrid_comparison(evaluation_csv, output_dir):
    if not os.path.exists(evaluation_csv):
        print(f"Error: {evaluation_csv} not found.")
        return
        
    df = pd.read_csv(evaluation_csv)
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Wait time comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Agent', y='Waiting Time', palette='Blues_d')
    plt.title('Average Waiting Time by Hybrid RL Agent', fontsize=15)
    plt.ylabel('Waiting Time (s)', fontsize=12)
    plt.xlabel('Agent Config', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hybrid_wait_time_comparison.png'))
    plt.close()
    
    # Queue length comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Agent', y='Queue Length', palette='Reds_d')
    plt.title('Average Queue Length by Hybrid RL Agent', fontsize=15)
    plt.ylabel('Queue Length (Vehicles)', fontsize=12)
    plt.xlabel('Agent Config', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hybrid_queue_comparison.png'))
    plt.close()
    
    print(f"Hybrid comparison charts generated at {output_dir}")

if __name__ == "__main__":
    plot_hybrid_comparison(
        os.path.join("..", "visualizations", "comparison_charts", "hybrid_evaluation_results.csv"),
        "comparison_charts"
    )
