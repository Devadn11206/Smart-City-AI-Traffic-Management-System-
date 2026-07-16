import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Configurations
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
algos = ['idqn', 'dqn', 'ddqn', 'ppo', 'presslight', 'colight']
metrics = ['reward', 'avg_waiting_time', 'avg_queue_length', 'throughput', 'avg_travel_time', 'fuel_consumption', 'avg_speed']

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results', '4x4')
    audit_dir = os.path.join(results_dir, 'publication_audit')
    os.makedirs(audit_dir, exist_ok=True)

    all_data = {}
    for algo in algos:
        file_path = os.path.join(results_dir, algo, '4x4_episode_metrics.csv')
        if os.path.exists(file_path):
            all_data[algo] = pd.read_csv(file_path)

    if not all_data:
        print("No data found!")
        return

    # 1. Convergence & Best Episode Fix
    audit_data = []
    for algo, df in all_data.items():
        # Smoothed reward (10-episode rolling average) to find true peak performance
        smoothed_reward = df['reward'].rolling(window=10, min_periods=1).mean()
        best_ep = smoothed_reward.idxmax() + 1
        
        tail_df = df.tail(50)
        audit_data.append({
            'Algorithm': algo.upper(),
            'Best Episode (Smoothed)': best_ep,
            'Reward Stability (Variance)': tail_df['reward'].var(),
            'Average Waiting Time': tail_df['avg_waiting_time'].mean(),
            'Average Queue Length': tail_df['avg_queue_length'].mean(),
            'Throughput': tail_df['throughput'].mean(),
            'Average Travel Time': tail_df['avg_travel_time'].mean(),
            'Fuel Consumption': tail_df['fuel_consumption'].mean(),
            'Average Speed': tail_df['avg_speed'].mean()
        })

    audit_df = pd.DataFrame(audit_data)

    # 2. Normalization (Min-Max scaling from 0 to 1, where 1 is best)
    def normalize_positive(col):
        return (col - col.min()) / (col.max() - col.min() + 1e-9)

    def normalize_negative(col):
        return (col.max() - col) / (col.max() - col.min() + 1e-9)

    norm_df = pd.DataFrame({'Algorithm': audit_df['Algorithm']})
    norm_df['Throughput_Norm'] = normalize_positive(audit_df['Throughput'])
    norm_df['Speed_Norm'] = normalize_positive(audit_df['Average Speed'])
    
    norm_df['Queue_Norm'] = normalize_negative(audit_df['Average Queue Length'])
    norm_df['Wait_Norm'] = normalize_negative(audit_df['Average Waiting Time'])
    norm_df['Travel_Norm'] = normalize_negative(audit_df['Average Travel Time'])
    norm_df['Fuel_Norm'] = normalize_negative(audit_df['Fuel Consumption'])
    norm_df['Stability_Norm'] = normalize_negative(audit_df['Reward Stability (Variance)'])

    # Weighted Score (Sum of normalized metrics)
    norm_df['Weighted_Score'] = (
        norm_df['Throughput_Norm'] + norm_df['Speed_Norm'] +
        norm_df['Queue_Norm'] + norm_df['Wait_Norm'] + 
        norm_df['Travel_Norm'] + norm_df['Fuel_Norm'] + 
        norm_df['Stability_Norm']
    )

    # Sort and rank
    norm_df = norm_df.sort_values(by='Weighted_Score', ascending=False).reset_index(drop=True)
    norm_df['Rank'] = norm_df.index + 1
    
    audit_df = pd.merge(norm_df[['Algorithm', 'Weighted_Score', 'Rank']], audit_df, on='Algorithm').sort_values('Rank')
    audit_df.to_csv(os.path.join(audit_dir, '4x4_audit_summary.csv'), index=False)
    norm_df.to_csv(os.path.join(audit_dir, '4x4_normalized_metrics.csv'), index=False)

    # 3. Publication Quality Graphs
    colors = sns.color_palette("husl", len(algos))
    for metric in metrics:
        plt.figure(figsize=(8, 5), dpi=300)
        for i, (algo, df) in enumerate(all_data.items()):
            if metric in df.columns:
                smoothed = df[metric].rolling(window=10, min_periods=1).mean()
                plt.plot(df['episode'], smoothed, label=algo.upper(), linewidth=2, color=colors[i])
                plt.plot(df['episode'], df[metric], alpha=0.15, color=colors[i]) # Faint raw data

        title_str = metric.replace("_", " ").title()
        plt.title(f'{title_str} Convergence (10-Ep Rolling Avg)', fontweight='bold')
        plt.xlabel('Episode', fontweight='bold')
        plt.ylabel(title_str, fontweight='bold')
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        plt.savefig(os.path.join(audit_dir, f'pub_{metric}_convergence.png'))
        plt.close()

    # 4. Radar Chart
    categories = ['Throughput', 'Speed', 'Queue (Inv)', 'Wait Time (Inv)', 'Travel Time (Inv)', 'Fuel (Inv)', 'Stability (Inv)']
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(10, 10), dpi=300)
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, size=11, fontweight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=9)
    plt.ylim(0, 1)

    for i, row in norm_df.iterrows():
        values = [
            row['Throughput_Norm'], row['Speed_Norm'], 
            row['Queue_Norm'], row['Wait_Norm'], 
            row['Travel_Norm'], row['Fuel_Norm'], 
            row['Stability_Norm']
        ]
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"{row['Algorithm']} (Score: {row['Weighted_Score']:.2f})")
        ax.fill(angles, values, alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True)
    plt.title("Algorithm Holistic Performance (Normalized)", size=16, fontweight='bold', y=1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(audit_dir, 'pub_radar_chart.png'))
    plt.close()

    # 5. Final Markdown Summary
    md_content = "# 4x4 Benchmark Final Audited Paper Summary\n\n"
    md_content += "This table presents the normalized weighted scoring out of 7.0, along with the corrected 'Best Episode' computation using a 10-episode rolling average to filter initialization artifacts.\n\n"
    
    md_content += "| Rank | Algorithm | Weighted Score (/7.0) | Best Episode | Reward Variance | Wait (s) | Queue | Throughput | Travel Time (s) | Fuel (ml) | Speed (m/s) |\n"
    md_content += "|---|---|---|---|---|---|---|---|---|---|---|\n"
    for _, row in audit_df.iterrows():
        md_content += f"| {int(row['Rank'])} | **{row['Algorithm']}** | {row['Weighted_Score']:.2f} | {row['Best Episode (Smoothed)']} | {row['Reward Stability (Variance)']:.1f} | {row['Average Waiting Time']:.1f} | {row['Average Queue Length']:.1f} | {row['Throughput']:.1f} | {row['Average Travel Time']:.1f} | {row['Fuel Consumption']:.2e} | {row['Average Speed']:.2f} |\n"

    with open(os.path.join(audit_dir, 'final_paper_summary.md'), 'w') as f:
        f.write(md_content)

    print(f"Audit complete. Outputs saved to {audit_dir}")

if __name__ == '__main__':
    main()
