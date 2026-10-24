import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set up directories
base_dir = r"c:\Users\ddnan\intenrship_IIIT"
manuscript_dir = os.path.join(base_dir, "manuscript")
figures_dir = os.path.join(manuscript_dir, "figures")

for d in ["benchmark", "pinn", "scalability", "mlops", "architecture"]:
    os.makedirs(os.path.join(figures_dir, d), exist_ok=True)

# Helper function to read CSVs safely
def safe_read(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

print("=== STARTING STAGE A: MULTI-SCALE BENCHMARK ANALYSIS ===")

# Dictionaries to store aggregated data
benchmark_data = {"2x2": {}, "4x4": {}, "6x6": {}}
algorithms = ["dqn", "ddqn", "ppo", "idqn", "presslight", "colight"]

# Load 2x2
dir_2x2 = os.path.join(base_dir, "benchmark_research", "2x2_benchmark", "results")
for alg in algorithms:
    path = os.path.join(dir_2x2, alg, "2x2_episode_metrics.csv")
    df = safe_read(path)
    if df is not None: benchmark_data["2x2"][alg] = df

# Load 4x4
dir_4x4 = os.path.join(base_dir, "benchmark_research", "4x4_benchmark", "results", "4x4")
for alg in algorithms:
    path = os.path.join(dir_4x4, alg, "4x4_episode_metrics.csv")
    df = safe_read(path)
    if df is not None: benchmark_data["4x4"][alg] = df

# Load 6x6
dir_6x6 = os.path.join(base_dir, "benchmark_research", "6x6_benchmark", "results")
for alg in algorithms:
    path = os.path.join(dir_6x6, alg, "6x6_episode_metrics.csv")
    df = safe_read(path)
    if df is not None: benchmark_data["6x6"][alg] = df

# Aggregate final performance (last 10% of episodes for converged performance)
summary_stats = []
for net in ["2x2", "4x4", "6x6"]:
    for alg, df in benchmark_data[net].items():
        if len(df) > 10:
            last_10_pct = df.tail(int(len(df)*0.1))
            summary_stats.append({
                "Network": net,
                "Algorithm": alg.upper(),
                "WaitTime_Mean": last_10_pct['avg_waiting_time'].mean() if 'avg_waiting_time' in df.columns else np.nan,
                "WaitTime_SD": last_10_pct['avg_waiting_time'].std() if 'avg_waiting_time' in df.columns else np.nan,
                "Queue_Mean": last_10_pct['avg_queue_length'].mean() if 'avg_queue_length' in df.columns else np.nan,
                "Queue_SD": last_10_pct['avg_queue_length'].std() if 'avg_queue_length' in df.columns else np.nan,
                "Throughput_Mean": last_10_pct['throughput'].mean() if 'throughput' in df.columns else np.nan,
                "Throughput_SD": last_10_pct['throughput'].std() if 'throughput' in df.columns else np.nan,
                "Reward_Mean": last_10_pct['reward'].mean() if 'reward' in df.columns else np.nan
            })

df_summary = pd.DataFrame(summary_stats)
df_summary.to_csv(os.path.join(manuscript_dir, "Table_4_Multi_Scale_Benchmark.csv"), index=False)

# --- Plotting Benchmark Figures ---
sns.set_theme(style="whitegrid")

# Fig 3: Waiting Time Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=df_summary, x="Network", y="WaitTime_Mean", hue="Algorithm")
plt.ylabel("Average Waiting Time (s)")
plt.title("Waiting Time Comparison Across Network Scales")
plt.savefig(os.path.join(figures_dir, "benchmark", "fig3_waiting_time.png"), dpi=300, bbox_inches='tight')
plt.close()

# Fig 4: Queue Length Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=df_summary, x="Network", y="Queue_Mean", hue="Algorithm")
plt.ylabel("Average Queue Length")
plt.title("Queue Length Comparison Across Network Scales")
plt.savefig(os.path.join(figures_dir, "benchmark", "fig4_queue_length.png"), dpi=300, bbox_inches='tight')
plt.close()

# Fig 6: Throughput Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=df_summary, x="Network", y="Throughput_Mean", hue="Algorithm")
plt.ylabel("Average Throughput")
plt.title("Throughput Comparison Across Network Scales")
plt.savefig(os.path.join(figures_dir, "benchmark", "fig6_throughput.png"), dpi=300, bbox_inches='tight')
plt.close()

print("=== STARTING STAGE B: PINN-INTEGRATED RL ANALYSIS ===")

pinn_data = {}
pinn_files = {
    "BASE-DQN-6x6": os.path.join(base_dir, "dqn_pinn_package", "training", "dqn_6x6_metrics.csv"),
    "PINN-MBPO-6x6": os.path.join(base_dir, "dqn_pinn_package", "training", "pi_mbpo_6x6_metrics.csv"),
    "BASE-DDQN-6x6": os.path.join(base_dir, "ddqn_pinn_package", "ddqn_baseline_metrics.csv"),
    "PINN-DDQN-6x6": os.path.join(base_dir, "ddqn_pinn_package", "pi_ddqn_metrics.csv"),
    "BASE-PPO-6x6": os.path.join(base_dir, "ppo_pinn_package", "training", "ppo_6x6_metrics.csv"),
    "PINN-PPO-6x6": os.path.join(base_dir, "ppo_pinn_package", "training", "ppo_pinn_6x6_metrics.csv"),
    "BASE-DDQN-10x10": os.path.join(base_dir, "ddqn_with_pinn_for_10x10", "ddqn_baseline_10x10_metrics.csv"),
    "PINN-DDQN-10x10": os.path.join(base_dir, "ddqn_with_pinn_for_10x10", "pi_ddqn_10x10_metrics.csv")
}

for name, path in pinn_files.items():
    df = safe_read(path)
    if df is not None:
        pinn_data[name] = df

pinn_summary = []
for name, df in pinn_data.items():
    if len(df) > 5:
        last_10_pct = df.tail(int(len(df)*0.1))
        algo_base = "DQN" if "DQN" in name else "DDQN" if "DDQN" in name else "PPO"
        is_pinn = "Yes" if "PINN" in name else "No"
        network = "10x10" if "10x10" in name else "6x6"
        pinn_summary.append({
            "Experiment": name,
            "Network": network,
            "Algorithm": algo_base,
            "PINN_Enabled": is_pinn,
            "WaitTime_Mean": last_10_pct['avg_waiting_time'].mean() if 'avg_waiting_time' in df.columns else np.nan,
            "Queue_Mean": last_10_pct['avg_queue_length'].mean() if 'avg_queue_length' in df.columns else np.nan,
            "Throughput_Mean": last_10_pct['throughput'].mean() if 'throughput' in df.columns else np.nan,
        })

df_pinn_summary = pd.DataFrame(pinn_summary)
df_pinn_summary.to_csv(os.path.join(manuscript_dir, "Table_6_RL_vs_PINN.csv"), index=False)

# Plotting PINN comparisons (Fig 14, 15, 16, 17)
# 6x6 Comparison
df_6x6_pinn = df_pinn_summary[df_pinn_summary['Network'] == '6x6']
if not df_6x6_pinn.empty:
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_6x6_pinn, x="Algorithm", y="WaitTime_Mean", hue="PINN_Enabled")
    plt.ylabel("Average Waiting Time (s)")
    plt.title("RL vs PINN-Integrated RL (6x6 Network)")
    plt.savefig(os.path.join(figures_dir, "pinn", "fig14_pinn_waiting_time.png"), dpi=300, bbox_inches='tight')
    plt.close()

# Scalability (6x6 vs 10x10 for DDQN)
df_ddqn_scale = df_pinn_summary[(df_pinn_summary['Algorithm'] == 'DDQN') & (df_pinn_summary['PINN_Enabled'] == 'Yes')]
if not df_ddqn_scale.empty:
    plt.figure(figsize=(6, 5))
    sns.barplot(data=df_ddqn_scale, x="Network", y="WaitTime_Mean")
    plt.ylabel("Average Waiting Time (s)")
    plt.title("DDQN+PINN Scalability (6x6 vs 10x10)")
    plt.savefig(os.path.join(figures_dir, "scalability", "fig18_scalability.png"), dpi=300, bbox_inches='tight')
    plt.close()

print("=== SCRIPT FINISHED ===")
