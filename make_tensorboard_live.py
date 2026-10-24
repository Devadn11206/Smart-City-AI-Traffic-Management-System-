import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import shutil

base_dir = r"c:\Users\ddnan\intenrship_IIIT"
log_dir = os.path.join(base_dir, "tensorboard_logs")

# Clean existing logs if any to prevent duplicate overlap
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir, exist_ok=True)

# Helper function to read CSVs safely
def safe_read(path):
    if os.path.exists(path):
        return pd.read_csv(path, on_bad_lines='skip')
    return None

def log_experiment_to_tensorboard(experiment_group, run_name, csv_path):
    df = safe_read(csv_path)
    if df is None or df.empty:
        print(f"Skipping {run_name} (No data at {csv_path})")
        return

    # TensorBoard logs are structured by directories. Group/RunName
    run_dir = os.path.join(log_dir, experiment_group, run_name)
    writer = SummaryWriter(log_dir=run_dir)
    
    print(f"Logging {run_name} to TensorBoard under {experiment_group}...")
    
    # Log metrics step by step
    for idx, row in df.iterrows():
        step = int(row['episode']) if 'episode' in row else idx
        
        for col in ['reward', 'avg_waiting_time', 'avg_queue_length', 'throughput', 'avg_travel_time', 'loss']:
            if col in row and not pd.isna(row[col]):
                writer.add_scalar(col, float(row[col]), step)
                
    writer.close()

# 1. Log Baseline Benchmark Experiments (6x6)
algorithms = ["dqn", "ddqn", "ppo", "idqn", "presslight", "colight"]
for alg in algorithms:
    csv_path = os.path.join(base_dir, "benchmark_research", "6x6_benchmark", "results", alg, "6x6_episode_metrics.csv")
    log_experiment_to_tensorboard(
        experiment_group="Baseline_6x6_Benchmarks",
        run_name=alg.upper(),
        csv_path=csv_path
    )

# 2. Log PINN Experiments (6x6 and 10x10)
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
    network = "10x10" if "10x10" in name else "6x6"
    log_experiment_to_tensorboard(
        experiment_group=f"Physics_Informed_RL_{network}",
        run_name=name,
        csv_path=path
    )

print("SUCCESS: All historical data has been written to TensorBoard logs!")
