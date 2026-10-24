import os
import pandas as pd
import mlflow
import time

base_dir = r"c:\Users\ddnan\intenrship_IIIT"
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Helper function to read CSVs safely
def safe_read(path):
    if os.path.exists(path):
        return pd.read_csv(path, on_bad_lines='skip')
    return None

def log_experiment_to_mlflow(experiment_name, run_name, csv_path, params=None):
    df = safe_read(csv_path)
    if df is None or df.empty:
        print(f"Skipping {run_name} (No data at {csv_path})")
        return

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        print(f"Logging {run_name} to experiment {experiment_name}...")
        
        # Log parameters
        if params:
            mlflow.log_params(params)
        
        # Log metrics step by step
        for idx, row in df.iterrows():
            step = int(row['episode']) if 'episode' in row else idx
            
            metrics_to_log = {}
            for col in ['reward', 'avg_waiting_time', 'avg_queue_length', 'throughput', 'avg_travel_time', 'loss']:
                if col in row and not pd.isna(row[col]):
                    metrics_to_log[col] = float(row[col])
                    
            if metrics_to_log:
                mlflow.log_metrics(metrics_to_log, step=step)

# 1. Log Baseline Benchmark Experiments (6x6)
algorithms = ["dqn", "ddqn", "ppo", "idqn", "presslight", "colight"]
for alg in algorithms:
    csv_path = os.path.join(base_dir, "benchmark_research", "6x6_benchmark", "results", alg, "6x6_episode_metrics.csv")
    log_experiment_to_mlflow(
        experiment_name="Baseline_6x6_Benchmarks",
        run_name=f"{alg.upper()}_6x6",
        csv_path=csv_path,
        params={"network": "6x6", "algorithm": alg.upper(), "type": "baseline"}
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
    is_pinn = "Yes" if "PINN" in name else "No"
    
    log_experiment_to_mlflow(
        experiment_name="Physics_Informed_RL",
        run_name=name,
        csv_path=path,
        params={"network": network, "pinn_enabled": is_pinn, "type": "pinn_ablation"}
    )

print("SUCCESS: All historical data has been pushed to the MLflow Live Dashboard!")
