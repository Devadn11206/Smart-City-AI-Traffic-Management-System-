import optuna
import os
import pandas as pd

base_dir = r"c:\Users\ddnan\intenrship_IIIT"
db_url = "sqlite:///optuna.db"

# Optuna requires a study to add trials to. 
# We'll minimize waiting time as the objective.
study = optuna.create_study(
    study_name="Traffic_Optimization_Benchmarks",
    storage=db_url,
    direction="minimize",
    load_if_exists=True
)

def safe_read(path):
    if os.path.exists(path):
        return pd.read_csv(path, on_bad_lines='skip')
    return None

def add_optuna_trial(algorithm, network, pinn_enabled, csv_path):
    df = safe_read(csv_path)
    if df is None or df.empty:
        print(f"Skipping {algorithm} (No data at {csv_path})")
        return
    
    # Use the final waiting time (mean of last 10 episodes) as the objective value
    if len(df) > 10:
        last_10 = df.tail(10)
    else:
        last_10 = df
        
    if 'avg_waiting_time' not in last_10.columns:
        print(f"Skipping {algorithm} (Missing avg_waiting_time)")
        return
        
    final_wait = float(last_10['avg_waiting_time'].mean())
    
    trial = optuna.trial.create_trial(
        params={
            "Algorithm": algorithm,
            "Network": network,
            "PINN_Enabled": "Yes" if pinn_enabled else "No"
        },
        distributions={
            "Algorithm": optuna.distributions.CategoricalDistribution(["DQN", "DDQN", "PPO", "IDQN", "PRESSLIGHT", "COLIGHT", "MBPO"]),
            "Network": optuna.distributions.CategoricalDistribution(["6x6", "10x10"]),
            "PINN_Enabled": optuna.distributions.CategoricalDistribution(["Yes", "No"])
        },
        value=final_wait
    )
    study.add_trial(trial)
    print(f"Added trial for {algorithm} ({network}) PINN={pinn_enabled} -> Wait Time: {final_wait:.2f}")

# 1. Baseline Benchmark Experiments (6x6)
algorithms = ["dqn", "ddqn", "ppo", "idqn", "presslight", "colight"]
for alg in algorithms:
    csv_path = os.path.join(base_dir, "benchmark_research", "6x6_benchmark", "results", alg, "6x6_episode_metrics.csv")
    add_optuna_trial(alg.upper(), "6x6", False, csv_path)

# 2. PINN Experiments
pinn_files = {
    "BASE-DQN-6x6": (os.path.join(base_dir, "dqn_pinn_package", "training", "dqn_6x6_metrics.csv"), "DQN", "6x6", False),
    "PINN-MBPO-6x6": (os.path.join(base_dir, "dqn_pinn_package", "training", "pi_mbpo_6x6_metrics.csv"), "MBPO", "6x6", True),
    "BASE-DDQN-6x6": (os.path.join(base_dir, "ddqn_pinn_package", "ddqn_baseline_metrics.csv"), "DDQN", "6x6", False),
    "PINN-DDQN-6x6": (os.path.join(base_dir, "ddqn_pinn_package", "pi_ddqn_metrics.csv"), "DDQN", "6x6", True),
    "BASE-PPO-6x6": (os.path.join(base_dir, "ppo_pinn_package", "training", "ppo_6x6_metrics.csv"), "PPO", "6x6", False),
    "PINN-PPO-6x6": (os.path.join(base_dir, "ppo_pinn_package", "training", "ppo_pinn_6x6_metrics.csv"), "PPO", "6x6", True),
    "BASE-DDQN-10x10": (os.path.join(base_dir, "ddqn_with_pinn_for_10x10", "ddqn_baseline_10x10_metrics.csv"), "DDQN", "10x10", False),
    "PINN-DDQN-10x10": (os.path.join(base_dir, "ddqn_with_pinn_for_10x10", "pi_ddqn_10x10_metrics.csv"), "DDQN", "10x10", True)
}

for name, (path, alg, net, is_pinn) in pinn_files.items():
    add_optuna_trial(alg, net, is_pinn, path)

print("SUCCESS: Data ported to Optuna study successfully!")
