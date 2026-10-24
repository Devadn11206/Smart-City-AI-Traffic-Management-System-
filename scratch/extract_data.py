import os
import pandas as pd
import json

base_path = r"c:\Users\ddnan\intenrship_IIIT"
results = {}

def get_metrics(filepath):
    if not os.path.exists(filepath):
        return "NOT AVAILABLE"
    try:
        df = pd.read_csv(filepath)
        if len(df) == 0:
            return "NOT AVAILABLE"
        last_10 = df.tail(10)
        
        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        
        # Helper to safely get column average
        def safe_mean(df, possible_names):
            for name in possible_names:
                if name in df.columns:
                    return float(df[name].mean())
            return None
            
        def safe_max(df, possible_names):
            for name in possible_names:
                if name in df.columns:
                    return float(df[name].max())
            return None
            
        def safe_min(df, possible_names):
            for name in possible_names:
                if name in df.columns:
                    return float(df[name].min())
            return None

        # Extract values
        episodes = int(df['episode'].max()) if 'episode' in df.columns else (int(df['Episode'].max()) if 'Episode' in df.columns else len(df))
        reward_cols = ['reward', 'MeanReward']
        
        final_reward = safe_mean(last_10, reward_cols)
        best_reward = safe_max(df, reward_cols)
        
        waiting_time = safe_mean(last_10, ['avg_waiting_time'])
        queue_length = safe_mean(last_10, ['avg_queue_length'])
        throughput = safe_mean(last_10, ['throughput'])
        travel_time = safe_mean(last_10, ['avg_travel_time'])
        fuel = safe_mean(last_10, ['fuel_consumption'])
        final_loss = safe_mean(last_10, ['loss'])
        
        return {
            "Episodes": episodes,
            "Final Reward": round(final_reward, 2) if final_reward is not None else "NOT AVAILABLE",
            "Best Reward": round(best_reward, 2) if best_reward is not None else "NOT AVAILABLE",
            "Waiting Time": round(waiting_time, 2) if waiting_time is not None else "NOT AVAILABLE",
            "Queue Length": round(queue_length, 2) if queue_length is not None else "NOT AVAILABLE",
            "Throughput": round(throughput, 2) if throughput is not None else "NOT AVAILABLE",
            "Travel Time": round(travel_time, 2) if travel_time is not None else "NOT AVAILABLE",
            "Fuel": round(fuel, 2) if fuel is not None else "NOT AVAILABLE",
            "Final Loss": round(final_loss, 4) if final_loss is not None else "NOT AVAILABLE"
        }
    except Exception as e:
        return f"ERROR: {str(e)}"

# 1. Benchmark RL
networks = ['2x2', '4x4', '6x6']
algs = ['dqn', 'ddqn', 'ppo', 'idqn', 'presslight', 'colight']
for net in networks:
    for alg in algs:
        p = os.path.join(base_path, "benchmark_research", f"{net}_benchmark", "results", alg, f"{net}_episode_metrics.csv")
        results[f"Baseline_{alg}_{net}"] = get_metrics(p)

# 2. PINN + DQN
results["Baseline_DQN_6x6_DqnPackage"] = get_metrics(os.path.join(base_path, "dqn_pinn_package", "training", "dqn_6x6_metrics.csv"))
results["PINN_DQN_6x6"] = get_metrics(os.path.join(base_path, "dqn_pinn_package", "training", "pi_mbpo_6x6_metrics.csv"))

# 3. PINN + DDQN 
results["Baseline_DDQN_6x6_DdqnPackage"] = get_metrics(os.path.join(base_path, "ddqn_pinn_package", "ddqn_baseline_metrics.csv"))
results["PINN_DDQN_6x6"] = get_metrics(os.path.join(base_path, "ddqn_pinn_package", "pi_ddqn_metrics.csv"))

results["Baseline_DDQN_10x10"] = get_metrics(os.path.join(base_path, "ddqn_with_pinn_for_10x10", "ddqn_baseline_10x10_metrics.csv"))
results["PINN_DDQN_10x10"] = get_metrics(os.path.join(base_path, "ddqn_with_pinn_for_10x10", "pi_ddqn_10x10_metrics.csv"))

# 4. PINN + PPO
results["Baseline_PPO_6x6_PpoPackage"] = get_metrics(os.path.join(base_path, "ppo_pinn_package", "training", "ppo_6x6_metrics.csv"))
results["PINN_PPO_6x6"] = get_metrics(os.path.join(base_path, "ppo_pinn_package", "training", "ppo_pinn_6x6_metrics.csv"))

# Also check root for DQN 6x6 just in case
results["Root_DQN_6x6"] = get_metrics(os.path.join(base_path, "dqn_6x6_metrics.csv"))

with open(os.path.join(base_path, "scratch", "compiled_results.json"), "w") as f:
    json.dump(results, f, indent=4)
print("Data extraction complete.")
