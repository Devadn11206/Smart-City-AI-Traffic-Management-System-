import os
import sys
import argparse
import pandas as pd
from statistics import mean, pstdev

# Add parent path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.hybrid_env import HybridSumoEnv
from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent

EVAL_SEEDS = [101, 202, 303]

def evaluate_agent(agent_type, env, num_seconds_eval=1000, seed=101):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(repo_root, "results", "hybrid_results")
    alt_results_dir = os.path.join(repo_root, "..", "results", "hybrid_results")

    model_path_best = os.path.join(results_dir, f"hybrid_{agent_type}_model_best.pth")
    model_path_final = os.path.join(results_dir, f"hybrid_{agent_type}_model_final.pth")
    model_path_generic = os.path.join(results_dir, f"hybrid_{agent_type}_model.pth")
    if os.path.exists(model_path_best):
        model_path = model_path_best
    elif os.path.exists(model_path_final):
        model_path = model_path_final
    elif os.path.exists(model_path_generic):
        model_path = model_path_generic
    else:
        # fallback to parent workspace results path
        fallback_best = os.path.join(alt_results_dir, f"hybrid_{agent_type}_model_best.pth")
        fallback_final = os.path.join(alt_results_dir, f"hybrid_{agent_type}_model_final.pth")
        fallback_generic = os.path.join(alt_results_dir, f"hybrid_{agent_type}_model.pth")
        if os.path.exists(fallback_best):
            model_path = fallback_best
        elif os.path.exists(fallback_final):
            model_path = fallback_final
        elif os.path.exists(fallback_generic):
            model_path = fallback_generic
        else:
            print(f"[{agent_type.upper()}] Model not found! Returning default values.")
            return 0.0, 0.0

    state_dim = env.state_dim
    action_dim = env.action_dim
    
    if agent_type == 'dqn':
        agent = DQNAgent(state_dim, action_dim)
    elif agent_type == 'ddqn':
        agent = DDQNAgent(state_dim, action_dim)
    elif agent_type == 'ppo':
        agent = PPOAgent(state_dim, action_dim)
        
    agent.load(model_path)
    
    state = env.reset(seed=seed)
    done = False
    
    while not done and env.current_step < num_seconds_eval:
        if agent_type == 'ppo':
            action, _ = agent.act(state, eval_mode=True)
        else:
            action = agent.act(state, eval_mode=True)
            
        next_state, reward, done, _ = env.step(action)
        state = next_state
        
    metrics = env.get_episode_metrics()
    return metrics


def aggregate_metrics(metrics_list):
    return {
        'wait_mean': float(mean([m.get('wait', 0.0) for m in metrics_list])) if metrics_list else 0.0,
        'queue_mean': float(mean([m.get('queue', 0.0) for m in metrics_list])) if metrics_list else 0.0,
        'wait_std': float(pstdev([m.get('wait', 0.0) for m in metrics_list])) if len(metrics_list) > 1 else 0.0,
        'queue_std': float(pstdev([m.get('queue', 0.0) for m in metrics_list])) if len(metrics_list) > 1 else 0.0,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()
    
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_file = os.path.join(repo_root, "sumo_network", "config.sumocfg")
    pinn_model_path = os.path.join(repo_root, "pinn_models", "trained_pinn.pth")
    
    if not os.path.exists(pinn_model_path):
        print("Error: PINN model not found. Please train it first.")
        sys.exit(1)
        
    env = HybridSumoEnv(config_file, pinn_model_path, use_gui=False, num_seconds=args.steps)
    
    agents = ['dqn', 'ddqn', 'ppo']
    results = []
    
    print("\nStarting Hybrid Evaluation...\n")
    
    for a in agents:
        print(f"Evaluating {a.upper()}...")
        per_seed_metrics = [evaluate_agent(a, env, args.steps, seed=seed) for seed in EVAL_SEEDS]
        stats = aggregate_metrics(per_seed_metrics)
        results.append({"Agent": f"Hybrid {a.upper()}", "Waiting Time": stats['wait_mean'], "Queue Length": stats['queue_mean']})
        print(f"[{a.upper()}] Avg Waiting Time: {stats['wait_mean']:.2f}s | Avg Queue Avg: {stats['queue_mean']:.2f}\n")

    env.close()
        
    df = pd.DataFrame(results)
    
    os.makedirs("../visualizations/comparison_charts", exist_ok=True)
    df.to_csv("../visualizations/comparison_charts/hybrid_evaluation_results.csv", index=False)
    
    print("\n--- FINAL HYBRID EVALUATION RESULTS ---")
    print(df.to_string(index=False))
