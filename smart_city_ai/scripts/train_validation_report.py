import os
import sys
import json
import csv
import subprocess
import time
import traceback
from pathlib import Path
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from environment.sumo_environment import SumoEnvironment
from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent
from scripts.generate_routes import build_routes

def run_training(agent_name, episodes):
    print(f"Starting validation training for {agent_name} ({episodes} episodes)")
    script_path = repo_root / "training" / f"train_{agent_name}.py"
    
    cmd = [sys.executable, str(script_path)]
    if agent_name == "ddqn":
        cmd.extend(["--episodes", str(episodes)])
    else:
        cmd.append(str(episodes))
        
    try:
        subprocess.run(
            cmd,
            check=True,
            cwd=str(repo_root)
        )
        print(f"Training completed successfully for {agent_name}.")
    except Exception as e:
        print(f"ERROR: Training failed for {agent_name}: {e}")
        raise e

def evaluate_agent(agent_name, env):
    print(f"Evaluating {agent_name}...")
    if agent_name == "dqn":
        agent = DQNAgent(env.state_dim, env.action_dim)
        model_path = repo_root / "models" / "dqn" / "dqn_model_best.pth"
    elif agent_name == "ddqn":
        agent = DDQNAgent(env.state_dim, env.action_dim)
        model_path = repo_root / "models" / "ddqn" / "ddqn_model_best.pth"
        if not model_path.exists():
            model_path = repo_root / "results" / "checkpoints" / "ddqn_latest.pth"
    elif agent_name == "ppo":
        agent = PPOAgent(env.state_dim, env.action_dim)
        model_path = repo_root / "models" / "demo" / "ppo_demo_model.pth"
    
    agent.load(str(model_path))

    # Clear previous hybrid logs
    csv_path = repo_root / 'results' / 'hybrid_logs' / 'reward_components.csv'
    try:
        if csv_path.exists():
            os.remove(csv_path)
    except Exception as e:
        print(f"Warning: Could not remove CSV: {e}")

    build_routes(seed=9999)
    state = env.reset(seed=9999)
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(state, eval_mode=True)
        if isinstance(action, tuple):
            action = action[0]
        state, reward, done, _ = env.step(action)
        total_reward += reward

    metrics = env.get_episode_metrics()
    metrics["reward"] = total_reward
    
    # Calculate sum of components from CSV
    components_sum = {}
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for col in df.columns:
            if col not in ['step', 'reward']:
                components_sum[col] = df[col].sum()
    metrics["components"] = components_sum

    print(f"Evaluation metrics for {agent_name}: {metrics}")
    return metrics

def generate_report(episodes=10):
    repo_root = Path(os.path.dirname(os.path.dirname(__file__)))
    
    config_file = repo_root / "sumo_network" / "config.sumocfg"
    env = SumoEnvironment(str(config_file), use_gui=False, num_seconds=3600)
    
    agents = ["dqn", "ddqn", "ppo"]
    results = {}
    
    for agent_name in agents:
        try:
            # Run the training for 10 episodes
            run_training(agent_name, episodes)
            metrics = evaluate_agent(agent_name, env)
            results[agent_name] = metrics
        except Exception as e:
            print(f"Skipping {agent_name} due to error: {e}")
            continue

    # Save raw results
    out_dir = repo_root / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "absolute_reward_validation.json", "w") as f:
        json.dump(results, f, indent=4)

    # Generate report
    report_path = repo_root / "results" / "absolute_reward_validation_report.md"
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("# Absolute Reward Validation Report\n\n")
        f.write("| Agent | Reward | Throughput | Avg Queue | Avg Wait | Switches | Emergency Delay | Actions Distribution |\n")
        f.write("|-------|--------|------------|-----------|----------|----------|-----------------|----------------------|\n")
        for agent_name in agents:
            if agent_name in results:
                m = results[agent_name]
                act_str = str(m.get('actions', {}))
                f.write(f"| **{agent_name.upper()}** | {m.get('reward', 0):.2f} | {m.get('throughput', 0):.0f} | {m.get('queue', 0):.2f} | {m.get('wait', 0):.2f} | {m.get('switches', 0)} | {m.get('emergency_delay', 0):.0f} | {act_str} |\n")
        
        f.write("\n## Reward Component Breakdown\n")
        f.write("Total sums accumulated during evaluation:\n\n")
        for agent_name in agents:
            if agent_name in results and results[agent_name].get("components"):
                f.write(f"### {agent_name.upper()}\n")
                comps = results[agent_name]["components"]
                
                # Calculate total absolute mass
                total_mass = sum(abs(v) for v in comps.values())
                
                # Sort components by dominance
                sorted_comps = sorted(comps.items(), key=lambda x: abs(x[1]), reverse=True)
                
                f.write("| Component | Raw Contribution | % Contribution | Dominance | Flag |\n")
                f.write("|-----------|------------------|----------------|-----------|------|\n")
                for rank, (k, v) in enumerate(sorted_comps, 1):
                    perc = (abs(v) / total_mass) * 100 if total_mass > 0 else 0
                    flag = "⚠️ **>40% DOMINANCE**" if perc > 40 else ""
                    f.write(f"| `{k}` | {v:.2f} | {perc:.1f}% | {rank} | {flag} |\n")
                f.write("\n")

        f.write("## Critical Policy Analysis\n")
        f.write("> [!INFO]\n")
        f.write("> Evaluated against the Absolute Penalties reward design. Review the component breakdowns above to ensure no single factor contributes >40%.\n")

if __name__ == "__main__":
    generate_report(episodes=30)
