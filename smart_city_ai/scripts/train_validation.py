import os
import sys
import json
import csv
import subprocess
import time
import traceback
from pathlib import Path

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
    print(f"Evaluation metrics for {agent_name}: {metrics}")
    return metrics

def main():
    agents = ["dqn", "ddqn", "ppo"]
    episodes = 5
    
    config_file = str(repo_root / "sumo_network" / "config.sumocfg")
    env = SumoEnvironment(config_file, use_gui=False, num_seconds=3600)

    results = {}
    
    for agent_name in agents:
        try:
            run_training(agent_name, episodes)
            metrics = evaluate_agent(agent_name, env)
            results[agent_name] = metrics
        except Exception as e:
            print(f"Skipping {agent_name} due to error: {e}")
            continue

    env.close()
    
    # Generate report
    report_path = repo_root / "results" / "reward_shaping_v2_validation_report.md"
    os.makedirs(repo_root / "results", exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# Reward Shaping V2 Validation Report\n\n")
        f.write("| Agent | Reward | Throughput | Avg Queue | Avg Wait | Switches | Emergency Delay |\n")
        f.write("|-------|--------|------------|-----------|----------|----------|-----------------|\n")
        for agent_name in agents:
            if agent_name in results:
                m = results[agent_name]
                f.write(f"| **{agent_name.upper()}** | {m.get('reward', 0):.2f} | {m.get('throughput', 0):.0f} | {m.get('queue', 0):.2f} | {m.get('wait', 0):.2f} | {m.get('switches', 0)} | {m.get('emergency_delay', 0):.0f} |\n")
        
        f.write("\n## Observations\n")
        f.write("- **Switch Count**: DDQN and PPO must have switches > 0 to indicate no mode collapse.\n")
        f.write("- **Emergency Delay**: Should be significantly reduced compared to 1M+ delays.\n")

if __name__ == "__main__":
    main()
