import os
import sys
import json
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

def evaluate_checkpoint(agent_name, env, episode):
    print(f"Evaluating {agent_name.upper()} at Episode {episode}...")
    
    if agent_name == "dqn":
        agent = DQNAgent(env.state_dim, env.action_dim)
        model_path = repo_root / "models" / "dqn" / f"dqn_model_ep_{episode}.pth"
    elif agent_name == "ddqn":
        agent = DDQNAgent(env.state_dim, env.action_dim)
        model_path = repo_root / "models" / "ddqn" / f"ddqn_model_ep_{episode}.pth"
    elif agent_name == "ppo":
        agent = PPOAgent(env.state_dim, env.action_dim)
        model_path = repo_root / "models" / "ppo" / f"ppo_model_ep_{episode}.pth"
    
    if not model_path.exists():
        print(f"  -> Checkpoint {model_path} not found. Skipping.")
        return None

    try:
        agent.load(str(model_path))
    except Exception as e:
        print(f"  -> Failed to load checkpoint: {e}")
        return None

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
    
    # Emergency completion tracking
    completed_evs = len(getattr(env, '_completed_evs', []))
    metrics["ev_completed"] = completed_evs
    
    print(f"  -> Reward: {total_reward:.2f}, Throughput: {metrics['throughput']}, Delay: {metrics['emergency_delay']:.0f}")
    return metrics

def generate_progress_report():
    config_file = repo_root / "sumo_network" / "config.sumocfg"
    env = SumoEnvironment(str(config_file), use_gui=False, num_seconds=3600)
    
    agents = {
        "dqn": [25, 50, 100, 150],
        "ddqn": [25, 50, 100, 150, 200],
        "ppo": [25, 50, 100, 150, 200, 300]
    }
    
    report_data = []

    for agent_name, milestones in agents.items():
        for ep in milestones:
            metrics = evaluate_checkpoint(agent_name, env, ep)
            if metrics:
                report_data.append({
                    "Agent": agent_name.upper(),
                    "Episode": ep,
                    "Reward": metrics.get("reward", 0),
                    "Throughput": metrics.get("throughput", 0),
                    "Queue": metrics.get("queue", 0),
                    "Wait": metrics.get("wait", 0),
                    "EV Delay": metrics.get("emergency_delay", 0),
                    "EV Complete": metrics.get("ev_completed", 0),
                    "Switches": metrics.get("switches", 0)
                })

    report_path = repo_root / "results" / "training_progress_report.md"
    out_dir = repo_root / "results"
    out_dir.mkdir(exist_ok=True)
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("# Phase 4 Convergence Training Progress Report\n\n")
        f.write("| Agent | Milestone (Ep) | Reward | Throughput | Avg Queue | Avg Wait | EV Delay | EV Complete | Switches |\n")
        f.write("|-------|----------------|--------|------------|-----------|----------|----------|-------------|----------|\n")
        
        for row in report_data:
            f.write(f"| **{row['Agent']}** | {row['Episode']} | {row['Reward']:.2f} | {row['Throughput']:.0f} | {row['Queue']:.2f} | {row['Wait']:.2f} | {row['EV Delay']:.0f} | {row['EV Complete']} | {row['Switches']} |\n")

    print(f"\nReport generated at: {report_path}")

if __name__ == "__main__":
    generate_progress_report()
