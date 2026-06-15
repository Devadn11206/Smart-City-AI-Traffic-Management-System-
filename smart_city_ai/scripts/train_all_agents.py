import os
import sys
import json
import csv
import subprocess
import time
import traceback
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from environment.sumo_environment import SumoEnvironment
from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent
from scripts.generate_routes import build_routes

LOG_FILE = repo_root / "logs" / "train_all.log"
JSON_OUT = repo_root / "results" / "training_summary.json"
CSV_OUT = repo_root / "results" / "final_rl_comparison.csv"

def log_msg(msg):
    print(msg)
    os.makedirs(LOG_FILE.parent, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def evaluate_agent(agent_name, env):
    log_msg(f"Evaluating {agent_name}...")
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
    else:
        raise ValueError(f"Unknown agent {agent_name}")

    try:
        agent.load(str(model_path))
        log_msg(f"Loaded {agent_name} model from {model_path}")
    except Exception as e:
        log_msg(f"ERROR: Could not load model for {agent_name}: {e}")
        return None

    # Fixed seed for evaluation to ensure fair comparison
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
    log_msg(f"Evaluation metrics for {agent_name}: {metrics}")
    return metrics

def run_training(agent_name, episodes):
    log_msg(f"Starting training for {agent_name} ({episodes} episodes)")
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
        log_msg(f"Training completed successfully for {agent_name}.")
    except subprocess.CalledProcessError as e:
        log_msg(f"ERROR: Training failed for {agent_name} with exit code {e.returncode}")
        raise e
    except Exception as e:
        log_msg(f"ERROR: Training crashed for {agent_name}: {e}")
        traceback.print_exc()
        raise e

def main():
    log_msg("--- STARTING MASTER TRAINING SCRIPT ---")
    
    os.makedirs(repo_root / "results", exist_ok=True)
    os.makedirs(repo_root / "models", exist_ok=True)
    
    agents = [
        ("dqn", 150),
        ("ddqn", 100),
        ("ppo", 150)
    ]
    
    # Load existing JSON if it exists to continue seamlessly
    summary_data = {}
    if JSON_OUT.exists():
        try:
            with open(JSON_OUT, "r") as f:
                summary_data = json.load(f)
            log_msg("Loaded existing training_summary.json")
        except Exception:
            pass

    # Ensure CSV has headers
    if not CSV_OUT.exists():
        with open(CSV_OUT, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["agent", "reward", "waiting_time", "queue_length", "throughput", "average_speed", "emergency_delay"])

    # Create environment once for evaluation
    config_file = str(repo_root / "sumo_network" / "config.sumocfg")
    env = SumoEnvironment(config_file, use_gui=False, num_seconds=3600)

    for agent_name, episodes in agents:
        if agent_name in summary_data:
            log_msg(f"Agent {agent_name} already exists in summary_data. Skipping training and using existing metrics.")
            continue
            
        try:
            run_training(agent_name, episodes)
            
            # Evaluation
            metrics = evaluate_agent(agent_name, env)
            if metrics:
                # Update JSON
                summary_data[agent_name] = metrics
                with open(JSON_OUT, "w") as f:
                    json.dump(summary_data, f, indent=4)
                
                # Append to CSV
                with open(CSV_OUT, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        agent_name,
                        f"{metrics.get('reward', 0):.2f}",
                        f"{metrics.get('wait', 0):.2f}",
                        f"{metrics.get('queue', 0):.2f}",
                        f"{metrics.get('throughput', 0):.2f}",
                        f"{metrics.get('avg_speed', 0):.2f}",
                        f"{metrics.get('emergency_delay', 0):.2f}"
                    ])
                log_msg(f"Successfully saved metrics for {agent_name}")
                
        except Exception as e:
            log_msg(f"Skipping to next agent due to failure in {agent_name}: {e}")
            continue

    env.close()
    log_msg("--- ALL TRAINING SESSIONS FINISHED ---")

if __name__ == "__main__":
    main()
