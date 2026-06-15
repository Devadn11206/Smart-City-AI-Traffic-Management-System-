import os
import sys
import subprocess
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent

def run_agent_training(agent, episodes):
    print(f"\n=========================================")
    print(f"Running Phase 4 Training for {agent.upper()} ({episodes} episodes)")
    print(f"=========================================\n")
    script_path = repo_root / "training" / f"train_{agent}.py"
    
    cmd = [sys.executable, str(script_path)]
    if agent == "ddqn":
        cmd.extend(["--episodes", str(episodes)])
    else:
        cmd.append(str(episodes))
        
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    agents = {
        "dqn": 150,
        "ddqn": 200,
        "ppo": 300
    }
    
    for agent, episodes in agents.items():
        try:
            run_agent_training(agent, episodes)
        except subprocess.CalledProcessError as e:
            print(f"Error training {agent}: {e}")
            
    print("\nPhase 4 Training Complete. Now running milestone evaluation...")
    eval_script = repo_root / "scripts" / "evaluate_milestones.py"
    subprocess.run([sys.executable, str(eval_script)], check=True)
