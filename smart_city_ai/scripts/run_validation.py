import os
import sys
import subprocess

repo_root = os.path.dirname(os.path.dirname(__file__))
os.chdir(repo_root)

episodes = 20
agents = ["dqn", "ddqn", "ppo"]

for agent in agents:
    print(f"\n=========================================")
    print(f"Running Validation for {agent.upper()} ({episodes} episodes)")
    print(f"=========================================\n")
    script_path = os.path.join(repo_root, "training", f"train_{agent}.py")
    
    cmd = [sys.executable, script_path]
    if agent == "ddqn":
        cmd.extend(["--episodes", str(episodes)])
    else:
        cmd.append(str(episodes))
        
    subprocess.run(cmd, check=True)
    
print("\nValidation complete. Running evaluation...")
subprocess.run([sys.executable, "scripts/train_validation_report.py"])
