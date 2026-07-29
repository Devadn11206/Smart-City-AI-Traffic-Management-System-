import os
import shutil

def copy_file(src, dest):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)
        print(f"Copied {src} -> {dest}")
    else:
        print(f"WARNING: File not found {src}")

def copy_dir(src, dest):
    if os.path.exists(src):
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print(f"Copied directory {src} -> {dest}")
    else:
        print(f"WARNING: Directory not found {src}")

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    export_dir = os.path.join(base_dir, "exports")
    
    ppo_dir = os.path.join(export_dir, "ppo_pinn_package")
    dqn_dir = os.path.join(export_dir, "dqn_pinn_package")
    
    print("=== Packaging PPO+PINN ===")
    ppo_files = [
        "training/train_ppo_pinn_6x6.py",
        "training/evaluate_ppo_vs_pinn.py",
        "training/pinn_ensemble_pretrained_6x6.pth",
        "rl_agents/ppo_pinn_agent.py",
        "rl_agents/ppo_agent.py",
        "environment/sumo_environment_6x6.py",
        "pinn_models/pinn_ensemble.py"
    ]
    for f in ppo_files:
        copy_file(os.path.join(base_dir, f), os.path.join(ppo_dir, f))
    copy_dir(os.path.join(base_dir, "sumo_network_6x6"), os.path.join(ppo_dir, "sumo_network_6x6"))
    
    print("\n=== Packaging DQN+PINN ===")
    dqn_files = [
        "training/train_pi_mbpo_6x6.py",
        "training/evaluate_dqn_vs_pinn.py",
        "training/pinn_ensemble_pretrained_6x6.pth",
        "rl_agents/dqn_agent.py",
        "rl_agents/pi_mbpo_agent.py",
        "environment/sumo_environment_6x6.py",
        "pinn_models/pinn_ensemble.py"
    ]
    for f in dqn_files:
        copy_file(os.path.join(base_dir, f), os.path.join(dqn_dir, f))
    copy_dir(os.path.join(base_dir, "sumo_network_6x6"), os.path.join(dqn_dir, "sumo_network_6x6"))
    
    # Optional: also copy __init__.py to all folders to make python module imports happy
    for target in [ppo_dir, dqn_dir]:
        for root, dirs, files in os.walk(target):
            if not os.path.exists(os.path.join(root, "__init__.py")):
                with open(os.path.join(root, "__init__.py"), "w") as f:
                    pass

    print(f"\nSuccessfully packaged to {export_dir}")

if __name__ == "__main__":
    main()
