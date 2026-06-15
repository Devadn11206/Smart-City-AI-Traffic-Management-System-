import subprocess
import sys
import os

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"🚀 STARTING TRAINING: {script_name}")
    print(f"{'='*50}\n")
    
    try:
        # Run the script and stream the output to the console
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: {script_name} failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    
    print(f"\n✅ COMPLETED TRAINING: {script_name}\n")

def main():
    # Make sure we're in the right directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.abspath(os.path.join(current_dir, '..'))
    os.chdir(workspace_dir)
    
    scripts_to_run = [
        "training/train_dqn.py",
        "training/train_ddqn.py",
        "training/train_ppo.py"
    ]
    
    for script in scripts_to_run:
        run_script(script)
        
    print("\n🎉 ALL TRAINING SESSIONS COMPLETED SUCCESSFULLY! 🎉")

if __name__ == "__main__":
    main()
