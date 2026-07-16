import os
import subprocess
import sys

def main():
    algos = ['idqn', 'dqn', 'ddqn', 'ppo', 'presslight', 'colight']
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for algo in algos:
        print(f"==========================================")
        print(f"Starting training for {algo.upper()}...")
        print(f"==========================================")
        
        # Use sys.executable to ensure the same Python environment
        cmd = [sys.executable, "train.py", "--algo", algo, "--episodes", "200"]
        
        # Run process sequentially
        subprocess.run(cmd, cwd=base_dir)
        
        print(f"Finished training for {algo.upper()}.\n")

    print("All training complete! Running analysis script...")
    subprocess.run([sys.executable, "analyze.py"], cwd=base_dir)
    print("Analysis complete! Check results/2x2 for graphs and reports.")

if __name__ == "__main__":
    main()
