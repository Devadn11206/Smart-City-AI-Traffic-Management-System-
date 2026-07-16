import os
import subprocess
import sys
import concurrent.futures

def run_algo(algo, base_dir):
    print(f"==========================================")
    print(f"Starting training for {algo.upper()}...")
    print(f"==========================================")
    
    # Use sys.executable to ensure the same Python environment
    cmd = [sys.executable, "train_4x4.py", "--algo", algo, "--episodes", "200"]
    
    # Run process
    subprocess.run(cmd, cwd=base_dir)
    
    print(f"Finished training for {algo.upper()}.\n")

def main():
    algos = ['idqn', 'dqn', 'ddqn', 'ppo', 'presslight', 'colight']
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Launching parallel training with max_workers=3...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_algo, algo, base_dir) for algo in algos]
        concurrent.futures.wait(futures)

    print("All training complete! Running analysis script...")
    subprocess.run([sys.executable, "analyze_4x4.py"], cwd=base_dir)
    print("Analysis complete! Check results/4x4 for graphs and reports.")

if __name__ == "__main__":
    main()
