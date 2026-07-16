import os
import sys
import subprocess
import concurrent.futures
from pathlib import Path

def run_sanity_check():
    print("\n--- RUNNING 6x6 SANITY CHECK ---")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_6x6.py')
    
    # We test DQN for 3 episodes to ensure environment loads, actions work, and rewards compute correctly
    # 10 episodes takes too long for a synchronous check before parallel spawning.
    print("Testing environment load and TraCI connection with DQN (3 episodes)...")
    try:
        subprocess.run(
            [sys.executable, script, '--algo', 'dqn', '--episodes', '3'],
            check=True
        )
        print("Sanity Check Passed: Environment loaded, TraCI connected, rewards computed successfully.\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Sanity Check Failed: {e}")
        return False

def train_algo(algo):
    print(f"[{algo.upper()}] Starting 200-episode training...")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_6x6.py')
    try:
        # Output to log file to prevent console spam
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', algo)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'training.log')
        
        with open(log_file, 'w') as f:
            process = subprocess.run(
                [sys.executable, script, '--algo', algo, '--episodes', '200'],
                check=True, stdout=f, stderr=subprocess.STDOUT
            )
        print(f"[{algo.upper()}] Training COMPLETED successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{algo.upper()}] Training FAILED. See {log_file} for details.")
        return False

def main():
    if not run_sanity_check():
        print("Aborting benchmark execution due to sanity check failure.")
        return
        
    algorithms = ['idqn', 'dqn', 'ddqn', 'ppo', 'presslight', 'colight']
    
    print("\n--- STARTING PARALLEL BENCHMARK TRAINING (max_workers=3) ---")
    
    # max_workers=3 to prevent system overload
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(train_algo, algorithms))
        
    if all(results):
        print("\n--- ALL ALGORITHMS TRAINED SUCCESSFULLY ---")
    else:
        print("\n--- SOME ALGORITHMS FAILED. CHECK LOGS. ---")
        # Continue to analysis anyway to plot what finished

    print("\n--- RUNNING ANALYSIS ---")
    analyze_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analyze_6x6.py')
    if os.path.exists(analyze_script):
        subprocess.run([sys.executable, analyze_script])
        
    print("\n--- RUNNING AUDIT ---")
    audit_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_6x6.py')
    if os.path.exists(audit_script):
        subprocess.run([sys.executable, audit_script])
        
    print("\n--- RUNNING SCALABILITY ANALYSIS ---")
    scalability_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scalability_analysis.py')
    if os.path.exists(scalability_script):
        subprocess.run([sys.executable, scalability_script])

    print("\n--- 6x6 BENCHMARK PIPELINE COMPLETE ---")

if __name__ == '__main__':
    main()
