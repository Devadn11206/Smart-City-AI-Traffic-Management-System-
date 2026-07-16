import os
import subprocess
import time

def run_sanity():
    base_dir = r"c:\Users\ddnan\intenrship_IIIT\benchmark_research"
    os.chdir(base_dir)
    
    experiments = ['dqn', 'ppo']
    
    for algo in experiments:
        print(f"Running sanity check on 4x4 for {algo}...")
        # Clean up any hanging traci/sumo first to avoid socket issues
        subprocess.run("taskkill /F /IM sumo.exe /T", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        
        cmd = f"python 4x4_benchmark\\scripts\\train_4x4.py --algo {algo} --episodes 10"
        subprocess.run(cmd, shell=True)
        print(f"Completed 10 episodes for {algo}. Waiting 5 seconds...")
        time.sleep(5)

if __name__ == "__main__":
    run_sanity()
