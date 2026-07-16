import os
import subprocess
import sys
import concurrent.futures

def run_algo(algo, base_dir, resume_ep=0):
    print(f"==========================================")
    print(f"Starting/Resuming training for {algo.upper()} (Resume Ep: {resume_ep})...")
    print(f"==========================================")
    
    cmd = [sys.executable, "train_4x4.py", "--algo", algo, "--episodes", "200"]
    if resume_ep > 0:
        cmd.extend(["--resume_ep", str(resume_ep)])
        
    subprocess.run(cmd, cwd=base_dir)
    print(f"Finished training for {algo.upper()}.\n")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define tasks: (algo, resume_ep)
    tasks = [
        ('presslight', 150),
        ('colight', 100)
    ]
    
    print("Launching parallel training with max_workers=3...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_algo, algo, base_dir, resume_ep) for algo, resume_ep in tasks]
        concurrent.futures.wait(futures)

    print("All training complete! Running analysis script...")
    subprocess.run([sys.executable, "analyze_4x4.py"], cwd=base_dir)
    print("Analysis complete! Check results/4x4 for graphs and reports.")

if __name__ == "__main__":
    main()
