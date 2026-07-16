import os
import sys
import shutil
import subprocess
import concurrent.futures

def clean_partial_runs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(os.path.dirname(base_dir), 'results')
    for algo in ['ppo', 'presslight', 'colight']:
        algo_dir = os.path.join(results_dir, algo)
        if os.path.exists(algo_dir):
            print(f'Deleting partial run dir: {algo_dir}')
            shutil.rmtree(algo_dir, ignore_errors=True)

def run_algo(algo_tuple):
    algo, resume_ep = algo_tuple
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_6x6.py')
    
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', algo)
    os.makedirs(log_dir, exist_ok=True)
    
    # We append to training.log if resuming, write if fresh
    mode = 'a' if resume_ep > 0 else 'w'
    log_file = os.path.join(log_dir, 'training.log')
    
    cmd = [sys.executable, script, '--algo', algo, '--episodes', '200']
    if resume_ep > 0:
        cmd.extend(['--resume_ep', str(resume_ep)])
        print(f'[{algo.upper()}] Resuming from ep {resume_ep} to 200...')
    else:
        print(f'[{algo.upper()}] Starting from scratch (ep 1 to 200)...')
        
    try:
        with open(log_file, mode) as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        print(f'[{algo.upper()}] Training COMPLETED successfully.')
        return True
    except subprocess.CalledProcessError as e:
        print(f'[{algo.upper()}] Training FAILED. See {log_file} for details.')
        return False

def main():
    # (algo, resume_ep)
    jobs = [
        ('ppo', 100),
        ('presslight', 150),
        ('colight', 100)
    ]
    
    print('\n--- RESUMING PARALLEL BENCHMARK TRAINING (max_workers=3) ---')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(run_algo, jobs))
        
    if all(results):
        print('\n--- ALL ALGORITHMS TRAINED SUCCESSFULLY ---')
    else:
        print('\n--- SOME ALGORITHMS FAILED. CHECK LOGS. ---')

    print('\n--- RUNNING ANALYSIS ---')
    analyze_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analyze_6x6.py')
    if os.path.exists(analyze_script):
        subprocess.run([sys.executable, analyze_script])
        
    print('\n--- RUNNING AUDIT ---')
    audit_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_6x6.py')
    if os.path.exists(audit_script):
        subprocess.run([sys.executable, audit_script])
        
    print('\n--- 6x6 BENCHMARK PIPELINE COMPLETE ---')

if __name__ == '__main__':
    main()
