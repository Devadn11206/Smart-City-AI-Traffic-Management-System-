import subprocess
import multiprocessing
import os
import pandas as pd

def run_episode(ep):
    print(f"Starting parallel process for Episode {ep}...")
    cmd = [
        r".venv\Scripts\python.exe",
        r"benchmark_research\pinn\generate_pinn_dataset.py",
        "--episode", str(ep)
    ]
    # run and wait
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f"Finished Episode {ep} with return code {result.returncode}")
    if result.returncode != 0:
        print(f"Error in Episode {ep}:")
        print(result.stderr)

def main():
    episodes = [1, 2, 3, 4, 5]
    print(f"Launching dataset generation in parallel for episodes: {episodes}")
    
    processes = []
    for ep in episodes:
        p = multiprocessing.Process(target=run_episode, args=(ep,))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("All parallel simulations finished. Merging CSVs...")
    
    historical_dir = r"benchmark_research\historical_data"
    merged_file = os.path.join(historical_dir, "traffic_history_action.csv")
    
    dfs = []
    for ep in episodes:
        ep_file = os.path.join(historical_dir, f"traffic_history_action_ep{ep}.csv")
        if os.path.exists(ep_file):
            df = pd.read_csv(ep_file)
            dfs.append(df)
            
    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        # Sort to preserve sequential order
        merged_df.sort_values(by=['algorithm', 'episode', 'intersection_id', 'step_number'], inplace=True)
        merged_df.to_csv(merged_file, index=False)
        print(f"Successfully merged all episodes into {merged_file} with {len(merged_df)} rows.")
        
        # Delete temporary files
        for ep in episodes:
            ep_file = os.path.join(historical_dir, f"traffic_history_action_ep{ep}.csv")
            if os.path.exists(ep_file):
                os.remove(ep_file)
                print(f"Cleaned up temporary file: {ep_file}")
    else:
        print("Error: No dataframes to merge!")

if __name__ == '__main__':
    main()
