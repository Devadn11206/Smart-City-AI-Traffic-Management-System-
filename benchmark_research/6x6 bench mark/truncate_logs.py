import os
import pandas as pd

results_dir = r'C:\Users\ddnan\intenrship_IIIT\benchmark_research\6x6_benchmark\results'
algos = ['idqn', 'dqn', 'ddqn']

for algo in algos:
    print(f'Truncating logs for {algo}...')
    ep_file = os.path.join(results_dir, algo, '6x6_episode_metrics.csv')
    if os.path.exists(ep_file):
        df = pd.read_csv(ep_file)
        if 'episode' in df.columns:
            df = df[df['episode'] <= 150]
        df.to_csv(ep_file, index=False)
        print(f'  Truncated {ep_file} to 150 episodes')
        
    int_file = os.path.join(results_dir, algo, '6x6_intersection_metrics.csv')
    if os.path.exists(int_file):
        df = pd.read_csv(int_file)
        if 'episode' in df.columns:
            df = df[df['episode'] <= 150]
        df.to_csv(int_file, index=False)
        print(f'  Truncated {int_file} to 150 episodes')
