#!/usr/bin/env python3
"""Simple hyperparameter sweep runner for DDQN using `training.train_ddqn.train_ddqn`.

This script runs short training sessions with different hyperparameter combos and
writes a CSV report to `results/hyperparam_sweeps/ddqn_sweep_results.csv`.

Usage: python tools/ddqn_sweep.py --episodes 10 --out results/hyperparam_sweeps/ddqn_sweep_results.csv
"""
import argparse
import itertools
import csv
import time
from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from training.train_ddqn import train_ddqn


def default_grid():
    return {
        'lr': [5e-5, 1e-4],
        'batch_size': [128, 256],
        'epsilon_decay': [0.999, 0.995],
        'target_update': [500, 1000],
    }


def run_sweep(grid, episodes=10, num_seconds=2000, seed=42, out_path=None):
    out_path = Path(out_path or (repo_root / 'results' / 'hyperparam_sweeps' / 'ddqn_sweep_results.csv'))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))

    header = keys + ['episodes', 'num_seconds', 'seed', 'best_reward', 'mean_reward', 'status', 'elapsed_s']

    # append mode
    write_header = not out_path.exists()
    with open(out_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(header)

        for i, combo in enumerate(combos):
            cfg = {k: v for k, v in zip(keys, combo)}
            cfg.update({'episodes': episodes, 'num_seconds': num_seconds, 'seed': seed + i})

            # ensure SUMO config path is absolute and points to smart_city_ai/sumo_network/config.sumocfg
            cfg.setdefault('config_file', str(repo_root / 'sumo_network' / 'config.sumocfg'))

            print(f"Running combo {i+1}/{len(combos)}: {cfg}")
            start = time.time()
            status = 'ok'
            try:
                result = train_ddqn(cfg)
                if result is None:
                    status = 'skipped'
                    best = ''
                    mean = ''
                else:
                    best = result.get('best_reward', '')
                    hist = result.get('reward_history', []) or []
                    mean = sum(hist) / len(hist) if hist else ''
            except Exception as e:
                status = f'error: {type(e).__name__}'
                best = ''
                mean = ''
            elapsed = round(time.time() - start, 2)

            row = [cfg[k] for k in keys] + [episodes, num_seconds, cfg['seed'], best, mean, status, elapsed]
            writer.writerow(row)
            csvfile.flush()

    print(f"Sweep finished. Results written to: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--num-seconds', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default=None)
    parser.add_argument('--grid-json', default=None, help='Path to JSON file with grid definition (optional)')
    args = parser.parse_args()

    grid = default_grid()
    run_sweep(grid, episodes=args.episodes, num_seconds=args.num_seconds, seed=args.seed, out_path=args.out)
