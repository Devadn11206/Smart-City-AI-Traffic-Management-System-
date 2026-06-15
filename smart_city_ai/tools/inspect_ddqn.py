#!/usr/bin/env python3
"""
Inspect the best Hybrid DDQN model and recent training logs for convergence analysis.
"""
import os
import sys
import pandas as pd
import torch
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

results_dir = repo_root / 'results' / 'hybrid_results'
logs_file = repo_root / 'results' / 'hybrid_logs' / 'hybrid_training_log.csv'


def inspect_model():
    model_best = None
    for p in [results_dir / 'hybrid_ddqn_model_best.pth', results_dir / 'hybrid_ddqn_model_final.pth', repo_root.parent / 'results' / 'hybrid_results' / 'hybrid_ddqn_model_best.pth']:
        if p.exists():
            model_best = p
            break

    print('Model path:', model_best)
    if model_best is None:
        print('No DDQN model found.')
        return

    data = torch.load(str(model_best), map_location='cpu')
    # data might be a dict with 'state_dict' or the state_dict directly
    state_dict = None
    if isinstance(data, dict) and 'state_dict' in data:
        state_dict = data['state_dict']
    elif isinstance(data, dict) and any(k.startswith('model') or 'q_network' in k for k in data.keys()):
        state_dict = {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}
    else:
        try:
            state_dict = data
        except Exception:
            state_dict = None

    if state_dict is None:
        print('Could not parse model file structure. Keys:', list(data.keys()) if isinstance(data, dict) else str(type(data)))
        return

    total_params = 0
    print('\nModel parameter summary:')
    for k, v in state_dict.items():
        if hasattr(v, 'shape'):
            num = 1
            for s in v.shape:
                num *= s
            total_params += num
            print(f"{k}: {tuple(v.shape)} -> {num} params")
    print(f"Total params: {total_params}")


def inspect_logs():
    if not logs_file.exists():
        print('Training log not found at', logs_file)
        return
    df = pd.read_csv(logs_file)
    # filter for ddqn
    if 'agent' in df.columns:
        ddqn = df[df['agent'].str.lower() == 'ddqn']
    else:
        ddqn = df
    print('\nLast 10 training rows for DDQN:')
    print(ddqn.tail(10).to_string(index=False))

    print('\nSummary statistics:')
    print(ddqn.describe())


if __name__ == '__main__':
    inspect_model()
    inspect_logs()
