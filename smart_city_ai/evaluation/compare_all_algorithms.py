import subprocess
import os
import json
import ast
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from utils.project_paths import ensure_repo_root_on_path
except ModuleNotFoundError:
    from smart_city_ai.utils.project_paths import ensure_repo_root_on_path

repo_root = ensure_repo_root_on_path(__file__)

MODES = [
    ('rl_only', 0, 0),
    ('rl_pinn', 1, 0),
    ('rl_pinn_aco', 1, 1),
]

RUN_CMD = [sys.executable, str(repo_root / 'integration' / 'rl_pinn_aco_integration.py')]
PINN_PATH = 'pinn_models/trained_pinn.pth'

def run_mode(mode_name, enable_pinn, enable_aco, duration=900, checkpoint_interval=250):
    args = RUN_CMD + [
        '--enable-pinn', str(int(enable_pinn)),
        '--enable-aco', str(int(enable_aco)),
        '--duration', str(duration),
        '--run-name', mode_name,
        '--metrics-checkpoint-interval', str(checkpoint_interval),
        '--metrics-output-dir', 'results/aco_results/checkpoints',
        '--decision-interval', '8',
        '--reroute-interval', '45',
        '--pinn-interval', '45'
    ]
    if enable_pinn:
        args += ['--pinn', PINN_PATH]
    logfile = repo_root / 'results' / 'aco_results' / f'{mode_name}.log'
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print('Running', mode_name)
    with open(logfile, 'w') as f:
        proc = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
    # parse metrics from logfile
    metrics = None
    with open(logfile, 'r') as f:
        for line in f:
            if line.startswith('METRICS:'):
                try:
                    metrics = ast.literal_eval(line.replace('METRICS:', '').strip())
                except Exception:
                    metrics = {'raw': line.replace('METRICS:', '').strip()}
    return mode_name, logfile, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=[m[0] for m in MODES], default=None)
    parser.add_argument('--duration', type=int, default=900)
    parser.add_argument('--checkpoint-interval', type=int, default=250)
    args = parser.parse_args()

    out = repo_root / 'results' / 'aco_results' / 'compare_summary.json'
    results = {}
    if args.mode is not None and out.exists():
        try:
            with open(out, 'r') as f:
                results = json.load(f)
        except Exception:
            results = {}
    modes = MODES if args.mode is None else [m for m in MODES if m[0] == args.mode]
    for mode_name, pinn, aco in modes:
        name, log, metrics = run_mode(mode_name, pinn, aco, duration=args.duration, checkpoint_interval=args.checkpoint_interval)
        # ensure JSON serializable (convert Path -> str)
        try:
            log_str = str(log)
        except Exception:
            log_str = log
        results[name] = {'log': log_str, 'metrics': metrics, 'duration': args.duration}

    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved summary to', out)

if __name__ == '__main__':
    main()
