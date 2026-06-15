import argparse
import subprocess
import sys
from pathlib import Path


MODES = [
    'rl_only',
    'rl_pinn',
    'rl_pinn_aco',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=900)
    parser.add_argument('--checkpoint-interval', type=int, default=250)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    runner = repo_root / 'evaluation' / 'compare_all_algorithms.py'

    for mode in MODES:
        print(f'Running {mode} as an isolated comparison run...')
        cmd = [
            sys.executable,
            str(runner),
            '--mode', mode,
            '--duration', str(args.duration),
            '--checkpoint-interval', str(args.checkpoint_interval),
        ]
        subprocess.run(cmd, cwd=str(repo_root), check=False)

    print('Split comparison run finished.')


if __name__ == '__main__':
    main()
