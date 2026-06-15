import argparse
import sys

from utils.project_paths import ensure_repo_root_on_path

repo_root = ensure_repo_root_on_path(__file__)

def evaluate(args):
    from evaluation.compare_all_algorithms import main as run_compare_all

    print('Starting ACO benchmark suite (RL-only, RL+PINN, RL+PINN+ACO).')
    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], '--duration', str(args.duration), '--checkpoint-interval', str(args.checkpoint_interval)]
        run_compare_all()
    finally:
        sys.argv = old_argv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='sumo_network/config.sumocfg')
    parser.add_argument('--pinn', default=None)
    parser.add_argument('--duration', type=int, default=900)
    parser.add_argument('--checkpoint-interval', type=int, default=250)
    args = parser.parse_args()
    evaluate(args)
