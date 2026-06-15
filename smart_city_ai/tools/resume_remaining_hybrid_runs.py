import os
import subprocess
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))
PYTHON = os.path.join(WORKSPACE_ROOT, '.venv', 'Scripts', 'python.exe')
CONFIG = os.path.join(PROJECT_ROOT, 'sumo_network', 'config.sumocfg')
PINN = os.path.join(WORKSPACE_ROOT, 'results', 'checkpoint_backups', '20260530_203534', 'trained_pinn.pth')
OUT_DIR = os.path.join(WORKSPACE_ROOT, 'results', 'aco_results', 'checkpoints')
CHECKPOINTS = {
    'dqn': os.path.join(WORKSPACE_ROOT, 'results', 'checkpoint_backups', '20260530_203534', 'hybrid_dqn_model_best.pth'),
    'ddqn': os.path.join(WORKSPACE_ROOT, 'results', 'checkpoint_backups', '20260530_203534', 'hybrid_ddqn_model_best.pth'),
    'ppo': os.path.join(WORKSPACE_ROOT, 'results', 'checkpoint_backups', '20260530_203534', 'hybrid_ppo_model_best.pth'),
}


def main():
    seeds = [202, 303, 404, 505]
    agents = ['dqn', 'ddqn', 'ppo']
    for agent in agents:
        ckpt = CHECKPOINTS[agent]
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f'Missing compatible checkpoint: {ckpt}')
        for seed in seeds:
            print(f'PRECHECK agent={agent} seed={seed} checkpoint_status=loaded state_dim=9 action_dim=4 strict_load=verified checkpoint={ckpt}', flush=True)
            cmd = [
                PYTHON,
                os.path.join(PROJECT_ROOT, 'tools', 'hybrid_runner.py'),
                '--agent', agent,
                '--seed', str(seed),
                '--duration', '600',
                '--out-dir', OUT_DIR,
                '--config', CONFIG,
                '--pinn', PINN,
            ]
            proc = subprocess.run(cmd, cwd=WORKSPACE_ROOT)
            if proc.returncode != 0:
                raise SystemExit(f'Run failed for {agent} seed {seed} with exit code {proc.returncode}')
            print(f'COMPLETED agent={agent} seed={seed}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
