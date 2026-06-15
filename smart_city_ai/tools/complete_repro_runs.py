import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
METRICS_DIR = os.path.join(ROOT, 'results', 'aco_results', 'checkpoints')
CONFIG = os.path.join(ROOT, 'sumo_network', 'config.sumocfg')
PINN = os.path.join(ROOT, 'pinn_models', 'trained_pinn.pth')

METHODS = ['fixed_time', 'hybrid_dqn', 'hybrid_ddqn', 'hybrid_ppo']
SEEDS = [101,202,303,404,505]

missing = []
for method in METHODS:
    for seed in SEEDS:
        if method == 'fixed_time':
            fname = os.path.join(METRICS_DIR, f'fixed_time_seed{seed}_metrics_latest.json')
        else:
            agent = method.replace('hybrid_', '')
            fname = os.path.join(METRICS_DIR, f'hybrid_{agent}_seed{seed}_metrics_latest.json')
        if not os.path.exists(fname):
            missing.append((method, seed))

print('Missing runs:', missing)

for method, seed in missing:
    print('Running missing', method, seed)
    if method == 'fixed_time':
        cmd = [
            'py', '-3', os.path.join(ROOT, 'integration', 'rl_pinn_aco_integration.py'),
            '--config', CONFIG,
            '--duration', '600',
            '--decision-interval', '10000',
            '--pinn', PINN,
            '--metrics-output-dir', METRICS_DIR,
            '--run-name', f'fixed_time_seed{seed}',
            '--seed', str(seed),
        ]
    else:
        agent = method.replace('hybrid_', '')
        cmd = [
            'py', '-3', os.path.join(ROOT, 'tools', 'hybrid_runner.py'),
            '--agent', agent,
            '--config', CONFIG,
            '--pinn', PINN,
            '--duration', '600',
            '--seed', str(seed),
            '--out-dir', METRICS_DIR,
        ]
    print('CMD:', ' '.join(cmd))
    subprocess.run(cmd)

print('Done')
