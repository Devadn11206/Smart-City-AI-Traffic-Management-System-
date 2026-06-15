import os
import subprocess
import json
import time
import math
from statistics import mean, stdev

METHODS = ['fixed_time', 'hybrid_dqn', 'hybrid_ddqn', 'hybrid_ppo']
SEEDS = [101, 202, 303, 404, 505]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CONFIG = os.path.join(ROOT, 'sumo_network', 'config.sumocfg')
PINN = os.path.join(ROOT, 'pinn_models', 'trained_pinn.pth')
METRICS_DIR = os.path.join(ROOT, 'results', 'aco_results', 'checkpoints')
LOGS_DIR = os.path.join(ROOT, 'results', 'logs')
MON_DIR = os.path.join(ROOT, 'results', 'monitoring')
REPORT_DIR = os.path.join(ROOT, 'results', 'aco_results')

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MON_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def run_fixed_time(seed):
    run_name = f'fixed_time_seed{seed}'
    cmd = [
        'python', os.path.join(ROOT, 'integration', 'rl_pinn_aco_integration.py'),
        '--config', CONFIG,
        '--duration', '600',
        '--decision-interval', '10000',
        '--pinn', PINN,
        '--metrics-output-dir', METRICS_DIR,
        '--run-name', run_name,
        '--seed', str(seed),
    ]
    logfile = os.path.join(LOGS_DIR, f'{run_name}_{int(time.time())}.log')
    with open(logfile, 'wb') as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    return proc.returncode


def run_hybrid(agent_type, seed):
    run_name = f'hybrid_{agent_type}_seed{seed}'
    cmd = [
        'python', os.path.join(ROOT, 'tools', 'hybrid_runner.py'),
        '--agent', agent_type.replace('hybrid_', ''),
        '--config', CONFIG,
        '--pinn', PINN,
        '--duration', '600',
        '--seed', str(seed),
        '--out-dir', METRICS_DIR,
    ]
    logfile = os.path.join(LOGS_DIR, f'{run_name}_{int(time.time())}.log')
    with open(logfile, 'wb') as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    return proc.returncode


def read_metrics_file(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def compute_stats(values):
    if not values:
        return {'mean': 0.0, 'std': 0.0, '95ci': [0.0, 0.0]}
    n = len(values)
    mu = mean(values)
    s = stdev(values) if n > 1 else 0.0
    # t critical for n
    t_crit = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228}
    t = t_crit.get(n-1, 1.96)
    se = s / math.sqrt(n) if n > 0 else 0.0
    low = mu - t * se
    high = mu + t * se
    return {'mean': mu, 'std': s, '95ci': [low, high]}


def aggregate_and_report():
    # gather per-method per-seed metrics
    per_seed = {}
    reliability = {'checkpoint_loads': {}, 'crashes': 0, 'exceptions': 0, 'monitoring_files': []}

    for method in METHODS:
        per_seed[method] = {}
        for seed in SEEDS:
            fname = ''
            if method == 'fixed_time':
                fname = os.path.join(METRICS_DIR, f'fixed_time_seed{seed}_metrics_latest.json')
            else:
                agent = method.replace('hybrid_', '')
                fname = os.path.join(METRICS_DIR, f'hybrid_{agent}_seed{seed}_metrics_latest.json')
            data = read_metrics_file(fname)
            per_seed[method][str(seed)] = data
            if data is None:
                reliability['crashes'] += 1
            elif isinstance(data, dict) and data.get('error'):
                reliability['exceptions'] += 1
            else:
                # checkpoint_status if present
                cs = data.get('checkpoint_status') if isinstance(data, dict) else None
                reliability['checkpoint_loads'].setdefault(method, []).append(cs)

    # compute aggregate stats
    aggregates = {}
    for method, seeds in per_seed.items():
        waits = []
        queues = []
        throughs = []
        speeds = []
        for s,d in seeds.items():
            if not d or (isinstance(d, dict) and d.get('error')):
                continue
            waits.append(float(d.get('wait', 0.0)))
            queues.append(float(d.get('queue', 0.0)))
            throughs.append(float(d.get('throughput', 0.0)))
            speeds.append(float(d.get('avg_speed', 0.0)))
        aggregates[method] = {
            'wait': compute_stats(waits),
            'queue': compute_stats(queues),
            'throughput': compute_stats(throughs),
            'avg_speed': compute_stats(speeds),
        }

    report = {
        'per_seed': per_seed,
        'aggregates': aggregates,
        'reliability': reliability,
        'methods': METHODS,
        'seeds': SEEDS,
    }
    outpath = os.path.join(REPORT_DIR, 'reproducibility_benchmark_report.json')
    with open(outpath, 'w') as f:
        json.dump(report, f, indent=2)
    print('Report written to', outpath)


if __name__ == '__main__':
    # run everything sequentially
    for method in METHODS:
        for seed in SEEDS:
            print('Running', method, 'seed', seed)
            if method == 'fixed_time':
                rc = run_fixed_time(seed)
            else:
                rc = run_hybrid(method, seed)
            print('Return code', rc)
            # do not abort on non-zero; continue
            time.sleep(2)

    # aggregate
    aggregate_and_report()
    print('Done')
