import os
import json
import math
from statistics import mean, stdev

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))
PROJECT_METRICS_DIR = os.path.join(PROJECT_ROOT, 'results', 'aco_results', 'checkpoints')
WORKSPACE_METRICS_DIR = os.path.join(WORKSPACE_ROOT, 'results', 'aco_results', 'checkpoints')
REPORT_DIR = os.path.join(WORKSPACE_ROOT, 'results', 'aco_results')
METHODS = ['fixed_time', 'hybrid_dqn', 'hybrid_ddqn', 'hybrid_ppo']
SEEDS = [101, 202, 303, 404, 505]


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
    t_crit = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571}
    t = t_crit.get(n, 1.96)
    se = s / math.sqrt(n) if n > 0 else 0.0
    low = mu - t * se
    high = mu + t * se
    return {'mean': mu, 'std': s, '95ci': [low, high]}


def aggregate_and_report():
    per_seed = {}
    reliability = {'checkpoint_loads': {}, 'crashes': 0, 'exceptions': 0, 'monitoring_files': [], 'telemetry_files': []}

    for method in METHODS:
        per_seed[method] = {}
        for seed in SEEDS:
            if method == 'fixed_time':
                candidates = [
                    os.path.join(PROJECT_METRICS_DIR, f'fixed_time_seed{seed}_metrics_latest.json'),
                    os.path.join(WORKSPACE_METRICS_DIR, f'fixed_time_seed{seed}_metrics_latest.json'),
                ]
            else:
                agent = method.replace('hybrid_', '')
                candidates = [
                    os.path.join(WORKSPACE_METRICS_DIR, f'hybrid_{agent}_seed{seed}_metrics_latest.json'),
                    os.path.join(PROJECT_METRICS_DIR, f'hybrid_{agent}_seed{seed}_metrics_latest.json'),
                ]
            data = None
            for candidate in candidates:
                data = read_metrics_file(candidate)
                if data is not None:
                    break
            per_seed[method][str(seed)] = data
            if data is None:
                reliability['crashes'] += 1
            elif isinstance(data, dict) and data.get('error'):
                reliability['exceptions'] += 1
            else:
                cs = data.get('checkpoint_status') if isinstance(data, dict) else None
                reliability['checkpoint_loads'].setdefault(method, []).append(cs)

    aggregates = {}
    for method, seeds in per_seed.items():
        waits=[]; queues=[]; throughs=[]; speeds=[]
        for s,d in seeds.items():
            if not d or (isinstance(d, dict) and d.get('error')):
                continue
            waits.append(float(d.get('wait',0.0)))
            queues.append(float(d.get('queue',0.0)))
            throughs.append(float(d.get('throughput',0.0)))
            speeds.append(float(d.get('avg_speed',0.0)))
        aggregates[method]={
            'wait': compute_stats(waits),
            'queue': compute_stats(queues),
            'throughput': compute_stats(throughs),
            'avg_speed': compute_stats(speeds),
        }

    # Method ranking by mean wait (lower better)
    ranking = sorted([(m, aggregates[m]['wait']['mean']) for m in METHODS], key=lambda x: x[1])

    monitoring_dirs = [
        os.path.join(WORKSPACE_ROOT, 'results', 'monitoring'),
        os.path.join(PROJECT_ROOT, 'results', 'monitoring'),
    ]
    monitoring_files = []
    for directory in monitoring_dirs:
        if os.path.isdir(directory):
            for name in sorted(os.listdir(directory)):
                if name.endswith('.jsonl'):
                    monitoring_files.append(os.path.join(directory, name))
    # deduplicate while preserving order
    seen = set()
    deduped_monitoring = []
    for path in monitoring_files:
        if path not in seen:
            deduped_monitoring.append(path)
            seen.add(path)
    reliability['monitoring_files'] = deduped_monitoring

    report = {
        'per_seed': per_seed,
        'aggregates': aggregates,
        'reliability': reliability,
        'methods': METHODS,
        'seeds': SEEDS,
        'ranking_by_wait': ranking,
    }
    outpath = os.path.join(REPORT_DIR, 'reproducibility_benchmark_report.json')
    with open(outpath, 'w') as f:
        json.dump(report, f, indent=2)
    print('Regenerated report at', outpath)


if __name__ == '__main__':
    aggregate_and_report()
