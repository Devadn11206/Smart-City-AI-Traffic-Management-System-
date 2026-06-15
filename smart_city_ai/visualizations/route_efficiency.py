import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_MODES = ['rl_only', 'rl_pinn', 'rl_pinn_aco']


def load_summary(summary_path):
    if not summary_path.exists():
        return {}
    with open(summary_path, 'r') as f:
        return json.load(f)


def compute_efficiency(metrics):
    wait = float(metrics.get('wait', 0.0) or 0.0)
    queue = float(metrics.get('queue', 0.0) or 0.0)
    throughput = float(metrics.get('throughput', 0.0) or 0.0)
    denom = max(1e-6, wait + queue + 1.0)
    return throughput / denom


def load_checkpoint_series(checkpoints_dir, mode):
    csv_path = checkpoints_dir / f'{mode}_metrics_checkpoints.csv'
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    for col in ['wait', 'queue', 'throughput']:
        if col not in df.columns:
            df[col] = 0.0
    df['route_efficiency'] = df.apply(lambda row: float(row.get('throughput', 0.0)) / max(1e-6, float(row.get('wait', 0.0)) + float(row.get('queue', 0.0)) + 1.0), axis=1)
    return df


def plot_route_efficiency(summary_path, checkpoints_dir, output_dir):
    summary = load_summary(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid')

    rows = []
    for mode in DEFAULT_MODES:
        metrics = (summary.get(mode) or {}).get('metrics', {})
        rows.append({
            'Mode': mode,
            'Efficiency': compute_efficiency(metrics),
            'Wait': float(metrics.get('wait', 0.0) or 0.0),
            'Queue': float(metrics.get('queue', 0.0) or 0.0),
            'Throughput': float(metrics.get('throughput', 0.0) or 0.0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'route_efficiency_summary.csv', index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Mode', y='Efficiency', palette='viridis')
    plt.title('Route Efficiency by Evaluation Mode')
    plt.ylabel('Efficiency Score')
    plt.xlabel('Mode')
    plt.tight_layout()
    plt.savefig(output_dir / 'route_efficiency_comparison.png')
    plt.close()

    # time-series snapshots from checkpoint logs
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for mode, ax in zip(DEFAULT_MODES, axes):
        series = load_checkpoint_series(checkpoints_dir, mode)
        if series.empty:
            ax.set_title(f'{mode} (no checkpoints found)')
            continue
        ax.plot(series['step'], series['route_efficiency'], label='Route Efficiency', color='tab:blue')
        ax.set_title(mode)
        ax.set_ylabel('Efficiency')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Simulation Step')
    plt.tight_layout()
    plt.savefig(output_dir / 'route_efficiency_timeseries.png')
    plt.close()

    print(f'Route-efficiency charts saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', default='results/aco_results/compare_summary.json')
    parser.add_argument('--checkpoints-dir', default='results/aco_results/checkpoints')
    parser.add_argument('--output-dir', default='visualizations/comparison_charts')
    args = parser.parse_args()

    plot_route_efficiency(
        Path(args.summary),
        Path(args.checkpoints_dir),
        Path(args.output_dir),
    )
