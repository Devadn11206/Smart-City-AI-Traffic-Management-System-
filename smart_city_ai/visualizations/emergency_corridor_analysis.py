import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_MODES = ['rl_only', 'rl_pinn', 'rl_pinn_aco']


def load_checkpoint_series(checkpoints_dir, mode):
    csv_path = checkpoints_dir / f'{mode}_metrics_checkpoints.csv'
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def plot_emergency_corridor_analysis(checkpoints_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid')

    summary_rows = []
    for mode in DEFAULT_MODES:
        df = load_checkpoint_series(checkpoints_dir, mode)
        if df.empty:
            continue

        if 'emergency_vehicle_count' not in df.columns:
            df['emergency_vehicle_count'] = 0
        if 'active_corridor_tls' not in df.columns:
            df['active_corridor_tls'] = 0
        if 'emergency_phase_count' not in df.columns:
            df['emergency_phase_count'] = 0

        summary_rows.append({
            'Mode': mode,
            'AvgEmergencyVehicles': float(df['emergency_vehicle_count'].mean()),
            'AvgCorridorTLS': float(df['active_corridor_tls'].mean()),
            'AvgEmergencyPhases': float(df['emergency_phase_count'].mean()),
            'PeakEmergencyVehicles': float(df['emergency_vehicle_count'].max()),
        })

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(df['step'], df['emergency_vehicle_count'], label='Emergency Vehicles', color='tab:red')
        ax1.set_ylabel('Emergency Vehicles')
        ax1.set_xlabel('Simulation Step')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Emergency Corridor Activity - {mode}')

        ax2 = ax1.twinx()
        ax2.plot(df['step'], df['active_corridor_tls'], label='Active Corridor TLS', color='tab:blue', alpha=0.8)
        ax2.set_ylabel('Active Corridor TLS')
        fig.tight_layout()
        fig.savefig(output_dir / f'{mode}_emergency_corridor_activity.png')
        plt.close(fig)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / 'emergency_corridor_summary.csv', index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=summary_df, x='Mode', y='AvgCorridorTLS', palette='magma')
        plt.title('Average Active Corridor TLS by Mode')
        plt.tight_layout()
        plt.savefig(output_dir / 'emergency_corridor_comparison.png')
        plt.close()

    print(f'Emergency-corridor analytics saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints-dir', default='results/aco_results/checkpoints')
    parser.add_argument('--output-dir', default='visualizations/comparison_charts')
    args = parser.parse_args()

    plot_emergency_corridor_analysis(Path(args.checkpoints_dir), Path(args.output_dir))
