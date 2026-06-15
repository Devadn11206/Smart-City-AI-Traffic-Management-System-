import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
METRICS_DIR = os.path.join(BASE, 'results', 'emergency_metrics')
OUT_DIR = os.path.join(BASE, 'visualizations', 'comparison_charts')
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    'quick_test': os.path.join(METRICS_DIR, 'quick_test_metrics.csv'),
    'longer_test': os.path.join(METRICS_DIR, 'longer_test_metrics.csv'),
}

def load_df(path, label):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['run'] = label
    return df

def main():
    q = load_df(FILES['quick_test'], 'quick_test')
    l = load_df(FILES['longer_test'], 'longer_test')
    if q.empty and l.empty:
        print('No metric files found.')
        return
    df = pd.concat([q, l], ignore_index=True)

    metrics = ['avg_speed', 'queue', 'corridor_efficiency', 'eta_seconds']
    for metric in metrics:
        if metric not in df.columns:
            continue
        plt.figure(figsize=(8,4))
        sns.lineplot(data=df, x='step', y=metric, hue='run', marker='o')
        plt.title(f'Comparison: {metric}')
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f'emergency_compare_{metric}.png')
        plt.savefig(out, dpi=150)
        plt.close()
        print('Saved', out)

    # scatter of ETA vs corridor_efficiency
    if 'eta_seconds' in df.columns and 'corridor_efficiency' in df.columns:
        plt.figure(figsize=(6,5))
        sns.scatterplot(data=df, x='eta_seconds', y='corridor_efficiency', hue='run')
        plt.title('ETA vs Corridor Efficiency')
        plt.tight_layout()
        out = os.path.join(OUT_DIR, 'eta_vs_efficiency.png')
        plt.savefig(out, dpi=150)
        plt.close()
        print('Saved', out)

if __name__ == '__main__':
    main()
