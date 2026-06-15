#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style='whitegrid')
repo_root = Path(__file__).parent.parent
csv_path = repo_root / 'visualizations' / 'comparison_charts' / 'per_vehicle_evaluation_results.csv'
out_dir = repo_root / 'visualizations' / 'comparison_charts'

if not csv_path.exists():
    print(f"per-vehicle CSV not found: {csv_path}")
    raise SystemExit(1)

df = pd.read_csv(csv_path)

# Ensure nice order
if 'Agent' in df.columns:
    order = df['Agent'].tolist()
else:
    order = None

# Avg wait per vehicle
plt.figure(figsize=(8,6))
sns.barplot(data=df, x='Agent', y='AvgWaitPerVehicle', palette='viridis')
plt.title('Average Wait per Vehicle')
plt.ylabel('Avg Wait (s)')
plt.xlabel('Agent')
plt.tight_layout()
plt.savefig(out_dir / 'per_vehicle_avg_wait.png', dpi=150)
plt.close()

# Throughput
plt.figure(figsize=(8,6))
sns.barplot(data=df, x='Agent', y='Throughput', palette='magma')
plt.title('Total Throughput (arrived vehicles)')
plt.ylabel('Throughput (vehicles)')
plt.xlabel('Agent')
plt.tight_layout()
plt.savefig(out_dir / 'per_vehicle_throughput.png', dpi=150)
plt.close()

print(f"Saved per-vehicle plots to {out_dir}")
