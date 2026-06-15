"""Collect TraCI latency samples by running the SumoController for a short duration.
Writes `results/traci_latency_samples.csv` with timestamped EWMA and p95 values.

Usage: python scripts/collect_traci_latency.py --duration 30
"""
import argparse
import csv
import time
from pathlib import Path
import sys

# ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dashboards.traci_controller import SumoController


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--duration', type=int, default=30, help='Seconds to run')
    p.add_argument('--out', type=str, default=str(REPO_ROOT / 'results' / 'traci_latency_samples.csv'))
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ctrl = SumoController()
    print('Starting SumoController...')
    ctrl.start()

    start = time.time()
    rows = []
    try:
        while time.time() - start < args.duration:
            snap = ctrl.get_snapshot()
            ts = time.time()
            rows.append({
                'timestamp': ts,
                'sim_time': snap.get('time', 0.0),
                'vehicle_count': snap.get('vehicle_count', 0),
                'traci_ewma_ms': snap.get('traci_step_latency_ms', 0.0),
                'traci_p95_ms': snap.get('traci_step_p95_ms', 0.0),
                'snap_ewma_ms': snap.get('snapshot_latency_ms', 0.0),
                'snap_p95_ms': snap.get('snapshot_p95_ms', 0.0),
            })
            print(f"t={snap.get('time',0):.1f}s vehicles={snap.get('vehicle_count',0)} traci_p95={snap.get('traci_step_p95_ms',0.0):.2f}ms")
            time.sleep(1.0)
    finally:
        print('Stopping controller...')
        ctrl.stop()

    # write CSV
    keys = ['timestamp', 'sim_time', 'vehicle_count', 'traci_ewma_ms', 'traci_p95_ms', 'snap_ewma_ms', 'snap_p95_ms']
    with out_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f'Wrote {len(rows)} samples to {out_path}')


if __name__ == '__main__':
    main()
