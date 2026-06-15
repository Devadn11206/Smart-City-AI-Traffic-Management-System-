#!/usr/bin/env python3
"""Analyze reward components and signal stability from generated CSV logs.

Inputs:
- results/hybrid_logs/reward_components.csv
- results/hybrid_logs/signal_timeline.csv

Outputs:
- visualizations/reward_analysis/*.png
- visualizations/reward_analysis/analysis_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent


def _rolling(series, window=25):
    return series.rolling(window=min(window, max(1, len(series))), min_periods=1).mean()


def analyze_reward_components(reward_csv: Path, out_dir: Path):
    if not reward_csv.exists():
        return None

    df = pd.read_csv(reward_csv)
    if df.empty:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = df.describe(include="all").transpose()
    summary.to_csv(out_dir / "reward_component_summary.csv")

    components = [c for c in ["queue_delta", "wait_delta", "throughput", "speed", "congestion", "switch_penalty", "persistence_bonus", "reward"] if c in df.columns]

    plt.figure(figsize=(12, 6))
    for col in components:
        plt.plot(df["step"], _rolling(df[col]), label=col)
    plt.title("Reward Components (rolling mean)")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "reward_components_rolling.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(df["reward"], bins=50, alpha=0.85)
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_dir / "reward_distribution.png", dpi=160)
    plt.close()

    spikes = df["reward"].abs().sort_values(ascending=False).head(20)
    spikes.to_csv(out_dir / "reward_spikes_top20.csv", index=False, header=["abs_reward"])
    return summary


def analyze_signal_timeline(signal_csv: Path, out_dir: Path):
    if not signal_csv.exists():
        return None

    df = pd.read_csv(signal_csv)
    if df.empty:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for policy, group in df.groupby("policy"):
        group = group.sort_values("step")
        switch_rate = float((group["action"] == 1).mean()) if "action" in group else 0.0
        avg_switches = float(group["switches"].iloc[-1]) if "switches" in group and len(group) else 0.0
        avg_queue = float(group["queue"].mean()) if "queue" in group else 0.0
        summary_rows.append({
            "policy": policy,
            "rows": len(group),
            "switch_action_rate": switch_rate,
            "final_switches": avg_switches,
            "avg_queue": avg_queue,
            "avg_wait": float(group["wait"].mean()) if "wait" in group else 0.0,
        })

        plt.figure(figsize=(12, 5))
        plt.step(group["step"], group["current_phase"], where="post")
        plt.title(f"Phase Timeline - {policy}")
        plt.xlabel("Step")
        plt.ylabel("Phase")
        plt.tight_layout()
        plt.savefig(out_dir / f"phase_timeline_{policy}.png", dpi=160)
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(group["step"], _rolling(group["queue"]), label="queue")
        if "wait" in group:
            plt.plot(group["step"], _rolling(group["wait"]), label="wait")
        plt.title(f"Queue / Wait Trend - {policy}")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"queue_wait_{policy}.png", dpi=160)
        plt.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "signal_stability_summary.csv", index=False)
    return summary_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-csv", default=str(repo_root / "results" / "hybrid_logs" / "reward_components.csv"))
    parser.add_argument("--signal-csv", default=str(repo_root / "results" / "hybrid_logs" / "signal_timeline.csv"))
    parser.add_argument("--out-dir", default=str(repo_root / "visualizations" / "reward_analysis"))
    args = parser.parse_args()

    reward_csv = Path(args.reward_csv)
    signal_csv = Path(args.signal_csv)
    out_dir = Path(args.out_dir)

    reward_summary = analyze_reward_components(reward_csv, out_dir)
    signal_summary = analyze_signal_timeline(signal_csv, out_dir)

    print(f"Saved analysis to {out_dir}")
    if reward_summary is not None:
        print("Reward components summary generated.")
    else:
        print("Reward CSV not found or empty.")
    if signal_summary is not None:
        print("Signal stability summary generated.")
    else:
        print("Signal CSV not found or empty.")


if __name__ == "__main__":
    main()
