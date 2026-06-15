#!/usr/bin/env python3
"""Clean RL-only benchmark for signal control stability.

Compares:
- Fixed-Time baseline
- DQN-only adaptive control

Uses SumoEnvironment directly so PINN, ACO, and emergency overrides stay disabled.
Also records a signal timeline CSV for switch-frequency and phase-duration analysis.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from statistics import mean, pstdev

import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from integration.hybrid_env import HybridSumoEnv
from rl_agents.dqn_agent import DQNAgent

EVAL_SEEDS = [101, 202, 303]
DEFAULT_PHASE_DURATIONS = [30, 15, 30, 15]


def _model_path() -> Path | None:
    results_dir = repo_root / "results" / "hybrid_results"
    candidates = [
        results_dir / "hybrid_dqn_model_best.pth",
        results_dir / "hybrid_dqn_model_final.pth",
        results_dir / "hybrid_dqn_model.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _env_make(steps: int) -> HybridSumoEnv:
    config_file = repo_root / "sumo_network" / "config.sumocfg"
    pinn_model_path = repo_root / "pinn_models" / "trained_pinn.pth"
    return HybridSumoEnv(str(config_file), str(pinn_model_path), use_gui=False, num_seconds=steps, pinn_weight=0.0, prediction_smoothing=0.0, seed=42)


def _load_dqn(env: HybridSumoEnv) -> DQNAgent:
    model_path = _model_path()
    agent = DQNAgent(env.state_dim, env.action_dim)
    if model_path is None:
        raise FileNotFoundError("No DQN checkpoint found in results/hybrid_results")
    agent.load(str(model_path))
    return agent


def _write_timeline_header(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed",
            "policy",
            "step",
            "current_phase",
            "action",
            "reward",
            "queue",
            "wait",
            "avg_speed",
            "switches",
        ])


def _append_timeline(csv_path: Path, rows):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def run_fixed_baseline(env: HybridSumoEnv, steps: int, seed: int, phase_durations=None):
    if phase_durations is None:
        phase_durations = DEFAULT_PHASE_DURATIONS

    env.reset(seed=seed)
    done = False
    phase_idx = 0
    timeline = []

    while not done and env.current_step < steps:
        duration = max(int(phase_durations[phase_idx % len(phase_durations)]), getattr(env, "min_green", 1))
        repeat_calls = max(1, int(duration / max(1, env.action_repeat)))
        for _ in range(repeat_calls):
            _, reward, done, _ = env.step(0)
            current_phase = -1
            if env.controlled_ts is not None:
                try:
                    import traci
                    current_phase = int(traci.trafficlight.getPhase(env.controlled_ts))
                except Exception:
                    current_phase = -1
            metrics = env.get_episode_metrics()
            timeline.append([
                seed,
                "fixed",
                env.current_step,
                current_phase,
                0,
                float(reward),
                float(metrics.get("queue", 0.0)),
                float(metrics.get("wait", 0.0)),
                float(metrics.get("avg_speed", 0.0)),
                float(metrics.get("switches", 0.0)),
            ])
            if done or env.current_step >= steps:
                break
        phase_idx += 1
        if not done and env.current_step < steps:
            _, reward, done, _ = env.step(1)
            current_phase = -1
            if env.controlled_ts is not None:
                try:
                    import traci
                    current_phase = int(traci.trafficlight.getPhase(env.controlled_ts))
                except Exception:
                    current_phase = -1
            metrics = env.get_episode_metrics()
            timeline.append([
                seed,
                "fixed",
                env.current_step,
                current_phase,
                1,
                float(reward),
                float(metrics.get("queue", 0.0)),
                float(metrics.get("wait", 0.0)),
                float(metrics.get("avg_speed", 0.0)),
                float(metrics.get("switches", 0.0)),
            ])

    metrics = env.get_episode_metrics()
    env.close()
    return metrics, timeline


def run_dqn(env: HybridSumoEnv, steps: int, seed: int):
    env.reset(seed=seed)
    agent = _load_dqn(env)
    done = False
    state = env._get_state()
    timeline = []

    while not done and env.current_step < steps:
        action = agent.act(state, eval_mode=True)
        next_state, reward, done, _ = env.step(action)
        current_phase = -1
        if env.controlled_ts is not None:
            try:
                import traci
                current_phase = int(traci.trafficlight.getPhase(env.controlled_ts))
            except Exception:
                current_phase = -1
        metrics = env.get_episode_metrics()
        timeline.append([
            seed,
            "dqn",
            env.current_step,
            current_phase,
            int(action),
            float(reward),
            float(metrics.get("queue", 0.0)),
            float(metrics.get("wait", 0.0)),
            float(metrics.get("avg_speed", 0.0)),
            float(metrics.get("switches", 0.0)),
        ])
        state = next_state

    metrics = env.get_episode_metrics()
    env.close()
    return metrics, timeline


def aggregate(metrics_list):
    return {
        "wait_mean": float(mean([m.get("wait", 0.0) for m in metrics_list])) if metrics_list else 0.0,
        "queue_mean": float(mean([m.get("queue", 0.0) for m in metrics_list])) if metrics_list else 0.0,
        "wait_std": float(pstdev([m.get("wait", 0.0) for m in metrics_list])) if len(metrics_list) > 1 else 0.0,
        "queue_std": float(pstdev([m.get("queue", 0.0) for m in metrics_list])) if len(metrics_list) > 1 else 0.0,
        "speed_mean": float(mean([m.get("avg_speed", 0.0) for m in metrics_list])) if metrics_list else 0.0,
        "throughput_mean": float(mean([m.get("throughput", 0.0) for m in metrics_list])) if metrics_list else 0.0,
        "switches_mean": float(mean([m.get("switches", 0.0) for m in metrics_list])) if metrics_list else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=str, default="101,202,303")
    args = parser.parse_args()

    seeds = [int(v.strip()) for v in args.seeds.split(",") if v.strip()]
    timeline_csv = repo_root / "results" / "hybrid_logs" / "signal_timeline.csv"
    _write_timeline_header(timeline_csv)

    results = []
    print(f"Running DQN-only stability evaluation ({args.steps} steps per seed)...")

    fixed_metrics = []
    for seed in seeds:
        env = _env_make(args.steps)
        metrics, timeline = run_fixed_baseline(env, args.steps, seed)
        fixed_metrics.append(metrics)
        _append_timeline(timeline_csv, timeline)
    fixed_stats = aggregate(fixed_metrics)
    results.append({"Agent": "Fixed-Time", "Waiting Time": fixed_stats["wait_mean"], "Queue Length": fixed_stats["queue_mean"], "Type": "Baseline"})
    print(f"[FIXED-TIME] Wait: {fixed_stats['wait_mean']:.2f}s | Queue: {fixed_stats['queue_mean']:.2f}")

    dqn_metrics = []
    for seed in seeds:
        env = _env_make(args.steps)
        metrics, timeline = run_dqn(env, args.steps, seed)
        dqn_metrics.append(metrics)
        _append_timeline(timeline_csv, timeline)
    dqn_stats = aggregate(dqn_metrics)
    results.append({"Agent": "DQN Only", "Waiting Time": dqn_stats["wait_mean"], "Queue Length": dqn_stats["queue_mean"], "Type": "RL"})
    print(f"[DQN] Wait: {dqn_stats['wait_mean']:.2f}s | Queue: {dqn_stats['queue_mean']:.2f}")

    df = pd.DataFrame(results)
    out_dir = repo_root / "visualizations" / "comparison_charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "dqn_signal_stability_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved results to: {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
