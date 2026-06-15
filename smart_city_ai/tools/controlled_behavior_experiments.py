#!/usr/bin/env python3
"""Controlled A/B experiments for emergency corridor and ACO behavior.

Runs OFF/ON pairs with the same seed, duration, and demand setup, then writes
comparison CSVs plus the raw debug/reroute logs used to prove the behavior.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
PYTHON = WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe"
EMERGENCY_SCRIPT = REPO_ROOT / "integration" / "run_emergency_system.py"
ACO_SCRIPT = REPO_ROOT / "integration" / "rl_pinn_aco_integration.py"
EMERGENCY_CONFIG = REPO_ROOT / "sumo_network" / "config.sumocfg"
PINN_CHECKPOINT = REPO_ROOT / "pinn_models" / "trained_pinn.pth"
OUTPUT_ROOT = WORKSPACE_ROOT / "results" / "controlled_experiments"


def run_command(args: List[str]) -> None:
    completed = subprocess.run(args, cwd=str(WORKSPACE_ROOT), check=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(args)}")


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def improvement_pct(baseline: float, improved: float, reverse: bool = False) -> float:
    if baseline == 0:
        return 0.0
    if reverse:
        return round((improved - baseline) / baseline * 100.0, 2)
    return round((baseline - improved) / baseline * 100.0, 2)


def summarize_emergency_debug(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "travel_time": 0.0,
            "average_speed": 0.0,
            "complete_stops": 0.0,
            "waiting_time": 0.0,
            "intersections_crossed": 0.0,
            "traffic_lights_changed": 0.0,
            "vehicles_slowed": 0.0,
            "vehicles_rerouted": 0.0,
        }

    vehicle_rows = []
    for vehicle_id, group in df.groupby("vehicle_id"):
        group = group.sort_values("step")
        travel_time = float(group["step"].max() - group["step"].min() + 1)
        average_speed = float(group["current_speed"].mean())
        complete_stops = float((group["current_speed"] <= 0.1).sum())
        waiting_time = float(group["waiting_time"].max()) if "waiting_time" in group else 0.0
        intersections_crossed = float(group["signalized_intersections_on_route"].max()) if "signalized_intersections_on_route" in group else 0.0
        traffic_lights_changed = float(group["traffic_light_changes"].max()) if "traffic_light_changes" in group else 0.0
        vehicle_rows.append(
            {
                "travel_time": travel_time,
                "average_speed": average_speed,
                "complete_stops": complete_stops,
                "waiting_time": waiting_time,
                "intersections_crossed": intersections_crossed,
                "traffic_lights_changed": traffic_lights_changed,
            }
        )

    slowed_ids = set()
    rerouted_ids = set()
    if "vehicles_cleared_ids" in df.columns:
        for value in df["vehicles_cleared_ids"].dropna().astype(str):
            slowed_ids.update(item for item in value.split(";") if item)
    if "vehicles_rerouted_ids" in df.columns:
        for value in df["vehicles_rerouted_ids"].dropna().astype(str):
            rerouted_ids.update(item for item in value.split(";") if item)

    agg = pd.DataFrame(vehicle_rows).mean(numeric_only=True).to_dict()
    agg["vehicles_slowed"] = float(len(slowed_ids))
    agg["vehicles_rerouted"] = float(len(rerouted_ids))
    return agg


def run_emergency_experiment(seed: int, duration: int) -> Dict[str, Dict[str, float]]:
    out_dir = OUTPUT_ROOT / "emergency"
    off_metrics = out_dir / "off_metrics"
    on_metrics = out_dir / "on_metrics"
    off_debug = out_dir / "emergency_debug_off.csv"
    on_debug = out_dir / "emergency_debug_on.csv"

    for path in [off_metrics, on_metrics]:
        path.mkdir(parents=True, exist_ok=True)

    base_args = [str(PYTHON), str(EMERGENCY_SCRIPT), "--config", str(EMERGENCY_CONFIG), "--pinn", str(PINN_CHECKPOINT), "--duration", str(duration), "--seed", str(seed), "--metrics-output-dir"]

    run_command(base_args + [str(off_metrics), "--run-name", "corridor_off", "--enable-corridor", "0", "--debug-log-path", str(off_debug)])
    run_command(base_args + [str(on_metrics), "--run-name", "corridor_on", "--enable-corridor", "1", "--debug-log-path", str(on_debug)])

    off_df = pd.read_csv(off_debug) if off_debug.exists() else pd.DataFrame()
    on_df = pd.read_csv(on_debug) if on_debug.exists() else pd.DataFrame()
    if not off_df.empty:
        off_df["scenario"] = "corridor_off"
    if not on_df.empty:
        on_df["scenario"] = "corridor_on"
    combined_debug = pd.concat([off_df, on_df], ignore_index=True) if not off_df.empty or not on_df.empty else pd.DataFrame()
    if not combined_debug.empty:
        combined_debug.to_csv(out_dir / "emergency_debug_log.csv", index=False)

    off_summary = summarize_emergency_debug(off_df)
    on_summary = summarize_emergency_debug(on_df)

    comparison_rows = [
        {"scenario": "corridor_off", **off_summary},
        {"scenario": "corridor_on", **on_summary},
        {
            "scenario": "improvement_pct",
            "travel_time": improvement_pct(off_summary["travel_time"], on_summary["travel_time"], reverse=False),
            "average_speed": improvement_pct(off_summary["average_speed"], on_summary["average_speed"], reverse=True),
            "complete_stops": improvement_pct(off_summary["complete_stops"], on_summary["complete_stops"], reverse=False),
            "waiting_time": improvement_pct(off_summary["waiting_time"], on_summary["waiting_time"], reverse=False),
            "intersections_crossed": improvement_pct(off_summary["intersections_crossed"], on_summary["intersections_crossed"], reverse=False),
            "traffic_lights_changed": improvement_pct(off_summary["traffic_lights_changed"], on_summary["traffic_lights_changed"], reverse=False),
            "vehicles_slowed": improvement_pct(off_summary["vehicles_slowed"], on_summary["vehicles_slowed"], reverse=False),
            "vehicles_rerouted": improvement_pct(off_summary["vehicles_rerouted"], on_summary["vehicles_rerouted"], reverse=False),
        },
    ]
    write_csv(out_dir / "emergency_corridor_comparison.csv", comparison_rows, comparison_rows[0].keys())

    return {"off": off_summary, "on": on_summary}


def summarize_aco_summary(summary: Dict[str, object]) -> Dict[str, float]:
    return {
        "average_travel_time": float(summary.get("average_travel_time", 0.0)),
        "average_waiting_time": float(summary.get("average_waiting_time", 0.0)),
        "average_queue_length": float(summary.get("average_queue_length", 0.0)),
        "throughput": float(summary.get("throughput", 0.0)),
        "reroutes": float(summary.get("reroutes", 0.0)),
        "route_changes": float(summary.get("route_changes", 0.0)),
        "average_congested_edge_utilization": float(summary.get("average_congested_edge_utilization", 0.0)),
        "completed_vehicles": float(summary.get("completed_vehicles", 0.0)),
    }


def run_aco_experiment(seed: int, duration: int) -> Dict[str, Dict[str, float]]:
    out_dir = OUTPUT_ROOT / "aco"
    off_metrics = out_dir / "off_metrics"
    on_metrics = out_dir / "on_metrics"
    off_reroute_log = out_dir / "aco_reroute_off.csv"
    on_reroute_log = out_dir / "aco_reroute_on.csv"

    for path in [off_metrics, on_metrics]:
        path.mkdir(parents=True, exist_ok=True)

    base_args = [str(PYTHON), str(ACO_SCRIPT), "--config", "sumo_network/config.sumocfg", "--pinn", "pinn_models/trained_pinn.pth", "--duration", str(duration), "--seed", str(seed), "--metrics-output-dir"]

    run_command(base_args + [str(off_metrics), "--run-name", "aco_off", "--enable-aco", "0", "--enable-pinn", "1", "--reroute-log", str(off_reroute_log)])
    run_command(base_args + [str(on_metrics), "--run-name", "aco_on", "--enable-aco", "1", "--enable-pinn", "1", "--reroute-log", str(on_reroute_log)])

    off_summary = summarize_aco_summary(load_json(off_metrics / "aco_off_summary.json"))
    on_summary = summarize_aco_summary(load_json(on_metrics / "aco_on_summary.json"))

    reroute_rows = []
    for scenario, path in [("aco_off", off_reroute_log), ("aco_on", on_reroute_log)]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["scenario"] = scenario
        reroute_rows.append(df)
    if reroute_rows:
        pd.concat(reroute_rows, ignore_index=True).to_csv(out_dir / "aco_reroute_log.csv", index=False)
    else:
        write_csv(out_dir / "aco_reroute_log.csv", [], ["scenario", "vehicle_id", "original_route", "new_route", "time_rerouted", "reason"])

    comparison_rows = [
        {"scenario": "aco_off", **off_summary},
        {"scenario": "aco_on", **on_summary},
        {
            "scenario": "improvement_pct",
            "average_travel_time": improvement_pct(off_summary["average_travel_time"], on_summary["average_travel_time"], reverse=False),
            "average_waiting_time": improvement_pct(off_summary["average_waiting_time"], on_summary["average_waiting_time"], reverse=False),
            "average_queue_length": improvement_pct(off_summary["average_queue_length"], on_summary["average_queue_length"], reverse=False),
            "throughput": improvement_pct(off_summary["throughput"], on_summary["throughput"], reverse=True),
            "reroutes": improvement_pct(off_summary["reroutes"], on_summary["reroutes"], reverse=True),
            "route_changes": improvement_pct(off_summary["route_changes"], on_summary["route_changes"], reverse=True),
            "average_congested_edge_utilization": improvement_pct(off_summary["average_congested_edge_utilization"], on_summary["average_congested_edge_utilization"], reverse=False),
            "completed_vehicles": improvement_pct(off_summary["completed_vehicles"], on_summary["completed_vehicles"], reverse=True),
        },
    ]
    write_csv(out_dir / "aco_comparison.csv", comparison_rows, comparison_rows[0].keys())

    return {"off": off_summary, "on": on_summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled behavior experiments for emergency corridor and ACO")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=1200)
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    emergency = run_emergency_experiment(args.seed, args.duration)
    aco = run_aco_experiment(args.seed, args.duration)

    report = {
        "emergency": emergency,
        "aco": aco,
        "artifacts": {
            "emergency_corridor_comparison": str(OUTPUT_ROOT / "emergency" / "emergency_corridor_comparison.csv"),
            "emergency_debug_log": str(OUTPUT_ROOT / "emergency" / "emergency_debug_log.csv"),
            "aco_comparison": str(OUTPUT_ROOT / "aco" / "aco_comparison.csv"),
            "aco_reroute_log": str(OUTPUT_ROOT / "aco" / "aco_reroute_log.csv"),
        },
    }
    (OUTPUT_ROOT / "controlled_behavior_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
