#!/usr/bin/env python3
"""Generate a compact production-readiness report from the latest artifacts."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from utils.telemetry import summarize_telemetry


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_first_json(*paths: Path):
    for path in paths:
        payload = load_json(path)
        if payload is not None:
            return payload
    return None


def find_latest_metrics() -> dict:
    report = {
        "benchmark_report": load_first_json(
            repo_root / "results" / "aco_results" / "reproducibility_benchmark_report.json",
            repo_root.parent / "results" / "aco_results" / "reproducibility_benchmark_report.json",
        ),
        "checkpoint_health": load_first_json(
            repo_root / "results" / "checkpoints" / "checkpoint_health_report.json",
            repo_root.parent / "results" / "checkpoints" / "checkpoint_health_report.json",
        ),
        "stress_test": load_first_json(
            repo_root / "results" / "stress_tests" / "stress_test_summary.json",
            repo_root.parent / "results" / "stress_tests" / "stress_test_summary.json",
        ),
        "performance_profile": load_first_json(
            repo_root / "results" / "performance" / "performance_profile.json",
            repo_root.parent / "results" / "performance" / "performance_profile.json",
        ),
    }
    return report


def build_report() -> str:
    payload = find_latest_metrics()
    telemetry_summary = summarize_telemetry(repo_root / "results" / "telemetry")
    if telemetry_summary.get("files_scanned", 0) == 0:
        telemetry_summary = summarize_telemetry(repo_root.parent / "results" / "telemetry")

    lines = ["# Production Readiness Report", ""]
    lines.append("## Checkpoint Hardening")
    checkpoint_health = payload.get("checkpoint_health") or {}
    if checkpoint_health:
        registry = checkpoint_health.get("registry", {})
        archived = checkpoint_health.get("archived_incompatible", [])
        lines.append(f"- Canonical checkpoints: {len(registry) if registry else 'not available'}")
        lines.append(f"- Active checkpoints verified: {sum(1 for item in registry.values() if item.get('compatible')) if registry else 'not available'}")
        lines.append(f"- Archived checkpoints: {len(archived) if archived else 'not available'}")
    else:
        lines.append("- Checkpoint health report not found yet.")

    lines.append("")
    lines.append("## Telemetry")
    lines.append(f"- Files scanned: {telemetry_summary.get('files_scanned', 0)}")
    lines.append(f"- Records: {telemetry_summary.get('records', 0)}")
    lines.append(f"- Scenario count: {telemetry_summary.get('scenarios', 0)}")
    lines.append(f"- Step count: {telemetry_summary.get('steps', 0)}")

    lines.append("")
    lines.append("## Stress Testing")
    stress = payload.get("stress_test") or {}
    if stress:
        for agent, result in stress.get("summary", {}).items():
            lines.append(f"- {agent.upper()}: episodes={result.get('episodes', 'not available')}, failures={result.get('failures', 'not available')}")
    else:
        lines.append("- Stress test summary not found yet.")

    lines.append("")
    lines.append("## Performance Profiling")
    profile = payload.get("performance_profile") or {}
    if profile:
        for agent, result in profile.items():
            lines.append(f"- {agent.upper()}: load={result.get('load_seconds', 'n/a')}s, env_step={result.get('mean_env_step_seconds', 'n/a')}s")
    else:
        lines.append("- Performance profile not found yet.")

    lines.append("")
    lines.append("## Benchmark Status")
    benchmark = payload.get("benchmark_report") or {}
    if benchmark:
        per_seed = benchmark.get("per_seed", {})
        run_count = sum(len(agent_runs) for agent_runs in per_seed.values())
        failed_count = sum(
            1
            for agent_runs in per_seed.values()
            for result in agent_runs.values()
            if isinstance(result, dict) and result.get("checkpoint_status") not in (None, "loaded")
        )
        lines.append(f"- Runs completed: {run_count}")
        lines.append(f"- Runs failed: {failed_count}")
        lines.append(f"- Report path: results/aco_results/reproducibility_benchmark_report.json")
    else:
        lines.append("- Reproducibility benchmark report not found yet.")

    lines.append("")
    lines.append("## Readiness Summary")
    lines.append("- Checkpoint loading now validates compatibility before use.")
    lines.append("- The benchmark runner emits structured telemetry.")
    lines.append("- Stress and profiling scripts are available for repeatable validation.")
    return "\n".join(lines)


def main() -> None:
    out_path = repo_root / "results" / "production_readiness_report.md"
    os.makedirs(out_path.parent, exist_ok=True)
    report = build_report()
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report)


if __name__ == "__main__":
    main()
