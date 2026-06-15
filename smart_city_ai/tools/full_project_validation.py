#!/usr/bin/env python3
"""Full runtime validation harness for the Smart City AI project."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALIDATION_DIR = ROOT / "results" / "validation"
LOG_DIR = VALIDATION_DIR / "logs"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def build_command_sequence(include_benchmark: bool) -> list[dict[str, object]]:
    commands = [
        {
            "name": "Smoke Test - Agents",
            "cmd": [sys.executable, str(ROOT / "tools" / "smoke_test_agents.py")],
        },
        {
            "name": "Smoke Test - PINN",
            "cmd": [sys.executable, str(ROOT / "tools" / "smoke_test_pinn.py")],
        },
        {
            "name": "Smoke Test - ACO",
            "cmd": [sys.executable, str(ROOT / "tools" / "smoke_test_aco.py")],
        },
        {
            "name": "Smoke Test - SUMO Integration",
            "cmd": [sys.executable, str(ROOT / "tools" / "smoke_test_integration.py")],
        },
        {
            "name": "Final Evaluation - Hybrid Models",
            "cmd": [sys.executable, str(ROOT / "tools" / "final_evaluation.py"), "--steps", "2000", "--seeds", "101,202,303"],
        },
        {
            "name": "ACO Comparison - RL vs PINN vs ACO",
            "cmd": [sys.executable, str(ROOT / "evaluation" / "compare_all_algorithms.py"), "--duration", "900"],
        },
        {
            "name": "Predictive Corridor Comparison",
            "cmd": [sys.executable, str(ROOT / "scripts" / "demo_predictive_corridor.py"), "--compare", "--duration", "2000", "--seed", "42"],
        },
        {
            "name": "Production Readiness Audit",
            "cmd": [sys.executable, str(ROOT / "tools" / "production_readiness_report.py")],
        },
    ]

    if include_benchmark:
        commands.append(
            {
                "name": "Long-Horizon Reproducibility Benchmark",
                "cmd": [sys.executable, str(ROOT / "tools" / "run_full_repro_benchmark.py")],
            }
        )

    return commands


def normalize_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def run_command(command: dict[str, object]) -> dict[str, object]:
    name = command["name"]
    cmd = command["cmd"]
    safe_name = normalize_name(name)
    log_path = LOG_DIR / f"{safe_name}.log"
    start = time.time()
    print(f"\n=== RUNNING: {name} ===")
    print(" ".join(str(item) for item in cmd))

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        duration = time.time() - start
        success = process.returncode == 0
        print(f"=== {name} {'PASSED' if success else 'FAILED'} (exit {process.returncode}) in {duration:.1f}s ===")
        return {
            "name": name,
            "cmd": [str(c) for c in cmd],
            "returncode": process.returncode,
            "success": success,
            "log": str(log_path.relative_to(ROOT)),
            "duration_seconds": round(duration, 2),
        }
    except Exception as exc:
        duration = time.time() - start
        print(f"=== {name} EXCEPTION after {duration:.1f}s ===")
        return {
            "name": name,
            "cmd": [str(c) for c in cmd],
            "returncode": -1,
            "success": False,
            "log": str(log_path.relative_to(ROOT)),
            "duration_seconds": round(duration, 2),
            "error": str(exc),
        }


def write_summary(results: list[dict[str, object]]) -> None:
    summary_path = VALIDATION_DIR / "full_validation_summary.json"
    markdown_path = VALIDATION_DIR / "full_validation_summary.md"
    summary = {
        "timestamp": time.time(),
        "root": str(ROOT),
        "results": results,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = ["# Full Project Validation Summary", "", f"Generated: {time.ctime()}", "", "## Steps", ""]
    for item in results:
        status = "✅ PASS" if item.get("success") else "❌ FAIL"
        lines.append(f"- **{item['name']}**: {status} (exit {item.get('returncode')})")
        lines.append(f"  - `log`: `{item['log']}`")
        lines.append(f"  - `duration`: {item['duration_seconds']}s")
        if item.get("error"):
            lines.append(f"  - `error`: {item['error']}")
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"\nSummary written to {summary_path.relative_to(ROOT)}")
    print(f"Markdown report written to {markdown_path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full runtime validation for the Smart City AI repository.")
    parser.add_argument(
        "--include-benchmark",
        action="store_true",
        help="Include the long-horizon reproducibility benchmark (may take longer).",
    )
    parser.add_argument(
        "--skip-predictive-corridor",
        action="store_true",
        help="Skip the predictive green wave comparison step.",
    )
    parser.add_argument(
        "--skip-aco-comparison",
        action="store_true",
        help="Skip the ACO comparison runner.",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip the final production readiness audit report.",
    )
    args = parser.parse_args()

    commands = build_command_sequence(include_benchmark=args.include_benchmark)
    if args.skip_predictive_corridor:
        commands = [c for c in commands if c["name"] != "Predictive Corridor Comparison"]
    if args.skip_aco_comparison:
        commands = [c for c in commands if c["name"] != "ACO Comparison - RL vs PINN vs ACO"]
    if args.skip_audit:
        commands = [c for c in commands if c["name"] != "Production Readiness Audit"]

    results = []
    for command in commands:
        results.append(run_command(command))

    write_summary(results)
    failed = [item for item in results if not item.get("success")]
    if failed:
        print(f"\n{len(failed)} step(s) failed. See logs in {LOG_DIR.relative_to(ROOT)}")
        return 1
    print("\nAll validation steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
