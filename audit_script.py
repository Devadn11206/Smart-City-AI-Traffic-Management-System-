import os
import json
import pandas as pd
import glob

def find_files(root_dir, pattern):
    return glob.glob(os.path.join(root_dir, '**', pattern), recursive=True)

audit_report = {
    "stages": {
        "benchmark": [],
        "pinn": []
    },
    "issues": []
}

root = r"c:\Users\ddnan\intenrship_IIIT"

print("Scanning for benchmark results...")
benchmark_dirs = [
    os.path.join(root, "benchmark_research", "2x2_benchmark"),
    os.path.join(root, "benchmark_research", "4x4_benchmark"),
    os.path.join(root, "benchmark_research", "6x6_benchmark"),
    os.path.join(root, "benchmark_research", "6x6 bench mark") # Check if it's the same
]

# For each benchmark folder, find algorithms and their results
for b_dir in benchmark_dirs:
    if not os.path.exists(b_dir):
        continue
    results_dir = os.path.join(b_dir, "results")
    if os.path.exists(results_dir):
        # find csv files
        csvs = find_files(results_dir, "*.csv")
        jsons = find_files(results_dir, "*.json")
        audit_report["stages"]["benchmark"].append({
            "dir": b_dir,
            "csv_count": len(csvs),
            "json_count": len(jsons),
            "csvs": csvs[:10],
            "jsons": jsons[:10]
        })

print("Scanning for PINN results...")
pinn_dirs = [
    os.path.join(root, "dqn_pinn_package"),
    os.path.join(root, "ddqn_pinn_package"),
    os.path.join(root, "ppo_pinn_package"),
    os.path.join(root, "ddqn_with_pinn_for_10x10")
]

for p_dir in pinn_dirs:
    if not os.path.exists(p_dir):
        continue
    csvs = find_files(p_dir, "*.csv")
    jsons = find_files(p_dir, "*.json")
    audit_report["stages"]["pinn"].append({
        "dir": p_dir,
        "csv_count": len(csvs),
        "json_count": len(jsons),
        "csvs": csvs[:10],
        "jsons": jsons[:10]
    })

with open(os.path.join(root, "audit_summary.json"), "w") as f:
    json.dump(audit_report, f, indent=2)

print("Audit script finished.")
