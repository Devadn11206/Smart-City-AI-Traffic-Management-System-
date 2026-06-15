import os
import sys
import subprocess
import json
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent

def run_subprocess(name, pinn_enabled):
    print(f"\nRunning {name}...")
    cmd = [
        sys.executable,
        str(repo_root / "integration" / "rl_pinn_aco_integration.py"),
        "--enable-pinn", str(1 if pinn_enabled else 0),
        "--enable-aco", "1",
        "--duration", "1000",
        "--seed", "123",
        "--run-name", name
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=True)
    
    # Read the summary
    summary_path = repo_root / "results" / "demo_metrics" / name / f"{name}_summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            return json.load(f)
    return {}

def run_test():
    print("--- STARTING PINN VS NO-PINN COMPARISON TEST ---")
    
    try:
        summary_no_pinn = run_subprocess("no_pinn_test", False)
        summary_with_pinn = run_subprocess("with_pinn_test", True)
        
        print("\n--- RESULTS ---")
        print(f"{'Metric':<25} | {'NO PINN':<15} | {'WITH PINN':<15}")
        print("-" * 60)
        
        keys = ["throughput", "wait", "queue", "avg_speed", "emergency_vehicle_count", "eta_seconds", "corridor_efficiency"]
        for k in keys:
            val1 = summary_no_pinn.get(k, 0)
            val2 = summary_with_pinn.get(k, 0)
            if isinstance(val1, float):
                print(f"{k:<25} | {val1:<15.2f} | {val2:<15.2f}")
            else:
                print(f"{k:<25} | {val1:<15} | {val2:<15}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        
    print("\nTest completed.")

if __name__ == "__main__":
    run_test()
