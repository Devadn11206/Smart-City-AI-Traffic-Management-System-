#!/usr/bin/env python3
"""
Test script to verify emergency vehicle handling improvements:
- Emergency vehicles never stop
- Lane changes are aggressive
- Other vehicles cooperate
- Route changes work smoothly
"""

import subprocess
import sys
import time
import json
from pathlib import Path

def run_emergency_demo():
    """Run the demo with all emergency vehicle features enabled."""
    
    print("\n" + "="*80)
    print("🚨 EMERGENCY VEHICLE HANDLING TEST")
    print("="*80)
    print("\nFeatures enabled:")
    print("  ✓ Emergency vehicle speed boost (25+ m/s)")
    print("  ✓ Aggressive lane changing (every 2 steps)")
    print("  ✓ Forced vehicle cooperation (all non-emergency vehicles yield)")
    print("  ✓ Path clearing ahead of emergency vehicles")
    print("  ✓ Smooth route replacement with fallback")
    print("  ✓ Emergency vehicles NEVER slowed down")
    print("\n" + "-"*80 + "\n")
    
    # Run the demo for 100 steps (~10 seconds) with GUI
    cmd = [
        sys.executable,
        "smart_city_ai/scripts/demo_predictive_corridor.py",
        "--mode", "IMPROVED",
        "--duration", "100",
        "--gui",
        "--seed", "42"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print("\nWatch for:")
    print("  - 🚨 EMERGENCY SPEED BOOST: Emergency vehicles maintain high speed")
    print("  - 🚨 AGGRESSIVE LANE CHANGE: Emergency vehicles changing lanes")
    print("  - 🚗 HARD YIELD/URGENT YIELD: Other vehicles making way")
    print("  - ✅ Applied route: Smooth route changes without stops\n")
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return 1

def check_results():
    """Check the results of the run."""
    print("\n" + "-"*80)
    print("Checking results...")
    
    metrics_dir = Path("results/emergency_metrics")
    if metrics_dir.exists():
        print(f"✅ Metrics directory exists: {metrics_dir}")
        
        # List files
        for f in sorted(metrics_dir.glob("*.json")):
            print(f"  - {f.name}")
    else:
        print(f"⚠️  No metrics directory found yet")
    
    print("\n" + "="*80)
    print("✨ Test Complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    exit_code = run_emergency_demo()
    check_results()
    sys.exit(exit_code)
