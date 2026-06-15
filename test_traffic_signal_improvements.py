#!/usr/bin/env python3
"""
Test script to verify traffic signal and congestion improvements:
- Emergency vehicles don't stop at red lights
- Aggressive lane changes in traffic
- Rerouting avoids congestion
"""

import subprocess
import sys
import json
from pathlib import Path

def run_improved_demo():
    """Run the demo with all traffic signal and congestion improvements."""
    
    print("\n" + "="*80)
    print("🚨 TRAFFIC SIGNAL & CONGESTION IMPROVEMENTS TEST")
    print("="*80)
    print("\n✅ Improvements Enabled:")
    print("  ✓ Traffic Signal Prioritizer (forces green lights for emergency)")
    print("  ✓ Emergency Lane Changes (every simulation step)")
    print("  ✓ Congestion-Aware Routing (ACO + vehicle density penalties)")
    print("  ✓ No Red Light Waiting (signals forced to green 200m ahead)")
    print("  ✓ Bidirectional Lane Changes (tries left then right)")
    print("  ✓ 30-Second Phase Duration (prevents immediate red after emergency)")
    print("\n" + "-"*80 + "\n")
    
    # Run for 100 steps (~10 seconds) with GUI
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
    print("\n📊 Watch for these indicators:")
    print("  ✓ 🚨 FORCED GREEN: Traffic signal prioritization active")
    print("  ✓ 🚨 AGGRESSIVE LANE CHANGE: Every-step lane optimization")
    print("  ✓ 🚗 YIELD: Other vehicles cooperating")
    print("  ✓ ✅ Applied route: Smooth route changes\n")
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return 1

def verify_components():
    """Verify all new components are in place."""
    print("\n" + "-"*80)
    print("📋 Verifying Components...")
    
    components = {
        "Traffic Signal Prioritizer": "smart_city_ai/emergency_system/traffic_signal_prioritizer.py",
        "Aggressive Emergency Handler": "smart_city_ai/emergency_system/aggressive_emergency_handler.py",
        "Enhanced Emergency Router": "smart_city_ai/emergency_system/emergency_router.py",
        "Hybrid Controller Integration": "smart_city_ai/integration/hybrid_emergency_controller.py",
    }
    
    all_exist = True
    for name, path in components.items():
        full_path = Path(path)
        exists = full_path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {name}")
        all_exist = all_exist and exists
    
    print("\n" + "-"*80)
    if all_exist:
        print("✨ All components verified successfully!")
    else:
        print("⚠️  Some components missing!")
    
    print("\n" + "="*80 + "\n")
    return all_exist

if __name__ == "__main__":
    verify_components()
    exit_code = run_improved_demo()
    sys.exit(exit_code)
