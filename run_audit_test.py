#!/usr/bin/env python3
"""
Direct audit test - runs emergency scenario with DEBUG logging.
"""
import logging
import sys
import os

# Create output directory first
os.makedirs('results/emergency_audit_run', exist_ok=True)

# Setup verbose logging BEFORE any other imports
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('results/emergency_audit_run/audit_trace.log', mode='w')
    ]
)

# Now import and run
if __name__ == '__main__':
    from smart_city_ai.integration.hybrid_emergency_controller import HybridEmergencyController
    from smart_city_ai.emergency_system.clearing_monitor import get_monitor
    
    # Change to workspace directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create controller
    print("Creating HybridEmergencyController...")
    controller = HybridEmergencyController(
        config_file='smart_city_ai/sumo_network/config.sumocfg',
        duration=1500,  # 15 seconds
        use_gui=False,
        metrics_output_dir='results/emergency_audit_run',
        run_name='audit_test',
    )
    
    print("Running emergency scenario with audit logging...")
    summary = controller.run()
    
    print("\nFinal metrics from summary:")
    print(summary)
    
    monitor = get_monitor()
    snap = monitor.snapshot()
    print(f"\nMonitor snapshot:")
    print(f"  lane_change_attempts: {snap.get('lane_change_attempts', 0)}")
    print(f"  lane_change_successes: {snap.get('lane_change_successes', 0)}")
    print(f"  unique_attempted: {snap.get('unique_lane_change_attempted', 0)}")
    print(f"  unique_succeeded: {snap.get('unique_lane_change_successes', 0)}")
