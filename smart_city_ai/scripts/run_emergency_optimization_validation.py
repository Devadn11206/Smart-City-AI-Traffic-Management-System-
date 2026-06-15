import sys
import os
import json
from pathlib import Path

# Ensure project root is in sys.path
repo_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(repo_root))

from integration.hybrid_emergency_controller import HybridEmergencyController
from emergency_system.clearing_monitor import get_monitor

def run_validation():
    print("Starting Phase 2 & 3 Emergency Vehicle Optimization Validation...")
    config_file = str(repo_root / 'sumo_network' / 'config.sumocfg')
    
    duration = 100
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])
    
    # Enable stop audit
    controller = HybridEmergencyController(
        config_file=config_file,
        pinn_model_path=None, # PINN logic falls back to zeroed
        duration=duration,
        use_gui=False,
        enable_corridor=True,
        run_name="emergency_validation"
    )
    controller.stop_audit_enabled = True
    
    # Run simulation
    summary = controller.run()
    
    # Collect Metrics
    monitor_snapshot = get_monitor().snapshot()
    
    # Emergency completion stats
    completed_evs = 0
    total_travel_time = 0.0
    total_delay = 0.0
    total_stops = 0
    total_detected = 0
    corridor_successes = 0

    # Parse stop events to identify EV stops
    ev_stop_counts = {}
    for ev in controller.stop_events:
        vid = ev['vehicle_id']
        ev_stop_counts[vid] = ev_stop_counts.get(vid, 0) + 1
        total_stops += 1

    # Extract info from debug logs
    debug_log = repo_root / 'results' / 'emergency_metrics' / 'emergency_validation_debug_log.csv'
    vehicles_data = {}
    if debug_log.exists():
        import csv
        with open(debug_log, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row.get('vehicle_id')
                if not vid: continue
                if vid not in vehicles_data:
                    vehicles_data[vid] = []
                    total_detected += 1
                vehicles_data[vid].append(row)
                
    for vid, rows in vehicles_data.items():
        if not rows: continue
        # Assume completed if not in active vehicles at the end
        if vid not in controller.detector.active_vehicles:
            completed_evs += 1
            travel_time = int(rows[-1]['step']) - int(rows[0]['step'])
            total_travel_time += travel_time
            total_delay += float(rows[-1]['waiting_time'])
            
            # Corridor success criteria
            stops = ev_stop_counts.get(vid, 0)
            if stops == 0:
                corridor_successes += 1

    avg_travel_time = (total_travel_time / completed_evs) if completed_evs > 0 else 0.0
    avg_delay = (total_delay / completed_evs) if completed_evs > 0 else 0.0
    stop_rate = (total_stops / completed_evs) if completed_evs > 0 else 0.0
    corridor_success_rate = (corridor_successes / completed_evs) if completed_evs > 0 else 0.0

    lane_attempts = monitor_snapshot.get('lane_change_attempts', 0)
    lane_successes = monitor_snapshot.get('lane_change_successes', 0)
    failed_lanes = lane_attempts - lane_successes
    lane_success_rate = (lane_successes / lane_attempts) if lane_attempts > 0 else 0.0
    
    aco_reroutes = monitor_snapshot.get('emergency_reroutes', 0)
    vma = monitor_snapshot.get('vehicles_moved_aside', 0)
    cleared_vehicles = vma if isinstance(vma, int) else len(vma)

    green_corridor = controller.green_corridor
    activation_count = green_corridor.metrics_activation_count
    reserved_edges = green_corridor.metrics_reserved_edge_count

    report = {
        "emergency_detected": total_detected,
        "emergency_completed": completed_evs,
        "emergency_stop_count": total_stops,
        "emergency_stop_rate": stop_rate,
        "average_emergency_delay": avg_delay,
        "average_emergency_travel_time": avg_travel_time,
        "average_response_time": avg_travel_time,
        "corridor_activation_count": activation_count,
        "corridor_success_rate": corridor_success_rate,
        "reserved_edge_count": reserved_edges,
        "green_wave_success_rate": corridor_success_rate, # Synonymous for this logic
        "cleared_vehicle_count": cleared_vehicles,
        "successful_lane_changes": lane_successes,
        "failed_lane_changes": failed_lanes,
        "lane_change_success_rate": lane_success_rate,
        "rerouted_vehicle_count": aco_reroutes,
        "aco_reroutes_caused": aco_reroutes,
        "emergency_eta_improvement": 0.0  # Would require baseline comparison, set to 0.0 placeholder or estimate
    }

    out_file = repo_root / 'results' / 'emergency_optimization_report.json'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Validation complete. Results saved to {out_file}")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

if __name__ == '__main__':
    run_validation()
