import json
import sys

def read_metrics():
    try:
        with open('results/emergency_optimization_report.json', 'r') as f:
            data = json.load(f)
        print(f"emergency_detected: {data.get('emergency_detected', 0)}")
        print(f"emergency_completed: {data.get('emergency_completed', 0)}")
        print(f"emergency_stop_rate: {data.get('emergency_stop_rate', 0.0):.2f}")
        print(f"lane_change_success_rate: {data.get('lane_change_success_rate', 0.0):.2f}")
        print(f"aco_reroutes_caused: {data.get('aco_reroutes_caused', 0)}")
        print(f"average_emergency_delay: {data.get('average_emergency_delay', 0.0):.2f}")
        print(f"average_emergency_travel_time: {data.get('average_emergency_travel_time', 0.0):.2f}")
        print(f"emergency_eta_improvement: {data.get('emergency_eta_improvement', 0.0):.2f}")
    except Exception as e:
        print(f"Failed to read metrics: {e}")

if __name__ == '__main__':
    read_metrics()
