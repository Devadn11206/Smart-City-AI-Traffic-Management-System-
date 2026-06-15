import json
import threading
from pathlib import Path


class ClearingMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.lane_change_attempts = 0
        self.lane_change_successes = 0
        self._attempted_vehicles = set()
        self._successful_vehicles = set()
        self.vehicles_moved_aside = set()
        self.emergency_reroutes = 0
        self.clear_calls = 0
        self.cleared_edges_total = 0
        self.attempted_edges_total = 0
        self.per_call_stats = []
        self.emergency_events = []

    def record_lane_change_attempt(self, vehicle_id: str):
        with self.lock:
            self.lane_change_attempts += 1
            self._attempted_vehicles.add(vehicle_id)

    def record_lane_change_success(self, vehicle_id: str):
        with self.lock:
            self.lane_change_successes += 1
            self._successful_vehicles.add(vehicle_id)
            self.vehicles_moved_aside.add(vehicle_id)

    def record_vehicle_moved_aside(self, vehicle_id: str):
        with self.lock:
            self.vehicles_moved_aside.add(vehicle_id)

    def record_reroute(self):
        with self.lock:
            self.emergency_reroutes += 1

    def record_clear_call(self, attempted_edges: int, cleared_edges: int, slowed: int, rerouted: int):
        with self.lock:
            self.clear_calls += 1
            self.attempted_edges_total += attempted_edges
            self.cleared_edges_total += cleared_edges
            self.per_call_stats.append({
                'attempted_edges': attempted_edges,
                'cleared_edges': cleared_edges,
                'slowed': slowed,
                'rerouted': rerouted,
            })

    def record_emergency_event(self, event: dict):
        with self.lock:
            self.emergency_events.append(dict(event))

    def snapshot(self):
        with self.lock:
            return {
                'lane_change_attempts': self.lane_change_attempts,
                'lane_change_successes': self.lane_change_successes,
                'unique_lane_change_attempted': len(self._attempted_vehicles),
                'unique_lane_change_successes': len(self._successful_vehicles),
                'vehicles_moved_aside': len(self.vehicles_moved_aside),
                'emergency_reroutes': self.emergency_reroutes,
                'clear_calls': self.clear_calls,
                'attempted_edges_total': self.attempted_edges_total,
                'cleared_edges_total': self.cleared_edges_total,
                'corridor_clearance_rate': (self.cleared_edges_total / self.attempted_edges_total) if self.attempted_edges_total else 0.0,
                'per_call_stats': list(self.per_call_stats),
                'emergency_events': list(self.emergency_events),
            }

    def write_reports(self, out_dir: str = 'results/emergency_metrics'):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        lane_report = {
            'lane_change_attempts': self.lane_change_attempts,
            'lane_change_successes': self.lane_change_successes,
            'unique_lane_change_attempted': len(self._attempted_vehicles),
            'unique_lane_change_successes': len(self._successful_vehicles),
            'vehicles_moved_aside': list(self.vehicles_moved_aside),
        }
        yielding_report = self.snapshot()
        lane_report['events'] = list(self.emergency_events)

        with open(Path(out_dir) / 'emergency_lane_clearing_report.json', 'w') as f:
            json.dump(lane_report, f, indent=2)

        with open(Path(out_dir) / 'emergency_yielding_report.json', 'w') as f:
            json.dump(yielding_report, f, indent=2)

        optimization_report = {
            'lane_change_attempts': self.lane_change_attempts,
            'lane_change_successes': self.lane_change_successes,
            'lane_change_success_rate': (self.lane_change_successes / self.lane_change_attempts) if self.lane_change_attempts else 0.0,
            'unique_lane_change_attempted': len(self._attempted_vehicles),
            'unique_lane_change_successes': len(self._successful_vehicles),
            'vehicles_moved_aside': len(self.vehicles_moved_aside),
            'emergency_reroutes': self.emergency_reroutes,
            'clear_calls': self.clear_calls,
            'events': list(self.emergency_events),
        }
        with open(Path(out_dir) / 'lane_change_optimization_report.json', 'w') as f:
            json.dump(optimization_report, f, indent=2)

        corridor_report = {
            'clear_calls': self.clear_calls,
            'attempted_edges_total': self.attempted_edges_total,
            'cleared_edges_total': self.cleared_edges_total,
            'corridor_clearance_rate': (self.cleared_edges_total / self.attempted_edges_total) if self.attempted_edges_total else 0.0,
            'emergency_reroutes': self.emergency_reroutes,
            'vehicles_moved_aside': len(self.vehicles_moved_aside),
            'events': list(self.emergency_events),
        }
        with open(Path(out_dir) / 'emergency_corridor_effectiveness_report.json', 'w') as f:
            json.dump(corridor_report, f, indent=2)


_monitor = ClearingMonitor()


def get_monitor() -> ClearingMonitor:
    return _monitor
