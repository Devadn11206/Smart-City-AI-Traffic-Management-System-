#!/usr/bin/env python3
"""Debug emergency vehicle flow, TLS phases, and corridor activation timing."""
import os
import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import traci
except Exception:
    traci = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from environment.sumo_environment import SumoEnvironment
from emergency_system.green_corridor_manager import GreenCorridorManager
from emergency_system.emergency_detector import EmergencyVehicleDetector
from emergency_system.emergency_priority import EmergencyPriorityManager
from utils.network_utils import build_adjacency_and_edgeinfo


class DebugFlowMonitor:
    def __init__(self, config_file: str, duration: int = 50):
        self.config_file = config_file
        self.duration = duration
        self.env = SumoEnvironment(config_file, use_gui=True, num_seconds=duration)
        repo_root = Path(__file__).resolve().parent.parent.parent
        self.adjacency, self.edge_info = build_adjacency_and_edgeinfo(
            str(repo_root / 'smart_city_ai' / 'sumo_network' / 'network.net.xml')
        )
        self.priority_manager = EmergencyPriorityManager()
        self.detector = EmergencyVehicleDetector(priority_manager=self.priority_manager)
        self.green_corridor = GreenCorridorManager(use_gui=True)
        
        # Track state
        self.emergency_vehicles = {}
        self.tls_state_log = defaultdict(list)
        self.vehicle_position_log = defaultdict(list)
        self.corridor_activation_log = []
        
    def run(self):
        self.env.start_simulation(seed=42)
        
        try:
            for sim_step in range(self.duration):
                traci.simulationStep()
                
                # Detect emergencies
                detections = self.detector.scan(sim_step)
                
                for vehicle_id, obs in detections.items():
                    if vehicle_id not in self.emergency_vehicles:
                        self.emergency_vehicles[vehicle_id] = {
                            'detected_at': sim_step,
                            'route': list(obs.route),
                        }
                        logger.info(f"[{sim_step}] NEW EMERGENCY: {vehicle_id} on edge {obs.road_id}, route={obs.route[:3]}...")
                    
                    # Log vehicle position
                    try:
                        x, y = traci.vehicle.getPosition(vehicle_id)
                        speed = traci.vehicle.getSpeed(vehicle_id)
                        self.vehicle_position_log[vehicle_id].append({
                            'step': sim_step,
                            'edge': obs.road_id,
                            'speed': speed,
                            'x': x, 'y': y,
                        })
                        
                        # Check if stopped at intersection
                        if speed < 0.1 and obs.road_id and ':' not in obs.road_id:
                            logger.warning(f"[{sim_step}] {vehicle_id} STOPPED on edge {obs.road_id} (speed={speed:.2f})")
                    except Exception:
                        pass
                    
                    # Activate corridor
                    route = list(obs.route)
                    if route:
                        try:
                            corridor = self.green_corridor.activate_predictive(
                                vehicle_id, route, sim_step=sim_step,
                                hold_steps=20, priority=obs.priority,
                                future_tls_sequence=self.green_corridor._route_tls_candidates(route[:5])
                            )
                            self.corridor_activation_log.append({
                                'step': sim_step,
                                'vehicle': vehicle_id,
                                'tls': corridor.tls_ids,
                            })
                            logger.info(f"[{sim_step}] Corridor active for {vehicle_id}: TLS={corridor.tls_ids}")
                        except Exception as e:
                            logger.exception(f"Failed to activate corridor for {vehicle_id}: {e}")
                
                # Sample TLS phases every 5 steps
                if sim_step % 5 == 0 and detections:
                    self._log_tls_state(sim_step, detections)
        
        except Exception as e:
            logger.exception(f"Simulation error: {e}")
        
        finally:
            self.env.close()
            self._generate_report()
    
    def _log_tls_state(self, sim_step: int, detections: dict):
        """Log current TLS phases."""
        try:
            tls_ids = traci.trafficlight.getIDList()
            for tls_id in tls_ids[:5]:  # Sample first 5
                phase = traci.trafficlight.getPhase(tls_id)
                state = traci.trafficlight.getRedYellowGreenState(tls_id)
                self.tls_state_log[tls_id].append({
                    'step': sim_step,
                    'phase': phase,
                    'state': state,
                })
        except Exception as e:
            logger.warning(f"Failed to read TLS state: {e}")
    
    def _generate_report(self):
        """Generate debug report."""
        report = {
            'emergency_vehicles': len(self.emergency_vehicles),
            'vehicle_details': {},
            'tls_changes': {},
            'corridor_activations': self.corridor_activation_log,
            'issues': [],
        }
        
        # Analyze each emergency vehicle
        for vid, data in self.emergency_vehicles.items():
            positions = self.vehicle_position_log.get(vid, [])
            if positions:
                # Count stops
                stops = sum(1 for p in positions if p['speed'] < 0.1)
                report['vehicle_details'][vid] = {
                    'detected_at': data['detected_at'],
                    'total_steps': len(positions),
                    'stop_count': stops,
                    'avg_speed': sum(p['speed'] for p in positions) / len(positions) if positions else 0,
                    'edges_traversed': list(set(p['edge'] for p in positions)),
                }
                
                if stops > len(positions) * 0.5:
                    report['issues'].append(f"{vid}: Stopped >50% of time ({stops}/{len(positions)})")
        
        # Count TLS phase changes
        for tls_id, states in self.tls_state_log.items():
            if states:
                changes = sum(1 for i in range(1, len(states)) if states[i]['phase'] != states[i-1]['phase'])
                report['tls_changes'][tls_id] = changes
        
        report_path = Path(__file__).resolve().parent.parent.parent / 'results' / 'emergency_metrics' / 'debug_flow_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        
        logger.info(f"\nDebug report saved to {report_path}")
        logger.info(f"Issues found: {len(report['issues'])}")
        for issue in report['issues']:
            logger.warning(f"  - {issue}")


if __name__ == '__main__':
    config_file = str(Path(__file__).resolve().parent.parent / 'sumo_network' / 'config.sumocfg')
    monitor = DebugFlowMonitor(config_file, duration=80)
    monitor.run()
