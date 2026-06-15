"""
Validation script for predictive green wave emergency corridor system.

Compares three modes:
1. OFF: No emergency corridor activation
2. CURRENT (Reactive): All TLS on full route activated immediately
3. IMPROVED (Predictive): Future TLS pre-activated in sequence

Metrics tracked:
- Emergency vehicle response time (time to reach destination)
- Corridor activation efficiency (TLS activated before vehicle arrival)
- Traffic clearing effectiveness (vehicles slowed/rerouted)
- General traffic impact (throughput, congestion on non-emergency routes)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import traci
except Exception:
    traci = None

from integration.hybrid_emergency_controller import HybridEmergencyController
from emergency_system.green_corridor_manager import GreenCorridorManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclass
class CorridorMetrics:
    """Metrics for a single emergency corridor activation."""
    mode: str  # 'OFF', 'CURRENT', 'IMPROVED'
    vehicle_id: str
    activation_step: int
    completion_step: int
    total_steps: int
    distance_traveled: float
    avg_speed: float
    max_speed: float
    waiting_time: float
    vehicles_slowed: int
    vehicles_rerouted: int
    tls_activated: int
    tls_pre_activated: int  # For predictive mode
    route_length: int
    actual_route_length: int
    
    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class PredictiveCorridorValidator:
    """Validator for predictive emergency corridor system."""
    
    def __init__(self, config_file: str, duration: int = 2000, results_dir: str = 'results/validation'):
        self.config_file = config_file
        self.duration = duration
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: List[Dict[str, object]] = []
        self.mode_results: Dict[str, List[CorridorMetrics]] = {'OFF': [], 'CURRENT': [], 'IMPROVED': []}
    
    def run_mode(self, mode: str, seed: Optional[int] = None, hospital_edges: Optional[List[str]] = None) -> Dict[str, object]:
        """
        Run emergency controller in specified mode.
        
        Args:
            mode: 'OFF' (disable corridor), 'CURRENT' (reactive), 'IMPROVED' (predictive)
            seed: Random seed for reproducibility
            hospital_edges: Hospital edge list for routing
        
        Returns:
            Summary metrics from the run
        """
        logger.info(f"Starting run with mode={mode}, seed={seed}")
        
        enable_corridor = mode != 'OFF'
        run_name = f'validation_{mode}_{seed}'
        metrics_dir = self.results_dir / 'emergency_metrics' / mode
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        controller = HybridEmergencyController(
            config_file=self.config_file,
            pinn_model_path=None,  # Can be enabled if PINN model available
            duration=self.duration,
            use_gui=False,
            lookahead_steps=10,
            reroute_interval=30,
            decision_interval=5,
            pinn_interval=30,
            emergency_hold_steps=20,
            hospital_edges=hospital_edges or [],
            metrics_output_dir=str(metrics_dir),
            run_name=run_name,
            enable_corridor=enable_corridor,
            seed=seed,
        )
        
        summary = controller.run()
        summary['mode'] = mode
        summary['seed'] = seed
        return summary
    
    def extract_corridor_metrics(self, debug_log_csv: Path, mode: str) -> List[CorridorMetrics]:
        """
        Extract corridor-specific metrics from debug log.
        
        Args:
            debug_log_csv: Path to debug log CSV
            mode: Mode identifier
        
        Returns:
            List of CorridorMetrics for each emergency vehicle
        """
        metrics_list: List[CorridorMetrics] = []
        
        if not debug_log_csv.exists():
            logger.warning(f"Debug log not found: {debug_log_csv}")
            return metrics_list
        
        try:
            with debug_log_csv.open('r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if not rows:
                return metrics_list
            
            # Group by vehicle_id
            vehicles: Dict[str, List[Dict]] = {}
            for row in rows:
                vid = row.get('vehicle_id', 'unknown')
                if vid not in vehicles:
                    vehicles[vid] = []
                vehicles[vid].append(row)
            
            # Extract metrics per vehicle
            for vehicle_id, vehicle_rows in vehicles.items():
                if not vehicle_rows:
                    continue
                
                first_row = vehicle_rows[0]
                last_row = vehicle_rows[-1]
                
                # Calculate aggregate metrics
                total_waiting = sum(float(r.get('waiting_time', 0)) for r in vehicle_rows)
                total_slowed = sum(int(r.get('vehicles_cleared', 0)) for r in vehicle_rows)
                total_rerouted = sum(int(r.get('vehicles_rerouted', 0)) for r in vehicle_rows)
                total_tls = int(first_row.get('active_corridor_intersections', 0))
                predictive_tls = int(first_row.get('predictive_corridor_tls', 0))
                
                start_step = int(first_row.get('step', 0))
                end_step = int(last_row.get('step', 0))
                total_steps = max(1, end_step - start_step)
                
                metrics = CorridorMetrics(
                    mode=mode,
                    vehicle_id=vehicle_id,
                    activation_step=start_step,
                    completion_step=end_step,
                    total_steps=total_steps,
                    distance_traveled=0.0,  # Would need position tracking
                    avg_speed=0.0,  # Would need detailed speed data
                    max_speed=0.0,
                    waiting_time=total_waiting,
                    vehicles_slowed=total_slowed,
                    vehicles_rerouted=total_rerouted,
                    tls_activated=total_tls,
                    tls_pre_activated=predictive_tls,
                    route_length=int(first_row.get('signalized_intersections_on_route', 0)),
                    actual_route_length=0,
                )
                metrics_list.append(metrics)
                self.mode_results[mode].append(metrics)
        
        except Exception as e:
            logger.exception(f"Failed to extract metrics from {debug_log_csv}: {e}")
        
        return metrics_list
    
    def generate_comparison_report(self) -> Dict[str, object]:
        """
        Generate comprehensive comparison report across modes.
        
        Returns:
            Comparison summary with statistics
        """
        report = {
            'timestamp': time.time(),
            'modes': {},
            'comparison': {},
        }
        
        # Per-mode statistics
        for mode, metrics_list in self.mode_results.items():
            if not metrics_list:
                report['modes'][mode] = {
                    'vehicle_count': 0,
                    'avg_response_time': 0.0,
                    'total_vehicles_cleared': 0,
                    'total_vehicles_rerouted': 0,
                    'avg_tls_activated': 0.0,
                }
                continue
            
            total_steps = sum(m.total_steps for m in metrics_list)
            total_cleared = sum(m.vehicles_slowed for m in metrics_list)
            total_rerouted = sum(m.vehicles_rerouted for m in metrics_list)
            avg_tls = sum(m.tls_activated for m in metrics_list) / len(metrics_list) if metrics_list else 0.0
            
            report['modes'][mode] = {
                'vehicle_count': len(metrics_list),
                'avg_response_time': total_steps / len(metrics_list) if metrics_list else 0.0,
                'total_vehicles_cleared': total_cleared,
                'total_vehicles_rerouted': total_rerouted,
                'avg_tls_activated': avg_tls,
                'avg_predictive_tls': sum(m.tls_pre_activated for m in metrics_list) / len(metrics_list) if metrics_list else 0.0,
            }
        
        # Comparisons: IMPROVED vs CURRENT, IMPROVED vs OFF
        if 'IMPROVED' in self.mode_results and 'CURRENT' in self.mode_results:
            improved_avg = report['modes']['IMPROVED']['avg_response_time']
            current_avg = report['modes']['CURRENT']['avg_response_time']
            if current_avg > 0:
                improvement_pct = (1.0 - improved_avg / current_avg) * 100.0
                report['comparison']['predictive_vs_reactive'] = {
                    'response_time_improvement_pct': improvement_pct,
                    'improved_avg_steps': improved_avg,
                    'current_avg_steps': current_avg,
                }
        
        if 'IMPROVED' in self.mode_results and 'OFF' in self.mode_results:
            improved_avg = report['modes']['IMPROVED']['avg_response_time']
            off_avg = report['modes']['OFF']['avg_response_time']
            if off_avg > 0:
                benefit_pct = (1.0 - improved_avg / off_avg) * 100.0
                report['comparison']['predictive_vs_off'] = {
                    'response_time_improvement_pct': benefit_pct,
                    'improved_avg_steps': improved_avg,
                    'off_avg_steps': off_avg,
                }
        
        return report
    
    def validate(self, modes: List[str] = None, num_seeds: int = 3, hospital_edges: Optional[List[str]] = None) -> Dict[str, object]:
        """
        Run validation across all modes with multiple seeds.
        
        Args:
            modes: Modes to test (default: ['OFF', 'CURRENT', 'IMPROVED'])
            num_seeds: Number of different seeds to test
            hospital_edges: Hospital edges for routing
        
        Returns:
            Full validation report
        """
        if modes is None:
            modes = ['OFF', 'CURRENT', 'IMPROVED']
        
        logger.info(f"Starting validation: modes={modes}, num_seeds={num_seeds}")
        
        all_summaries = []
        
        for mode in modes:
            logger.info(f"Validating mode: {mode}")
            for seed in range(num_seeds):
                try:
                    summary = self.run_mode(mode, seed=seed, hospital_edges=hospital_edges)
                    summary['run_index'] = seed
                    all_summaries.append(summary)
                    
                    # Extract detailed corridor metrics
                    metrics_dir = self.results_dir / 'emergency_metrics' / mode
                    debug_log = metrics_dir / f'validation_{mode}_{seed}_debug_log.csv'
                    self.extract_corridor_metrics(debug_log, mode)
                    
                    logger.info(f"Completed {mode} seed {seed}")
                except Exception as e:
                    logger.exception(f"Failed to run {mode} seed {seed}: {e}")
        
        # Generate comparison report
        comparison = self.generate_comparison_report()
        
        # Save results
        results = {
            'timestamp': time.time(),
            'duration': self.duration,
            'modes_tested': modes,
            'num_seeds': num_seeds,
            'run_summaries': all_summaries,
            'comparison': comparison,
            'detailed_metrics': {mode: [asdict(m) for m in metrics] for mode, metrics in self.mode_results.items()},
        }
        
        results_file = self.results_dir / 'validation_report.json'
        results_file.write_text(json.dumps(results, indent=2), encoding='utf-8')
        logger.info(f"Validation complete. Report saved to {results_file}")
        
        return results


def main():
    """Run predictive corridor validation."""
    repo_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    config_file = str(repo_root / 'sumo_network' / 'network.sumocfg')
    
    validator = PredictiveCorridorValidator(
        config_file=config_file,
        duration=2000,
        results_dir='results/validation'
    )
    
    # Run validation with 1 seed per mode (can increase for more robust results)
    results = validator.validate(
        modes=['OFF', 'CURRENT', 'IMPROVED'],
        num_seeds=1,
        hospital_edges=None,
    )
    
    # Print summary
    print("\n" + "="*60)
    print("PREDICTIVE CORRIDOR VALIDATION SUMMARY")
    print("="*60)
    for mode, stats in results['comparison'].get('', {}).items():
        print(f"\n{mode}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    if 'comparison' in results and results['comparison']:
        print("\nCOMPARISON RESULTS:")
        if 'predictive_vs_reactive' in results['comparison']:
            pct = results['comparison']['predictive_vs_reactive'].get('response_time_improvement_pct', 0)
            print(f"  Predictive vs Reactive: {pct:.1f}% improvement")
        if 'predictive_vs_off' in results['comparison']:
            pct = results['comparison']['predictive_vs_off'].get('response_time_improvement_pct', 0)
            print(f"  Predictive vs Off: {pct:.1f}% improvement")


if __name__ == '__main__':
    main()
