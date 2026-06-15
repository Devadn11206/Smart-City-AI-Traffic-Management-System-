#!/usr/bin/env python3
"""
PINN Ablation Study: Controlled A/B Experiment
Measures the impact of PINN on system performance while keeping all other components constant.

Experiment Design:
  A (Baseline): Without PINN - ACO + Emergency Corridor + Predictive GW (no predictions)
  B (Enhanced): With PINN - ACO + Emergency Corridor + Predictive GW (with predictions)

Same conditions for both:
  - Simulation duration: 100 seconds (1000 steps)
  - Traffic demand: Fixed vehicle generation
  - Random seed: Fixed for reproducibility
  - Emergency schedule: Same emergency vehicle triggers

Metrics collected:
  1. Average Waiting Time
  2. Average Queue Length
  3. Average Speed
  4. Throughput
  5. Emergency Travel Time
  6. Emergency Vehicle Stops
  7. ACO Reroutes
  8. Corridor Activations
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s'
)
logger = logging.getLogger(__name__)

workspace_root = Path(__file__).parent
sys.path.insert(0, str(workspace_root / 'smart_city_ai'))

EXPERIMENT_CONFIG = {
    'duration_seconds': 100,
    'num_steps': 1000,
    'step_length': 0.1,
    'random_seed': 42,
    'traffic_demand_multiplier': 1.0,
}

def run_experiment(enable_pinn: bool, experiment_name: str) -> dict:
    """
    Run a single experiment configuration.
    
    Args:
        enable_pinn: Whether to enable PINN predictions
        experiment_name: Name for logging
    
    Returns:
        Dictionary with collected metrics
    """
    logger.info("\n" + "="*80)
    logger.info(f"RUNNING EXPERIMENT: {experiment_name}")
    logger.info("="*80)
    logger.info(f"PINN Enabled: {enable_pinn}")
    logger.info(f"Duration: {EXPERIMENT_CONFIG['duration_seconds']} seconds")
    logger.info(f"Random Seed: {EXPERIMENT_CONFIG['random_seed']}")
    
    try:
        from integration.hybrid_emergency_controller import HybridEmergencyController
        
        # Locate SUMO configuration
        sumo_config = workspace_root / 'smart_city_ai' / 'sumo_network' / 'config.sumocfg'
        
        if not sumo_config.exists():
            logger.error(f"SUMO config not found: {sumo_config}")
            return None
        
        # Initialize controller with/without PINN
        pinn_model_path = None
        if enable_pinn:
            pinn_model_path = str(workspace_root / 'smart_city_ai' / 'pinn_models' / 'trained_pinn.pth')
            if not Path(pinn_model_path).exists():
                logger.warning(f"PINN model not found: {pinn_model_path}")
                pinn_model_path = None
        
        logger.info(f"Initializing Hybrid Emergency Controller...")
        logger.info(f"  - PINN: {'Enabled' if enable_pinn and pinn_model_path else 'Disabled'}")
        logger.info(f"  - ACO: Enabled")
        logger.info(f"  - Emergency Corridor: Enabled")
        logger.info(f"  - Traffic Clearing: Enabled")
        
        controller = HybridEmergencyController(
            config_file=str(sumo_config),
            pinn_model_path=pinn_model_path if enable_pinn else None,
            duration=EXPERIMENT_CONFIG['duration_seconds'],
            use_gui=False,
            seed=EXPERIMENT_CONFIG['random_seed'],
            metrics_output_dir=str(workspace_root / 'results' / f'ablation_experiment_{experiment_name}'),
            run_name=experiment_name,
        )
        
        # Metrics collection
        metrics = {
            'steps': [],
            'waiting_times': [],
            'queue_lengths': [],
            'speeds': [],
            'throughputs': [],
            'emergency_eta': [],
            'emergency_stops': [],
            'aco_reroutes': [],
            'corridor_activations': [],
            'pinn_predictions': [],
        }
        
        logger.info(f"\nRunning simulation...")
        start_time = time.time()
        
        # Run the full simulation
        result = controller.run()
        
        runtime_sec = time.time() - start_time
        
        # Extract metrics from the latest output
        latest_json = workspace_root / 'results' / f'ablation_experiment_{experiment_name}' / f'{experiment_name}_latest.json'
        metrics_file = workspace_root / 'results' / f'ablation_experiment_{experiment_name}' / f'{experiment_name}_metrics.csv'
        
        summary = {
            'experiment': experiment_name,
            'pinn_enabled': enable_pinn,
            'simulation_duration_sec': EXPERIMENT_CONFIG['duration_seconds'],
            'runtime_sec': runtime_sec,
        }
        
        # Try to load metrics from output files
        try:
            if latest_json.exists():
                with open(latest_json, 'r') as f:
                    latest_metrics = json.load(f)
                    summary.update(latest_metrics)
        except Exception as e:
            logger.warning(f"Failed to load latest metrics: {e}")
        
        logger.info(f"\n✅ Experiment completed: {experiment_name}")
        logger.info(f"Runtime: {runtime_sec:.1f} seconds")
        
        return summary
        
    except Exception as e:
        logger.error(f"✗ Experiment failed: {e}", exc_info=True)
        return None

def generate_comparison_report(result_a: dict, result_b: dict) -> dict:
    """Generate comparison report between two experiments."""
    
    report = {
        'experiment_a': result_a['experiment'],
        'experiment_b': result_b['experiment'],
        'timestamp': datetime.now().isoformat(),
        'comparison': {},
    }
    
    # Define metrics to compare
    metrics_to_compare = [
        ('avg_waiting_time', 'lower_is_better'),
        ('avg_queue_length', 'lower_is_better'),
        ('avg_speed', 'higher_is_better'),
        ('avg_throughput', 'higher_is_better'),
        ('avg_emergency_eta', 'lower_is_better'),
        ('total_emergency_stops', 'lower_is_better'),
        ('total_aco_reroutes', 'higher_is_better'),  # More reroutes = better adaptation
        ('avg_corridor_activations', 'higher_is_better'),
    ]
    
    for metric, direction in metrics_to_compare:
        if metric in result_a and metric in result_b:
            value_a = result_a[metric]
            value_b = result_b[metric]
            
            # Calculate difference
            if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
                abs_diff = value_b - value_a
                if value_a != 0:
                    pct_change = (abs_diff / abs(value_a)) * 100
                else:
                    pct_change = 0.0
                
                # Determine improvement direction
                if direction == 'lower_is_better':
                    improvement = abs_diff < 0  # Negative difference is good
                else:
                    improvement = abs_diff > 0  # Positive difference is good
                
                report['comparison'][metric] = {
                    'baseline': value_a,
                    'with_pinn': value_b,
                    'absolute_difference': abs_diff,
                    'percentage_change': pct_change,
                    'improved': improvement,
                    'direction': direction,
                }
    
    # Overall assessment
    improvements = sum(1 for m in report['comparison'].values() if m['improved'])
    total_metrics = len(report['comparison'])
    report['overall_assessment'] = {
        'metrics_improved': improvements,
        'total_metrics': total_metrics,
        'improvement_rate': (improvements / total_metrics * 100) if total_metrics > 0 else 0.0,
    }
    
    return report

def main():
    """Run the PINN ablation study."""
    logger.info("\n")
    logger.info("█" * 80)
    logger.info("█" + " " * 78 + "█")
    logger.info("█" + "  PINN ABLATION STUDY - Controlled A/B Experiment".center(78) + "█")
    logger.info("█" + " " * 78 + "█")
    logger.info("█" * 80)
    
    logger.info("\nExperiment Configuration:")
    logger.info(f"  Duration: {EXPERIMENT_CONFIG['duration_seconds']} seconds")
    logger.info(f"  Steps: {EXPERIMENT_CONFIG['num_steps']}")
    logger.info(f"  Random Seed: {EXPERIMENT_CONFIG['random_seed']}")
    logger.info(f"  Traffic Demand: {EXPERIMENT_CONFIG['traffic_demand_multiplier']}x")
    
    # Run Experiment A: Without PINN
    logger.info("\n" + "-" * 80)
    logger.info("EXPERIMENT A: WITHOUT PINN (Baseline)")
    logger.info("-" * 80)
    result_a = run_experiment(enable_pinn=False, experiment_name="Baseline (No PINN)")
    
    if result_a is None:
        logger.error("Experiment A failed")
        return False
    
    # Run Experiment B: With PINN
    logger.info("\n" + "-" * 80)
    logger.info("EXPERIMENT B: WITH PINN (Enhanced)")
    logger.info("-" * 80)
    result_b = run_experiment(enable_pinn=True, experiment_name="With PINN")
    
    if result_b is None:
        logger.error("Experiment B failed")
        return False
    
    # Generate comparison report
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING COMPARISON REPORT")
    logger.info("=" * 80)
    
    comparison_report = generate_comparison_report(result_a, result_b)
    
    # Save JSON report
    output_dir = workspace_root / 'results' / 'pinn_ablation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / 'pinn_ablation_report.json'
    with open(json_path, 'w') as f:
        json.dump({
            'metadata': {
                'experiment_type': 'PINN Ablation Study',
                'timestamp': datetime.now().isoformat(),
                'configuration': EXPERIMENT_CONFIG,
            },
            'experiment_a': result_a,
            'experiment_b': result_b,
            'comparison': comparison_report,
        }, f, indent=2)
    
    logger.info(f"✓ JSON report saved: {json_path}")
    
    # Create comparison table
    comparison_data = []
    for metric, details in comparison_report['comparison'].items():
        comparison_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'Baseline (No PINN)': f"{details['baseline']:.4f}",
            'With PINN': f"{details['with_pinn']:.4f}",
            'Absolute Difference': f"{details['absolute_difference']:.4f}",
            'Percentage Change': f"{details['percentage_change']:.2f}%",
            'Improved': '✓ Yes' if details['improved'] else '✗ No',
            'Direction': details['direction'].replace('_', ' ').title(),
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    csv_path = output_dir / 'pinn_comparison_table.csv'
    df_comparison.to_csv(csv_path, index=False)
    
    logger.info(f"✓ CSV table saved: {csv_path}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)
    
    print("\n" + df_comparison.to_string(index=False))
    
    logger.info("\n" + "=" * 80)
    logger.info("FINAL ASSESSMENT")
    logger.info("=" * 80)
    
    improvement_rate = comparison_report['overall_assessment']['improvement_rate']
    logger.info(f"\nMetrics Improved: {comparison_report['overall_assessment']['metrics_improved']}/{comparison_report['overall_assessment']['total_metrics']}")
    logger.info(f"Improvement Rate: {improvement_rate:.1f}%")
    
    # Key findings
    logger.info("\n" + "-" * 80)
    logger.info("KEY FINDINGS")
    logger.info("-" * 80)
    
    # Congestion management
    waiting_time_improved = comparison_report['comparison'].get('avg_waiting_time', {}).get('improved', False)
    queue_improved = comparison_report['comparison'].get('avg_queue_length', {}).get('improved', False)
    
    logger.info("\n✓ CONGESTION MANAGEMENT:")
    if waiting_time_improved or queue_improved:
        if waiting_time_improved:
            wt_diff = comparison_report['comparison']['avg_waiting_time']['percentage_change']
            logger.info(f"  ✓ Waiting time reduced by {abs(wt_diff):.2f}%")
        if queue_improved:
            q_diff = comparison_report['comparison']['avg_queue_length']['percentage_change']
            logger.info(f"  ✓ Queue length reduced by {abs(q_diff):.2f}%")
        logger.info("  → PINN improves congestion management ✓")
    else:
        logger.info("  ✗ No improvement in congestion metrics")
    
    # Route selection
    reroutes_improved = comparison_report['comparison'].get('total_aco_reroutes', {}).get('improved', False)
    logger.info("\n✓ ROUTE SELECTION:")
    if reroutes_improved:
        reroute_diff = comparison_report['comparison']['total_aco_reroutes']['percentage_change']
        logger.info(f"  ✓ ACO reroutes increased by {reroute_diff:.2f}%")
        logger.info("  → PINN enables more dynamic routing ✓")
    else:
        logger.info("  ≈ ACO reroutes remained similar")
    
    # Emergency response
    eta_improved = comparison_report['comparison'].get('avg_emergency_eta', {}).get('improved', False)
    stops_improved = comparison_report['comparison'].get('total_emergency_stops', {}).get('improved', False)
    corridors_improved = comparison_report['comparison'].get('avg_corridor_activations', {}).get('improved', False)
    
    logger.info("\n✓ EMERGENCY RESPONSE:")
    if eta_improved:
        eta_diff = comparison_report['comparison']['avg_emergency_eta']['percentage_change']
        logger.info(f"  ✓ Emergency ETA reduced by {abs(eta_diff):.2f}%")
    if stops_improved:
        stops_diff = comparison_report['comparison']['total_emergency_stops']['percentage_change']
        logger.info(f"  ✓ Emergency stops reduced by {abs(stops_diff):.2f}%")
    if corridors_improved:
        corr_diff = comparison_report['comparison']['avg_corridor_activations']['percentage_change']
        logger.info(f"  ✓ Corridor activations increased by {corr_diff:.2f}%")
    
    if eta_improved or corridors_improved:
        logger.info("  → PINN improves emergency response performance ✓")
    else:
        logger.info("  ≈ Emergency performance remained similar")
    
    logger.info("\n" + "=" * 80)
    logger.info("CONCLUSION")
    logger.info("=" * 80)
    
    total_improvements = comparison_report['overall_assessment']['metrics_improved']
    if total_improvements >= 5:
        logger.info("\n✅ PINN provides SIGNIFICANT performance improvements")
        logger.info(f"   {total_improvements}/8 metrics improved with PINN enabled")
    elif total_improvements >= 3:
        logger.info("\n✓ PINN provides MODERATE performance improvements")
        logger.info(f"   {total_improvements}/8 metrics improved with PINN enabled")
    else:
        logger.info("\n≈ PINN impact is LIMITED")
        logger.info(f"   Only {total_improvements}/8 metrics improved with PINN enabled")
    
    logger.info("\nGenerated Files:")
    logger.info(f"  - {json_path.name}")
    logger.info(f"  - {csv_path.name}")
    
    return True

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
