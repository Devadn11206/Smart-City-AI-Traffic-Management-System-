"""
Example script demonstrating predictive green wave emergency corridor system.

This script showcases:
1. Emergency vehicle detection
2. Predictive corridor sequence building (2-5 TLS ahead)
3. Green wave phase synchronization
4. Enhanced traffic clearing
5. Metrics comparison vs reactive corridor

Usage:
  python demo_predictive_corridor.py [--mode IMPROVED|CURRENT|OFF] [--duration 2000]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from integration.hybrid_emergency_controller import HybridEmergencyController

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)


def run_predictive_corridor_demo(
    mode: str = 'IMPROVED',
    duration: int = 2000,
    seed: int = 42,
    use_gui: bool = False,
    pinn_model_path: Optional[str] = None,
):
    """
    Run demonstration of predictive green wave emergency corridor.
    
    Args:
        mode: Corridor mode ('IMPROVED', 'CURRENT', or 'OFF')
        duration: Simulation duration in steps
        seed: Random seed for reproducibility
        use_gui: Whether to use SUMO GUI
        pinn_model_path: Optional path to a trained PINN checkpoint
    """
    
    repo_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    config_file = str(repo_root / 'sumo_network' / 'config.sumocfg')
    
    logger.info(f"Starting {mode} mode demonstration")
    logger.info(f"Config: {config_file}")
    logger.info(f"Duration: {duration} steps, Seed: {seed}")
    if not use_gui:
        logger.warning('SUMO GUI disabled: screenshots and GUI-only visual confirmation will be skipped.')
    
    # Configure based on mode
    enable_corridor = mode != 'OFF'
    run_name = f'demo_{mode}'
    metrics_dir = repo_root / 'results' / 'demo_metrics' / mode
    
    # Create controller
    controller = HybridEmergencyController(
        config_file=config_file,
        pinn_model_path=pinn_model_path,
        duration=duration,
        use_gui=use_gui,
        lookahead_steps=10,
        reroute_interval=30,
        decision_interval=5,
        pinn_interval=30,
        emergency_hold_steps=20,
        hospital_edges=[],
        metrics_output_dir=str(metrics_dir),
        run_name=run_name,
        enable_corridor=enable_corridor,
        seed=seed,
        debug_log_path=str(metrics_dir / f'{run_name}_debug.csv'),
    )
    
    if pinn_model_path:
        logger.info(f'Using PINN model: {pinn_model_path}')
    else:
        logger.warning('No PINN model path provided; running without PINN predictions.')
    logger.info(f"Corridor mode: {'PREDICTIVE' if enable_corridor else 'DISABLED'}")
    
    # Run simulation
    start_time = time.time()
    summary = controller.run()
    elapsed = time.time() - start_time
    
    # Log results
    logger.info("="*70)
    logger.info("SIMULATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Elapsed time: {elapsed:.1f}s")
    logger.info(f"Simulation duration: {summary.get('duration', 'N/A')} steps")
    logger.info(f"Emergency vehicles: {summary.get('emergency_vehicle_count', 0)}")
    logger.info(f"Active corridors: {summary.get('active_corridors', 0)}")
    logger.info(f"Blocked edges: {summary.get('blocked_edges', 0)}")
    
    if 'last_cleared_slowed' in summary:
        logger.info(f"Total vehicles slowed: {summary['last_cleared_slowed']}")
    if 'last_cleared_rerouted' in summary:
        logger.info(f"Total vehicles rerouted: {summary['last_cleared_rerouted']}")
    
    logger.info("="*70)
    logger.info(f"Results saved to: {metrics_dir}")
    logger.info(f"Debug log: {metrics_dir / f'{run_name}_debug.csv'}")
    logger.info(f"Latest metrics: {metrics_dir / f'{run_name}_latest.json'}")
    logger.info(f"Summary: {metrics_dir / f'{run_name}_summary.json'}")
    
    return summary


def compare_modes(duration: int = 2000, seed: int = 42, use_gui: bool = False, pinn_model_path: Optional[str] = None):
    """
    Run and compare all three modes: OFF, CURRENT (reactive), IMPROVED (predictive).
    
    Args:
        duration: Simulation duration in steps
        seed: Random seed for reproducibility
        use_gui: Whether to use SUMO GUI
        pinn_model_path: Optional path to a trained PINN checkpoint
    """
    
    results = {}
    
    for mode in ['OFF', 'CURRENT', 'IMPROVED']:
        logger.info(f"\n{'='*70}")
        logger.info(f"Running mode: {mode}")
        logger.info(f"{'='*70}\n")
        
        try:
            summary = run_predictive_corridor_demo(
                mode=mode,
                duration=duration,
                seed=seed,
                use_gui=use_gui,
                pinn_model_path=pinn_model_path,
            )
            results[mode] = summary
        except Exception as e:
            logger.exception(f"Failed to run {mode}: {e}")
            results[mode] = {'error': str(e)}
    
    # Comparison report
    logger.info("\n" + "="*70)
    logger.info("COMPARISON REPORT")
    logger.info("="*70)
    
    comparison = {
        'timestamp': time.time(),
        'duration': duration,
        'seed': seed,
        'modes': results,
    }
    
    # Calculate metrics
    for mode, summary in results.items():
        if 'error' in summary:
            continue
        
        logger.info(f"\n{mode}:")
        logger.info(f"  Emergency vehicles: {summary.get('emergency_vehicle_count', 0)}")
        logger.info(f"  Active corridors: {summary.get('active_corridors', 0)}")
        logger.info(f"  Vehicles slowed: {summary.get('last_cleared_slowed', 0)}")
        logger.info(f"  Vehicles rerouted: {summary.get('last_cleared_rerouted', 0)}")
    
    # Save comparison
    repo_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    comparison_file = repo_root / 'results' / 'demo_metrics' / 'comparison_report.json'
    comparison_file.parent.mkdir(parents=True, exist_ok=True)
    comparison_file.write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    
    logger.info(f"\nComparison saved to: {comparison_file}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Predictive green wave emergency corridor demonstration'
    )
    parser.add_argument(
        '--mode',
        choices=['OFF', 'CURRENT', 'IMPROVED'],
        default='IMPROVED',
        help='Corridor mode (default: IMPROVED)',
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=2000,
        help='Simulation duration in steps (default: 2000)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)',
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Use SUMO GUI',
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Run and compare all three modes',
    )
    parser.add_argument(
        '--pinn-model-path',
        default=None,
        help='Path to trained PINN model for predictive congestion predictions',
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_modes(duration=args.duration, seed=args.seed, use_gui=args.gui, pinn_model_path=args.pinn_model_path)
    else:
        run_predictive_corridor_demo(
            mode=args.mode,
            duration=args.duration,
            seed=args.seed,
            use_gui=args.gui,
            pinn_model_path=args.pinn_model_path,
        )


if __name__ == '__main__':
    main()
