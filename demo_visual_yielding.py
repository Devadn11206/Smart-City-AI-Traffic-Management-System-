#!/usr/bin/env python3
"""
Demo script showcasing improved visual lane yielding for emergency vehicles.
Shows vehicles visibly moving aside with speed reduction and aggressive lane changes.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging for visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Add smart_city_ai to path
workspace_root = Path(__file__).parent
sys.path.insert(0, str(workspace_root / 'smart_city_ai'))

def run_visual_yielding_demo():
    """Run emergency vehicle demo with enhanced visual lane yielding."""
    try:
        from integration.hybrid_emergency_controller import HybridEmergencyController
        from environment.sumo_env import SUMOEnvironment
        import traci
        
        logger.info("=" * 80)
        logger.info("VISUAL LANE YIELDING DEMO - Enhanced Emergency Response")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Features:")
        logger.info("  ✓ Immediate speed reduction for blocking vehicles (60% reduction)")
        logger.info("  ✓ Aggressive lane change attempts in preferred direction")
        logger.info("  ✓ 'Courtesy wave' effect - multiple yielding strategies in sequence")
        logger.info("  ✓ Visual feedback through logging and metrics")
        logger.info("")
        
        # Initialize SUMO environment
        sumo_config = workspace_root / 'smart_city_ai' / 'sumo_network' / 'smart_city.sumocfg'
        if not sumo_config.exists():
            logger.error(f"SUMO config not found: {sumo_config}")
            return False
        
        logger.info(f"Loading SUMO network: {sumo_config}")
        env = SUMOEnvironment(
            config_file=str(sumo_config),
            use_gui=True,  # Enable GUI for visual feedback
            step_length=0.1,
        )
        
        logger.info("Initializing Hybrid Emergency Controller...")
        controller = HybridEmergencyController(
            pinn_model_path=str(workspace_root / 'smart_city_ai' / 'pinn_models' / 'trained_pinn.pth'),
            aco_router_enabled=True,
            emergency_corridor_enabled=True,
            predictive_gw_enabled=True,
            traffic_clearing_enabled=True,
            use_gui=True,
        )
        
        # Run simulation with visual yielding
        logger.info("\nStarting simulation - Watch vehicles yield for the ambulance!")
        logger.info("Duration: 60 seconds | Visual Features: Speed reduction, lane changes, traffic clearing")
        logger.info("-" * 80)
        
        total_yielded = 0
        max_steps = 600  # 60 seconds at 0.1s per step
        
        for step in range(max_steps):
            # Step simulation
            env.step()
            result = controller.step()
            
            # Get yielding report from traffic clearing system
            if hasattr(controller, 'traffic_clearing'):
                yielding_report = controller.traffic_clearing.get_yielding_report()
                currently_yielding = yielding_report.get('currently_yielding_vehicles', 0)
                total_yielded = yielding_report.get('total_vehicles_that_yielded', 0)
                
                # Print status every 10 steps
                if step % 10 == 0:
                    logger.info(
                        f"Step {step:3d} | "
                        f"Currently yielding: {currently_yielding:2d} vehicles | "
                        f"Total yielded: {total_yielded:3d} | "
                        f"Active corridors: {result.get('active_corridors', 0)}"
                    )
        
        logger.info("-" * 80)
        logger.info(f"\n✅ Demo completed successfully!")
        logger.info(f"   Total vehicles that yielded: {total_yielded}")
        logger.info(f"   Emergency vehicle passages: {result.get('emergency_vehicle_count', 0)}")
        logger.info(f"   Active corridors during run: {result.get('active_corridors', 0)}")
        
        # Generate report
        output_dir = workspace_root / 'results' / 'visual_yielding_demo'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report = {
            'demo': 'Visual Lane Yielding Enhancement',
            'duration_seconds': 60,
            'total_vehicles_yielded': total_yielded,
            'features_enabled': [
                'Speed reduction (60% for blocking vehicles)',
                'Aggressive lane changes (permissive mode 511)',
                'Courtesy wave effect (multiple attempts in sequence)',
                'Visual feedback through logging',
                'Screenshot capture on successful clears'
            ],
            'metrics': {
                'simulation_steps': max_steps,
                'total_yielded_vehicles': total_yielded,
                'active_corridors': result.get('active_corridors', 0),
                'emergency_vehicles': result.get('emergency_vehicle_count', 0),
            }
        }
        
        import json
        report_path = output_dir / 'visual_yielding_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n📊 Report saved to: {report_path}")
        logger.info(f"\n🎥 Screenshots saved to: results/emergency_metrics/screenshots/")
        
        return True
        
    except Exception as e:
        logger.error(f"Error running demo: {e}", exc_info=True)
        return False
    finally:
        try:
            if 'env' in locals():
                env.close()
        except:
            pass

if __name__ == '__main__':
    success = run_visual_yielding_demo()
    sys.exit(0 if success else 1)
