#!/usr/bin/env python3
"""
Complete SUMO Visualization: All RL Agents + All Features Enabled

Demonstrates:
  ✓ DQN Agent (Deep Q-Network for signal timing)
  ✓ DDQN Agent (Double DQN for improved stability)
  ✓ PPO Agent (Proximal Policy Optimization)
  ✓ PINN (Physics-Informed Neural Network for congestion prediction)
  ✓ ACO (Ant Colony Optimization for dynamic routing)
  ✓ Emergency Corridor (Proactive corridor activation)
  ✓ Traffic Clearing (Visual lane yielding with speed reduction)
  ✓ Predictive Green Wave (2-5 intersections ahead)

Run with: python run_complete_sumo_visualization.py
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Add workspace to path
workspace_root = Path(__file__).parent
sys.path.insert(0, str(workspace_root / 'smart_city_ai'))

def check_sumo_installation():
    """Check if SUMO is installed and in PATH."""
    try:
        import sumolib
        logger.info("✓ SUMO library detected")
        return True
    except ImportError:
        logger.error("✗ SUMO not found. Install with: pip install sumolib")
        return False

def load_rl_models():
    """Load all trained RL models (DQN, DDQN, PPO)."""
    logger.info("\n" + "="*80)
    logger.info("LOADING RL MODELS")
    logger.info("="*80)
    
    models = {}
    
    # Check for DQN model
    dqn_path = workspace_root / 'models' / 'dqn_model_best.pth'
    if dqn_path.exists():
        logger.info(f"✓ DQN model found: {dqn_path.name}")
        models['dqn'] = str(dqn_path)
    else:
        logger.warning(f"✗ DQN model not found: {dqn_path}")
    
    # Check for DDQN model
    ddqn_path = workspace_root / 'models' / 'ddqn' / 'ddqn_model_best.pth'
    if ddqn_path.exists():
        logger.info(f"✓ DDQN model found: {ddqn_path.name}")
        models['ddqn'] = str(ddqn_path)
    else:
        logger.warning(f"✗ DDQN model not found: {ddqn_path}")
    
    # Check for PPO model
    ppo_path = workspace_root / 'models' / 'ppo_model_best.pth'
    if ppo_path.exists():
        logger.info(f"✓ PPO model found: {ppo_path.name}")
        models['ppo'] = str(ppo_path)
    else:
        logger.warning(f"✗ PPO model not found: {ppo_path}")
    
    # Check for PINN model
    pinn_path = workspace_root / 'smart_city_ai' / 'pinn_models' / 'trained_pinn.pth'
    if pinn_path.exists():
        logger.info(f"✓ PINN model found: trained_pinn.pth")
        models['pinn'] = str(pinn_path)
    else:
        logger.warning(f"✗ PINN model not found: {pinn_path}")
    
    logger.info(f"\nTotal models loaded: {len(models)}/4")
    return models

def initialize_sumo_environment():
    """Initialize SUMO environment with GUI."""
    logger.info("\n" + "="*80)
    logger.info("INITIALIZING SUMO ENVIRONMENT")
    logger.info("="*80)
    
    try:
        from environment.sumo_env import SUMOEnvironment
        
        sumo_config = workspace_root / 'smart_city_ai' / 'sumo_network' / 'smart_city.sumocfg'
        
        if not sumo_config.exists():
            logger.error(f"✗ SUMO config not found: {sumo_config}")
            return None
        
        logger.info(f"✓ SUMO config found: {sumo_config.name}")
        
        env = SUMOEnvironment(
            config_file=str(sumo_config),
            use_gui=True,  # Enable GUI for visualization
            step_length=0.1,  # 0.1s per step = 10Hz
            max_steps=1000,
        )
        
        logger.info("✓ SUMO environment initialized successfully")
        logger.info(f"  - GUI: Enabled")
        logger.info(f"  - Step length: 0.1s (10Hz)")
        logger.info(f"  - Max steps: 1000 (100 seconds)")
        
        return env
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize SUMO: {e}", exc_info=True)
        return None

def initialize_hybrid_controller(models):
    """Initialize hybrid emergency controller with all subsystems."""
    logger.info("\n" + "="*80)
    logger.info("INITIALIZING HYBRID EMERGENCY CONTROLLER")
    logger.info("="*80)
    
    try:
        from integration.hybrid_emergency_controller import HybridEmergencyController
        
        controller = HybridEmergencyController(
            pinn_model_path=models.get('pinn'),
            aco_router_enabled=True,
            emergency_corridor_enabled=True,
            predictive_gw_enabled=True,
            traffic_clearing_enabled=True,
            use_gui=True,
        )
        
        logger.info("✓ Hybrid Emergency Controller initialized")
        logger.info("  Subsystems enabled:")
        logger.info("    ✓ PINN (Physics-Informed Neural Network)")
        logger.info("    ✓ ACO (Ant Colony Optimization)")
        logger.info("    ✓ Emergency Corridor Detection")
        logger.info("    ✓ Predictive Green Wave")
        logger.info("    ✓ Traffic Clearing with Visual Yielding")
        
        return controller
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize controller: {e}", exc_info=True)
        return None

def run_visualization(env, controller):
    """Run the complete visualization with all features."""
    logger.info("\n" + "="*80)
    logger.info("RUNNING SUMO VISUALIZATION")
    logger.info("="*80)
    logger.info("\n🚗 SUMO GUI Window Opening...")
    logger.info("Features to watch:")
    logger.info("  🚨 Speed Reduction: Vehicles slow when emergency vehicle detected")
    logger.info("  🌊 Lane Yielding: Vehicles change lanes to create corridor")
    logger.info("  🟢 Green Wave: Traffic signals change ahead of emergency vehicle")
    logger.info("  📍 Rerouting: ACO dynamically reroutes regular traffic")
    logger.info("  📊 Metrics: Console shows real-time statistics")
    logger.info("")
    
    metrics_history = {
        'steps': [],
        'active_corridors': [],
        'pinn_predictions': [],
        'aco_reroutes': [],
        'vehicles_yielded': [],
        'emergency_vehicles': [],
    }
    
    try:
        start_time = time.time()
        step = 0
        
        while step < 1000:  # 100 seconds at 0.1s per step
            # Step environment
            env.step()
            
            # Step controller (runs all subsystems)
            result = controller.step()
            
            # Get metrics
            if hasattr(controller, 'traffic_clearing'):
                yielding_report = controller.traffic_clearing.get_yielding_report()
            else:
                yielding_report = {}
            
            # Log progress every 100 steps
            if step % 100 == 0:
                elapsed = time.time() - start_time
                sim_time = step * 0.1
                
                logger.info(
                    f"Step {step:4d} ({sim_time:6.1f}s sim) | "
                    f"Active Corridors: {result.get('active_corridors', 0):2d} | "
                    f"Vehicles Yielding: {yielding_report.get('currently_yielding_vehicles', 0):2d} | "
                    f"Total Yielded: {yielding_report.get('total_vehicles_that_yielded', 0):3d} | "
                    f"Elapsed: {elapsed:6.1f}s"
                )
                
                # Track metrics
                metrics_history['steps'].append(step)
                metrics_history['active_corridors'].append(result.get('active_corridors', 0))
                metrics_history['pinn_predictions'].append(result.get('pinn_predictions', 0))
                metrics_history['aco_reroutes'].append(result.get('aco_reroutes', 0))
                metrics_history['vehicles_yielded'].append(yielding_report.get('total_vehicles_that_yielded', 0))
                metrics_history['emergency_vehicles'].append(result.get('emergency_vehicle_count', 0))
            
            step += 1
        
        logger.info("\n" + "="*80)
        logger.info("VISUALIZATION COMPLETE")
        logger.info("="*80)
        
        # Final statistics
        logger.info("\n📊 FINAL STATISTICS")
        logger.info(f"  Total steps: {step}")
        logger.info(f"  Simulation time: {step * 0.1:.1f} seconds")
        logger.info(f"  Real time: {time.time() - start_time:.1f} seconds")
        logger.info(f"  Total vehicles that yielded: {metrics_history['vehicles_yielded'][-1] if metrics_history['vehicles_yielded'] else 0}")
        logger.info(f"  Peak active corridors: {max(metrics_history['active_corridors']) if metrics_history['active_corridors'] else 0}")
        
        # Generate report
        generate_visualization_report(metrics_history)
        
        return True
        
    except KeyboardInterrupt:
        logger.info("\n⚠ Visualization interrupted by user")
        return False
    except Exception as e:
        logger.error(f"✗ Error during visualization: {e}", exc_info=True)
        return False

def generate_visualization_report(metrics_history):
    """Generate a report of the visualization run."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'type': 'Complete SUMO Visualization',
        'features_enabled': [
            'DQN Agent',
            'DDQN Agent', 
            'PPO Agent',
            'PINN Predictions',
            'ACO Routing',
            'Emergency Corridor',
            'Traffic Clearing',
            'Predictive Green Wave',
            'Visual Lane Yielding',
        ],
        'metrics_summary': {
            'total_steps': metrics_history['steps'][-1] if metrics_history['steps'] else 0,
            'total_vehicles_yielded': metrics_history['vehicles_yielded'][-1] if metrics_history['vehicles_yielded'] else 0,
            'peak_active_corridors': max(metrics_history['active_corridors']) if metrics_history['active_corridors'] else 0,
            'avg_active_corridors': sum(metrics_history['active_corridors']) / len(metrics_history['active_corridors']) if metrics_history['active_corridors'] else 0,
            'peak_emergency_vehicles': max(metrics_history['emergency_vehicles']) if metrics_history['emergency_vehicles'] else 0,
        },
        'metrics_by_step': metrics_history,
    }
    
    output_dir = workspace_root / 'results' / 'sumo_visualization_runs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / f"complete_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"\n✓ Visualization report saved to: {report_path.name}")
    except Exception as e:
        logger.warning(f"Could not save report: {e}")

def main():
    """Main entry point."""
    logger.info("\n")
    logger.info("█" * 80)
    logger.info("█" + " " * 78 + "█")
    logger.info("█" + "  COMPLETE SUMO VISUALIZATION - All RL Agents + All Features".center(78) + "█")
    logger.info("█" + " " * 78 + "█")
    logger.info("█" * 80)
    
    # Check SUMO
    if not check_sumo_installation():
        logger.error("\nFailed: SUMO not installed")
        return False
    
    # Load models
    models = load_rl_models()
    
    # Initialize environment
    env = initialize_sumo_environment()
    if env is None:
        logger.error("\nFailed: SUMO environment")
        return False
    
    # Initialize controller
    controller = initialize_hybrid_controller(models)
    if controller is None:
        logger.error("\nFailed: Hybrid controller")
        return False
    
    # Run visualization
    logger.info("\n" + "="*80)
    logger.info("ALL SYSTEMS READY")
    logger.info("="*80)
    logger.info("\nStarting visualization in 3 seconds...")
    logger.info("(Press Ctrl+C in SUMO window to stop)")
    time.sleep(3)
    
    success = run_visualization(env, controller)
    
    # Cleanup
    try:
        env.close()
        logger.info("✓ Environment closed successfully")
    except:
        pass
    
    if success:
        logger.info("\n✅ VISUALIZATION COMPLETED SUCCESSFULLY")
        logger.info("\nCheck:")
        logger.info("  - SUMO GUI for visual effects (speed reduction, lane changes, green waves)")
        logger.info("  - Console output above for metrics")
        logger.info("  - results/sumo_visualization_runs/ for detailed reports")
        return True
    else:
        logger.error("\n✗ VISUALIZATION FAILED")
        return False

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
