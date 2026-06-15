import argparse
import os
import sys
import logging
from pathlib import Path


def _configure_repo_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _configure_sumo_path() -> None:
    if 'SUMO_HOME' not in os.environ:
        raise SystemExit("please declare environment variable 'SUMO_HOME'")
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the hybrid emergency response system.')
    parser.add_argument('--config', default=os.path.join('sumo_network', 'config.sumocfg'), help='SUMO config file path')
    parser.add_argument('--pinn', default=os.path.join('pinn_models', 'trained_pinn.pth'), help='Path to the trained PINN checkpoint')
    parser.add_argument('--duration', type=int, default=2000, help='Simulation duration in steps')
    parser.add_argument('--gui', action='store_true', help='Run with SUMO-GUI')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for the simulation')
    parser.add_argument('--enable-corridor', type=int, default=1, help='Enable emergency corridor coordination (1/0)')
    parser.add_argument('--lookahead-steps', type=int, default=10, help='PINN lookahead horizon')
    parser.add_argument('--reroute-interval', type=int, default=30, help='Reroute interval in simulation steps')
    parser.add_argument('--decision-interval', type=int, default=5, help='Control decision interval in simulation steps')
    parser.add_argument('--pinn-interval', type=int, default=30, help='PINN update interval in simulation steps')
    parser.add_argument('--emergency-hold-steps', type=int, default=20, help='How long to hold green corridor phases')
    parser.add_argument('--metrics-output-dir', default=os.path.join('results', 'emergency_metrics'), help='Directory for emergency metrics output')
    parser.add_argument('--run-name', default='full_emergency', help='Run name used in output file names')
    parser.add_argument('--debug-log-path', default='', help='Optional CSV path for per-emergency debug logging')
    parser.add_argument('--hospital-edges', default='', help='Comma-separated list of hospital edge IDs')
    return parser


def main() -> int:
    _configure_repo_path()
    _configure_sumo_path()

    # configure basic logging so components using logging.info/debug emit to console
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    from integration.hybrid_emergency_controller import HybridEmergencyController
    from utils.monitoring import start_monitoring

    parser = build_parser()
    args = parser.parse_args()
    hospital_edges = [edge.strip() for edge in args.hospital_edges.split(',') if edge.strip()]

    controller = HybridEmergencyController(
        config_file=os.path.abspath(args.config),
        pinn_model_path=os.path.abspath(args.pinn) if args.pinn else None,
        duration=args.duration,
        use_gui=args.gui,
        lookahead_steps=args.lookahead_steps,
        reroute_interval=args.reroute_interval,
        decision_interval=args.decision_interval,
        pinn_interval=args.pinn_interval,
        emergency_hold_steps=args.emergency_hold_steps,
        hospital_edges=hospital_edges,
        metrics_output_dir=os.path.abspath(args.metrics_output_dir),
        run_name=args.run_name,
        enable_corridor=bool(int(args.enable_corridor)),
        seed=args.seed,
        debug_log_path=os.path.abspath(args.debug_log_path) if args.debug_log_path else None,
    )
    # start background monitoring
    try:
        start_monitoring(out_dir=os.path.join('results', 'monitoring'), interval=10.0)
    except Exception:
        logging.getLogger(__name__).exception('Failed to start background monitoring')
    summary = controller.run()
    print(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
