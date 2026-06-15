from pathlib import Path
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from integration.hybrid_emergency_controller import HybridEmergencyController

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parent.parent
    config_file = str((repo_root / 'sumo_network' / 'config.sumocfg').resolve())

    controller = HybridEmergencyController(
        config_file=config_file,
        pinn_model_path=None,
        duration=60,
        use_gui=False,
        lookahead_steps=10,
        reroute_interval=10,
        decision_interval=2,
        pinn_interval=30,
        emergency_hold_steps=40,
        hospital_edges=[],
        metrics_output_dir=str(repo_root / 'results' / 'emergency_metrics' / 'stop_audit_short'),
        run_name='stop_audit_short',
        enable_corridor=True,
        seed=42,
        debug_log_path=str(repo_root / 'results' / 'emergency_metrics' / 'stop_audit_short' / 'stop_audit_debug.csv')
    )

    # enable the stop audit
    controller.stop_audit_enabled = True

    summary = controller.run()
    print('Run summary:', summary)
    report_path = repo_root / 'results' / 'response_analysis' / 'emergency_stop_root_cause_report.json'
    if report_path.exists():
        print('Stop audit report:', report_path)
    else:
        print('No report generated')
