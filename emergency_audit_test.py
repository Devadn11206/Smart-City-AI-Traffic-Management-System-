#!/usr/bin/env python3
"""
Audit test for emergency lane clearing system.
Runs with DEBUG logging to capture detailed audit traces.
"""
import logging
import subprocess
import sys
import os

# Enable DEBUG logging for all modules
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

# Get the root logger and set to DEBUG
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger('smart_city_ai').setLevel(logging.DEBUG)
logging.getLogger('emergency_system').setLevel(logging.DEBUG)

# Run the emergency system with reduced duration for quick audit
cmd = [
    sys.executable,
    'smart_city_ai/integration/run_emergency_system.py',
    '--duration', '15',
    '--config', 'smart_city_ai/sumo_network/config.sumocfg',
    '--metrics-output-dir', 'results/emergency_audit_run',
    '--run-name', 'audit_test',
    '--debug-log-path', 'results/emergency_audit_run/audit_debug.csv'
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
sys.exit(result.returncode)
