#!/usr/bin/env python3
"""
Run predictive corridor demo with SUMO GUI
Uses DQN with all features: PINN, emergency corridor, traffic clearing
"""
import subprocess
import sys

cmd = [
    sys.executable,
    'smart_city_ai/scripts/demo_predictive_corridor.py',
    '--mode', 'IMPROVED',
    '--duration', '300',
    '--seed', '101',
    '--gui'
]

print('='*80)
print('DQN with Predictive Corridor - SUMO GUI')
print('='*80)
print('\nFeatures enabled:')
print('  ✓ DQN traffic signal control')
print('  ✓ PINN congestion predictions')
print('  ✓ Predictive green wave corridor')
print('  ✓ Emergency vehicle detection')
print('  ✓ Traffic clearing with lane yielding')
print('  ✓ Real-time metrics and visualization')
print('\n' + '='*80 + '\n')

try:
    subprocess.run(cmd, check=True)
except KeyboardInterrupt:
    print('\n✓ Simulation stopped by user')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)

