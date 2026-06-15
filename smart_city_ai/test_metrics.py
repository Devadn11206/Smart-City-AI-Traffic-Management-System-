#!/usr/bin/env python3
import sys
from pathlib import Path

repo_root = Path('.')
sys.path.insert(0, str(repo_root))

from integration.hybrid_env import HybridSumoEnv

config_file = repo_root / 'sumo_network/config.sumocfg'
pinn_model = repo_root / 'pinn_models/trained_pinn.pth'

print('Testing HybridSumoEnv.get_metrics()...')
env = HybridSumoEnv(str(config_file), str(pinn_model), num_seconds=1000)
env.reset()

# Run a few steps and check metrics
for i in range(5):
    env.step(0)
    metrics = env.get_metrics()
    print(f'Step {i}: wait={metrics["wait"]:.2f}, queue={metrics["queue"]:.2f}')
    
env.close()
print('Done!')
