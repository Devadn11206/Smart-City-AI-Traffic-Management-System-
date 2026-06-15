import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
script_dir = Path(__file__).resolve().parent
for path in (repo_root, script_dir):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import traci
from metrics_collector import collect_metrics, save_metrics

def main():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    if '--gui' in sys.argv:
        sumoBinary = "sumo-gui"
    else:
        sumoBinary = "sumo" # using headless sumo by default for faster automated run

    config_path = str((repo_root / 'sumo_network' / 'config.sumocfg').resolve())

    sumoCmd = [sumoBinary, "-c", config_path, "--start", "--quit-on-end"]

    traci.start(sumoCmd)
    
    metrics_list = []
    step = 0
    max_steps = 3600
    
    print("Starting simulation...")
    while step < max_steps:
        traci.simulationStep()
        
        # Collect metrics every 10 steps to reduce overhead
        if step % 10 == 0:
            metrics = collect_metrics(step)
            metrics_list.append(metrics)
            
        step += 1
        
    traci.close()
    print("Simulation finished.")
    
    # Save collected metrics
    metrics_output = str((repo_root / 'outputs' / 'metrics' / 'simulation_metrics.csv').resolve())
    save_metrics(metrics_list, metrics_output)

if __name__ == "__main__":
    main()
