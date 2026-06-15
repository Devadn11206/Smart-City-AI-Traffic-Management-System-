import os
import sys
import argparse
import pandas as pd
import numpy as np

# Ensure SUMO is in PATH
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

def collect_data(steps=3000, output_file="datasets/physics_traffic_data.csv"):
    config_file = "../sumo_network/config.sumocfg"
    sumo_binary = checkBinary('sumo')
    sumo_cmd = [sumo_binary, "-c", config_file, "--no-warnings", "true", "--random"]
    
    traci.start(sumo_cmd)
    
    # Get all non-internal edges
    edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
    
    # Pre-fetch edge lengths
    edge_lengths = {}
    for e in edges:
        # Get lanes of the edge, use the first lane's length
        lanes = traci.edge.getLaneNumber(e)
        if lanes > 0:
            lane_id = e + "_0"
            try:
                length = traci.lane.getLength(lane_id)
                # Physical length of the edge segment
                edge_lengths[e] = length 
            except:
                edge_lengths[e] = 100.0 # fallback
        else:
            edge_lengths[e] = 100.0

    data = []
    
    print(f"Starting data collection for {steps} steps on {len(edges)} edges...")
    for step in range(steps):
        traci.simulationStep()
        
        for edge_id in edges:
            n_veh = traci.edge.getLastStepVehicleNumber(edge_id)
            speed = traci.edge.getLastStepMeanSpeed(edge_id)
            
            # Density: vehicles per meter
            length = edge_lengths.get(edge_id, 100.0)
            density = n_veh / length
            
            data.append({
                "time_step": step,
                "edge_id": edge_id,
                "density": density,
                "velocity": speed
            })
            
        if step % 500 == 0:
            print(f"Step {step}/{steps} collected.")
            
    traci.close()
    
    df = pd.DataFrame(data)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Data collection complete! Saved to {output_file}")
    print(f"Total records: {len(df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000, help="Number of simulation steps")
    parser.add_argument("--output", type=str, default="datasets/physics_traffic_data.csv")
    args = parser.parse_args()
    
    collect_data(args.steps, args.output)
