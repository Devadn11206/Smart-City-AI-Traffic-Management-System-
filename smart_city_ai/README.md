# Smart City AI Transportation System - Phase 1

This project phase sets up a realistic environment of a hybrid smart city using SUMO (Simulation of Urban MObility).

## Features
- **Complex Topography**: Not a basic grid network. Contains highways (4 lanes), ring roads, commercial zones (high density), residential zones (low speed, narrow roads), roundabout, and T-junctions.
- **Various Vehicle Types**: Includes passenger cars, trucks, buses, motorcycles, and emergency vehicles (police, ambulance, fire trucks) with unique parameters.
- **Traffic Spawning**: A combination of pre-defined realistic flow routes (e.g. continuous highway flows, continuous ring-road) and randomized traffic generation.
- **Traffic Signal System**: Auto-configured traffic lights at complex junction nodes.
- **TraCI Integration**: Contains `metrics_collector.py` that hooks into SUMO simulation to pull queue length, vehicle counts, average speed, and wait times needed for future RL (DQN, PPO, etc) learning parameters.

## Project Structure
- `sumo_network/`: Contains all generated XML configurations (nodes, edges, network, routes, config).
- `scripts/`: Python scripts to auto-generate the map, configure routes, invoke random traffic, and execute the actual simulation with TraCI.
- `outputs/`: Metrics data, saved as CSVs, and paths for logs/screenshots.

## How to Run

1. Make sure you have SUMO installed and `SUMO_HOME` environment variable set.
2. Ensure you have Python installed.
3. Open your terminal in the `smart_city_ai` folder.
4. Run the simulation script:
   ```bash
   python scripts/run_simulation.py
   ```
This will open the `sumo-gui`. Press the *Play* button to watch the smart city simulation run, complete with metrics collection.

## Full Project Validation
For a full runtime validation and audit across the main model and planning components, run the validation harness:
```bash
python tools/full_project_validation.py
```
To include the long-horizon reproducibility benchmark, add `--include-benchmark`.
