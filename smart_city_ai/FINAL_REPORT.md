# Final Report

## Status

The core SUMO, RL, PINN, ACO, and emergency corridor systems are implemented and runnable. The project is now stabilized with repo-root-aware runners, a benchmark dashboard, and deployment docs.

## Working Modules

- SUMO environment and traffic generation
- DQN, DDQN, and PPO agents
- PINN congestion prediction pipeline
- ACO routing and pheromone management
- Emergency detection, routing, corridor control, ETA prediction, and clearing
- Evaluation and visualization scripts
- Streamlit dashboard

## Remaining Gaps

- Live dashboard controls are background-run launchers rather than a direct TraCI control plane.
- Long-horizon benchmarks are executable but still slow because SUMO/TraCI is the bottleneck.
- Some scripts continue to ignore exceptions internally when SUMO state is transient.