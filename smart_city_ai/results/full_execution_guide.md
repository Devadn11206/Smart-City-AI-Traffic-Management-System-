# Smart City Traffic Management System
## Full Execution Guide

This document provides a comprehensive command matrix for executing every combination of features, visualizations, benchmarks, and tests available in the repository.

All commands should be run from the root of the repository. If you are in `c:\Users\ddnan\intenrship_IIIT`, you must first run:
```bash
cd smart_city_ai
```




---

## 1. Feature Scenarios (Command Matrix)

### SCENARIO A: ALL FEATURES ENABLED
*(RL + PINN + ACO + Emergency Corridor + Traffic Clearing + Metrics)*
```bash
python integration/rl_pinn_aco_integration.py --enable-aco 1 --enable-pinn 1
```

### SCENARIO B: RL ONLY
*(No PINN, No ACO. Emergency Corridor operates autonomously in the env)*
```bash
python integration/rl_pinn_aco_integration.py --enable-aco 0 --enable-pinn 0
```

### SCENARIO C & P: PINN VISUALIZATION
*(Visualize PINN density/velocity predictions operating alongside RL)*
```bash
python integration/rl_pinn_aco_integration.py --gui --enable-pinn 1 --enable-aco 0
```

### SCENARIO D & O: ACO VISUALIZATION
*(Visualize dynamic ACO route optimization avoiding congestion)*
```bash
python integration/rl_pinn_aco_integration.py --gui --enable-aco 1 --enable-pinn 0
```

### SCENARIO E & N: EMERGENCY CORRIDOR VISUALIZATION
*(Dedicated visualization focusing solely on `amb1`, `pol1`, `fire1` and green-wave clearance)*
```bash
python scripts/demo_predictive_corridor.py --mode IMPROVED --gui
```

### SCENARIO Q & L: FULL PROJECT VISUAL DEMO
*(This is the ultimate command for tomorrow's demo. Visualizes RL, PINN, ACO, and Emergency systems simultaneously).*
```bash
python integration/rl_pinn_aco_integration.py --gui --enable-aco 1 --enable-pinn 1 --duration 2000 --seed 42
```

---

## 2. Agent Visualization

### PPO Demo
*(Visualize the primary production agent)*
```bash
python scripts/visualize_agent.py --agent ppo
```

### DDQN Demo
*(Visualize the DDQN benchmark agent)*
```bash
python scripts/visualize_agent.py --agent ddqn
```

### DQN Demo
*(Visualize the baseline DQN agent)*
```bash
python scripts/visualize_agent.py --agent dqn
```

---

## 3. Training & Validation

### Train All Agents Automatically
*(Orchestrates sequential training for DQN, DDQN, and PPO)*
```bash
python scripts/train_all_agents.py
```

### Standalone Validation Runs
*(Runs inference and dumps metrics without GUI)*
```bash
python scripts/train_validation.py
```

### Benchmark Metric Generation
*(Evaluates all saved milestones and generates `training_progress_report.md`)*
```bash
python scripts/evaluate_milestones.py
```

### Emergency Validation Test
*(Validates EV throughput and clearance rates)*
```bash
python scripts/run_emergency_optimization_validation.py
```

---

## 4. Smoke Tests (Unit & Integration)
*Use these if the GUI fails to launch or if you suspect environment corruption.*

**Test PINN Mathematics (Headless):**
```bash
python tools/smoke_test_pinn.py
```

**Test ACO Mathematics (Headless):**
```bash
python tools/smoke_test_aco.py
```

**Test Full Pipeline Integration (Headless):**
```bash
python tools/smoke_test_integration.py
```

---

## 5. Troubleshooting & Common Failures

1. **`TraCIException: Vehicle is not known`**
   - *Cause:* Occurs if the routing optimizer attempts to reroute a vehicle that has just exited the map.
   - *Fix:* The ACO router has `try/except` blocks built in. Safe to ignore if it prints as a warning.

2. **SUMO GUI freezes or turns black**
   - *Cause:* The `rl_pinn_aco_integration.py` script is flooding the TraCI socket with too many phase updates per second.
   - *Fix:* Adjust the GUI "Delay" slider at the top of the SUMO window to `50ms` or `100ms` immediately after launching.

3. **`FileNotFoundError: ppo_demo_model.pth`**
   - *Cause:* The `models/demo/` directory was deleted or modified.
   - *Fix:* Re-copy `models/ppo/ppo_model_ep_100.pth` to `models/demo/ppo_demo_model.pth`.

4. **Throughput reads exactly 0 in custom scripts**
   - *Cause:* You are using `traci.simulation.getArrivedIDList()` instead of `traci.simulation.getArrivedNumber()`.
   - *Fix:* `getArrivedIDList()` only returns the vehicles that arrived in the exact millisecond of the query. Use `getArrivedNumber()` for accurate cumulative throughput.
