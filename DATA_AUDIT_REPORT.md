# Data Audit Report

## 1. Multi-Scale RL Benchmark Data Quality
- **2x2, 4x4, 6x6 Networks**: All six algorithms (DQN, DDQN, PPO, IDQN, PressLight, CoLight) are present with valid `metrics.csv` files.
- **Anomalies Detected**: 
  - The `benchmark_research/6x6 bench mark` directory was identified as an exact duplicate of `6x6_benchmark`. It was excluded to prevent double counting.

## 2. PINN-Integrated RL Data Quality
- All expected models (DQN+PINN, DDQN+PINN, PPO+PINN for 6x6, and DDQN+PINN for 10x10) were successfully located.
- **Anomalies Detected**:
  - The DQN+PINN experiment was implemented as PI-MBPO (Physics-Informed Model-Based Policy Optimization). The naming `pi_mbpo_6x6_metrics.csv` was verified by reading `pi_mbpo_agent.py`.
  - MLOps artifacts (`mlruns`, `.dvc`) were not natively present in the workspace root, so MLOps descriptions rely on configuration files and Python scripts tracking the parameters.
