# Execution Inventory

The following is a comprehensive inventory of all executables in the `smart_city_ai` repository capable of launching SUMO, TraCI, RL, PINN, ACO, Emergency Corridor, and Benchmarks.

### Master Integration Scripts
* `integration/rl_pinn_aco_integration.py` - Runs the complete Smart City simulation with RL, PINN, ACO, and Emergency systems.
* `integration/hybrid_smart_city_controller.py` - Manages multi-agent RL (DQN, DDQN, PPO) alongside TraCI.
* `scripts/demo_predictive_corridor.py` - Dedicated demonstration of Emergency Detection, Corridor, and Traffic Clearing.
* `scripts/start_demo.py` - Launches the Streamlit dashboard.

### Reinforcement Learning (RL)
* `training/train_ppo.py` - Train the PPO agent from scratch.
* `training/train_ddqn.py` - Train the DDQN agent from scratch.
* `training/train_dqn.py` - Train the DQN agent from scratch.
* `scripts/train_all_agents.py` - Automated training orchestration for all 3 agents.
* `scripts/run_convergence_training.py` - Specialized long-duration convergence training.
* `scripts/visualize_agent.py` - Visualize a single trained agent (e.g., PPO) in the SUMO GUI.
* `scripts/visualize_multi_agents.py` - Visualize the multi-agent controller system in the GUI.

### Benchmarks & Validations
* `scripts/evaluate_milestones.py` - Evaluate saved training milestones and generate `training_progress_report.md`.
* `scripts/demo_validation.py` - Verify demo model (`ppo_demo_model.pth`) health, metric generation, and TraCI connection.
* `scripts/train_validation.py` - Run a standalone metric generation loop for a specified model.
* `scripts/run_stop_audit.py` - Analyze stop-and-go behavior and traffic flow bottlenecks.
* `scripts/run_emergency_optimization_validation.py` - Specific validation testing emergency vehicle completion.

### PINN & ACO Scripts
* `tools/smoke_test_pinn.py` - Verify PINN traffic prediction.
* `tools/smoke_test_aco.py` - Verify ACO dynamic route selection.
* `tools/smoke_test_integration.py` - End-to-end smoke testing of PINN+ACO+RL sequence.

### Utility Scripts
* `scripts/generate_network.py` - Generate SUMO `.net.xml`.
* `scripts/generate_routes.py` - Generates traffic flows and emergency trips.
* `scripts/trace_ev_arrival.py` - Diagnostic trace for EV route clearance.
* `tools/verify_checkpoints.py` - Verifies weight matrices inside `.pth` files.
