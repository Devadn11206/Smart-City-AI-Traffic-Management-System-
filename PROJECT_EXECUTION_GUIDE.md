# PROJECT_EXECUTION_GUIDE.md

This guide provides an end-to-end execution workflow (from setup to reproducible runs).

## 1) What modules exist (high-level)
The project is organized into:
- `smart_city_ai/tools/` : runnable training/eval/bench scripts
- `smart_city_ai/integration/` : integrated SUMO+TraCI controllers (RL+PINN+ACO, emergency system)
- `smart_city_ai/environment/` : SUMO environment wrapper for RL control
- `smart_city_ai/emergency_system/` : emergency detectors/routers/corridor managers/clearing
- `smart_city_ai/aco_routing/` : ACO pheromone manager + route optimizer/router
- `smart_city_ai/rl_agents/` : DQN/DDQN/PPO agent implementations
- `run_pinn_ablation_study.py` (repo root): explicit PINN A/B experiment driver

## 2) Preflight checks (must do)
### 2.1 SUMO_HOME
You must set:
- `SUMO_HOME` to your SUMO install folder.

Quick check:
```bat
where sumo
where sumo-gui
```

### 2.2 PINN checkpoint exists (if you want predictive mode)
```bat
dir smart_city_ai\pinn_models\trained_pinn.pth
```

### 2.3 RL checkpoints exist (if you want evaluation)
The hybrid runner loads “active checkpoints” managed by `utils/checkpoint_registry.py`.

Quick check (typical locations):
```bat
dir results\checkpoints\*.pth
dir results\hybrid_results\*.pth
dir models\ddqn\*.pth
```

## 3) Reproducible run order
### Phase A: Train RL agents
Train DQN / DDQN / PPO (one or multiple seeds).

### Phase B: Verify RL checkpoints load
Use the diagnostic tool:
```bat
python smart_city_ai\tools\rl_diagnostic.py --agent ddqn --steps 500
python smart_city_ai\tools\rl_diagnostic.py --agent ppo --steps 500
python smart_city_ai\tools\rl_diagnostic.py --compare
```

### Phase C: PINN ablation A/B
```bat
python run_pinn_ablation_study.py
```

### Phase D: Emergency corridor + predictive green wave
```bat
python smart_city_ai\integration\run_emergency_system.py --config smart_city_ai\sumo_network\config.sumocfg --pinn smart_city_ai\pinn_models\trained_pinn.pth --duration 2000 --enable-corridor 1 --seed 42 --metrics-output-dir results\emergency_metrics --run-name emergency_corridor_all
```

### Phase E: Integrated simulation + benchmarking
Run:
```bat
python smart_city_ai\tools\run_full_repro_benchmark.py
```

Or evaluation:
```bat
python smart_city_ai\tools\rl_comprehensive_benchmark.py --duration 600 --seeds 101 202 303 404 505 --output results\rl_benchmark
```

## 4) Key runtime artifacts
- RL evaluation metrics JSON: `results\aco_results\checkpoints\hybrid_{agent}_seed{seed}_metrics_latest.json`
- Full benchmark report: `results\rl_benchmark\benchmark_report.json`
- PINN ablation: `results\pinn_ablation\*`
- Emergency system summary: `results\emergency_metrics\{run_name}_summary.json`

## 5) Troubleshooting audit (important)
If you get TraCI / config errors:
1) Ensure `sumo_network\config.sumocfg` exists at the path used in your command.
2) Ensure `SUMO_HOME` is set.
3) Close any lingering TraCI processes and rerun (TraCI uses TCP ports).

One known issue: running from the wrong working directory can cause relative paths to fail.
Use absolute or run from repo root (`c:\Users\ddnan\intenrship_IIIT`).

