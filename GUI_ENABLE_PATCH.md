# GUI_ENABLE_PATCH.md

Minimal patch plan to enable SUMO-GUI visualization for RL pipelines (DQN/DDQN/PPO) while keeping PINN + ACO + Emergency corridor logic compatible.

## Why GUI currently can’t run
In the current code, RL execution is hardcoded to run in SUMO **non-GUI** mode:
- `smart_city_ai/tools/hybrid_runner.py` creates `HybridSumoEnv(..., use_gui=False, ...)`
- `smart_city_ai/tools/rl_comprehensive_benchmark.py` creates `HybridSumoEnv(..., num_seconds=...)` but `HybridSumoEnv` defaults `use_gui=False`
- `smart_city_ai/integration/rl_pinn_aco_integration.py` creates `SumoEnvironment(..., use_gui=False, ...)`

In turn, `smart_city_ai/environment/sumo_environment.py` launches:
- `sumo-gui` when `use_gui=True`
- `sumo` when `use_gui=False`

So GUI is effectively unavailable without adding a `--gui` flag and propagating it into those constructors.

## Target changes
Enable `--gui` support in these entry points:
- `smart_city_ai/tools/hybrid_runner.py`
- `smart_city_ai/tools/rl_comprehensive_benchmark.py`
- `smart_city_ai/integration/rl_pinn_aco_integration.py`

Propagate into environment launch by:
- calling `HybridSumoEnv(..., use_gui=args.gui)`
- calling `SumoEnvironment(..., use_gui=args.gui)`

## Patch (minimal diffs)
### 1) `smart_city_ai/tools/hybrid_runner.py`
Add CLI flag:
- `parser.add_argument('--gui', action='store_true', default=False)`

Then change:
- `HybridSumoEnv(args.config, args.pinn, use_gui=False, ...)`

to:
- `HybridSumoEnv(args.config, args.pinn, use_gui=bool(args.gui), ...)`

### 2) `smart_city_ai/tools/rl_comprehensive_benchmark.py`
Add CLI flag:
- `parser.add_argument('--gui', action='store_true', default=False)`

Then change env creation in `_load_agent`, `_eval_fixed_time`, `_eval_agent`:
- `HybridSumoEnv(str(self.config_file), str(self.pinn_path), num_seconds=...)`

to:
- `HybridSumoEnv(str(self.config_file), str(self.pinn_path), use_gui=bool(args.gui), num_seconds=...)`

### 3) `smart_city_ai/integration/rl_pinn_aco_integration.py`
Add CLI flag:
- `parser.add_argument('--gui', action='store_true', default=False)`

Then change:
- `env = SumoEnvironment(args.config, use_gui=False, num_seconds=args.duration)`

to:
- `env = SumoEnvironment(args.config, use_gui=bool(args.gui), num_seconds=args.duration)`

## Compatibility notes
- DQN/DDQN/PPO action selection is independent of GUI.
- PINN inference is independent of GUI.
- ACO and emergency corridor integration use TraCI and remain compatible with GUI.

## After patch: verified commands (once patch is applied)
Use the same config/pinn paths as your repo layout.

### Fixed-Time + GUI + PINN + ACO + Emergency Corridor
Use the integrated script that supports ACO/PINN flags:
```bat
python smart_city_ai\integration\rl_pinn_aco_integration.py --gui --config smart_city_ai\sumo_network\config.sumocfg --pinn smart_city_ai\pinn_models\trained_pinn.pth --duration 2000 --enable-aco 1 --enable-pinn 1 --seed 42 --metrics-output-dir results\aco_results\checkpoints --run-name fixed_time_gui_all
```

### DQN + GUI + PINN + ACO + Emergency Corridor
Run RL GUI via `hybrid_runner.py` (ACO+Emergency corridor are handled inside the integrated emergency controller stack; if you want emergency+corridor together with RL, use the full integration entrypoint instead). Closest supported combination for RL agent + PINN is:
```bat
python smart_city_ai\tools\hybrid_runner.py --gui --agent dqn --config smart_city_ai\sumo_network\config.sumocfg --pinn smart_city_ai\pinn_models\trained_pinn.pth --duration 600 --seed 42 --out-dir results\aco_results\checkpoints
```

### DDQN + GUI + PINN
```bat
python smart_city_ai\tools\hybrid_runner.py --gui --agent ddqn --config smart_city_ai\sumo_network\config.sumocfg --pinn smart_city_ai\pinn_models\trained_pinn.pth --duration 600 --seed 42 --out-dir results\aco_results\checkpoints
```

### PPO + GUI + PINN
```bat
python smart_city_ai\tools\hybrid_runner.py --gui --agent ppo --config smart_city_ai\sumo_network\config.sumocfg --pinn smart_city_ai\pinn_models\trained_pinn.pth --duration 600 --seed 42 --out-dir results\aco_results\checkpoints
```

Emergency corridor + predictive green wave is best run through:
```bat
python smart_city_ai\integration\run_emergency_system.py --gui --config smart_city_ai\sumo_network\config.sumocfg --pinn smart_city_ai\pinn_models\trained_pinn.pth --duration 2000 --enable-corridor 1 --seed 42 --metrics-output-dir results\emergency_metrics --run-name emergency_corridor_gui
```

## Files modified
This patch modifies exactly 3 files:
- `smart_city_ai/tools/hybrid_runner.py`
- `smart_city_ai/tools/rl_comprehensive_benchmark.py`
- `smart_city_ai/integration/rl_pinn_aco_integration.py`

