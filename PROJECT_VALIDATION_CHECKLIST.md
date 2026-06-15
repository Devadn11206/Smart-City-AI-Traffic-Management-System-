# PROJECT_VALIDATION_CHECKLIST.md

Execution + validation checklist.

## A) Project structure validation
- [ ] `smart_city_ai\sumo_network\config.sumocfg` exists
- [ ] `smart_city_ai\pinn_models\trained_pinn.pth` exists (for PINN-enabled runs)
- [ ] ACO components import successfully:
  - `smart_city_ai\aco_routing\aco_router.py`
- [ ] Emergency components import successfully:
  - `smart_city_ai\emergency_system\emergency_router.py`

## B) RL checkpoint loading
Run:
```bat
python smart_city_ai\tools\rl_diagnostic.py --agent ddqn --steps 50
```
Evidence file/console:
- [ ] Console shows `✓ Loaded DDQN from ...pth`

Also verify compare:
```bat
python smart_city_ai\tools\rl_diagnostic.py --compare
```
- [ ] No `Model not found` messages

## C) PINN is actively generating predictions
Run PINN ablation:
```bat
python run_pinn_ablation_study.py
```
Validation evidence:
- [ ] `results\pinn_ablation\pinn_ablation_report.json` exists
- [ ] `With PINN` experiment shows `pinn_prediction_count`-related metrics in emergency outputs (or via its summary JSON)

Additionally, in integrated runs you should see:
- [ ] `pinn_prediction_count` in checkpoints JSON (`*_metrics_latest.json` / emergency latest)

## D) ACO rerouting is performing
Run integrated ACO+RL:
```bat
python smart_city_ai\integration\rl_pinn_aco_integration.py --config smart_city_ai\sumo_network\config.sumocfg --pinn smart_city_ai\pinn_models\trained_pinn.pth --duration 600 --enable-aco 1 --enable-pinn 1 --seed 42 --metrics-output-dir results\aco_results\checkpoints --run-name aco_reroute_check
```
Validation evidence:
- [ ] Output summary JSON includes a non-zero `reroutes` (or reroute-related debug CSV exists)
- [ ] `results\aco_results\pheromones_final.csv` created

## E) Emergency corridor activation
Run emergency corridor:
```bat
python smart_city_ai\integration\run_emergency_system.py --config smart_city_ai\sumo_network\config.sumocfg --pinn smart_city_ai\pinn_models\trained_pinn.pth --duration 600 --enable-corridor 1 --seed 42 --metrics-output-dir results\emergency_metrics --run-name emergency_corridor_validation
```
Evidence:
- [ ] `results\emergency_metrics\emergency_corridor_validation_summary.json` exists
- [ ] Summary includes `active_corridors` > 0 when emergency vehicles exist
- [ ] Checkpoints CSV/JSON exist under `results\emergency_metrics\` (run-specific names)

## F) Traffic clearing
Same emergency corridor run as above.
Evidence:
- [ ] Emergency system monitor reports “cleared” lanes/events (JSON reports under `results\emergency_metrics\*clear*report.json`)

## G) Predictive Green Wave
Evidence:
- [ ] With PINN + `--enable-corridor 1` should show higher corridor activations / longer proactive sequences than `--pinn` missing

## H) Multi-emergency support
Evidence:
- [ ] Emergency system debug logs/telemetry show multiple emergency vehicle IDs in runs
- [ ] Corridor activation happens for more than one emergency vehicle

## I) Reproducibility audit
- [ ] Run with the same seed twice and compare:
  - RL benchmark wait/queue/throughput mean and variance
  - Emergency summary values


