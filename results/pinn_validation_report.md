# PINN Validation Report (300s integrated run)

## Checkpoint path
- Requested PINN checkpoint: `smart_city_ai/pinn_models/trained_pinn.pth`
- Resolved (runtime): `C:\Users\ddnan\intenrship_IIIT\smart_city_ai\pinn_models\trained_pinn.pth`

## Model architecture
- Class: `TrafficPINN`
- Location: `smart_city_ai/pinn_models/pinn_model.py`
- Inputs: `(t, x)`
- Outputs: `(rho, v)`
- Physics loss: continuity residual `rho_t + (rho*v)_x = 0`

## Training script used
- Script: `smart_city_ai/pinn_models/train_pinn.py`
- Saved checkpoint to: `smart_city_ai/pinn_models/trained_pinn.pth`

## Load / fallback status
- Loader: `smart_city_ai/integration/rl_pinn_aco_integration.py::load_pinn()`
- Expected behavior:
  - If checkpoint loads: logs “PINN loaded successfully …”
  - If checkpoint fails: falls back to a zero predictor (and logs the exception)

## Observed prediction activity (from integration metrics)
- Metrics file: `results/aco_results/checkpoints/integration_metrics_latest.json`
- Observed `pinn_prediction_count`: **0**
- Observed `pinn_prediction_mean`: **0.0**
- Observed `pinn_prediction_max`: **0.0**
- Current integration metrics step snapshot: **31** (note: run may still be in progress when captured)

## Successful integration status
- Integrated runner used: `smart_city_ai/integration/full_hybrid_system.py`
- PINN enabled flag: `--enable-pinn 1`
- PINN predictions: **NOT ACTIVE** (prediction count is 0 in latest metrics snapshot)

## Notes / likely causes
- The integration snapshot indicates PINN predictor is being called, but produced **no edge predictions** at the time of measurement.
- `rl_pinn_aco_integration.py` computes predictions over `edges = list(adjacency.keys())`.
  - If adjacency keys do not align with `edge_map` keys inside the checkpoint, predictions may still return 0.0, but prediction_count should still be non-zero. This suggests the prediction loop may not be executing as expected (e.g., `args.enable_pinn`/`args.pinn` path mismatch) or metrics snapshot captured before first prediction refresh.


