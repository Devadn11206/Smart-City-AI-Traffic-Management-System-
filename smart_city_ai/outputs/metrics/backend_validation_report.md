# Backend Validation Report

Date: 2026-05-27

## Scope
Backend AI system validation for SUMO + TraCI control, RL agents, PINN integration, ACO routing, and emergency orchestration.

## Verified Working
- TraCI controller startup and stepping are stable in short smoke runs.
- Latest short latency run completed cleanly, wrote samples to `results/traci_latency_samples.csv`, and no longer emits the shutdown socket traceback.
- SUMO started and ran under the controller with real-time factor above 17 in the latest short smoke run.
- PINN checkpoint loading works with the current `trained_pinn.pth` artifact.
- Emergency corridor logic and traffic clearing are active and producing metrics.
- RL comparison outputs were generated successfully.

## Broader Stress Sweep
- Rush-hour stability snapshot: `blocked_edges = 0`, `active_corridor_tls = 3`, `corridor_efficiency = 0.6`.
- Accident injection snapshot: `blocked_edges = 0`, `queue = 0.05`, `corridor_efficiency ~= 0.93`.
- Multi-emergency coordination snapshot: `blocked_edges = 0`, `active_corridors = 4`, `corridor_efficiency = 0.6`.
- Backend healthcheck snapshot: `blocked_edges = 0`, `active_corridors = 3`, `corridor_efficiency = 0.6`.

## Evidence
- TraCI smoke run: 8 seconds, real time factor `14.8055`, UPS `1325.834959`, `8` samples written.
- Emergency smoke summary: `blocked_edges = 0`, `wait = 0.0`, `queue = 0.0`, `corridor_efficiency = 0.6`.
- Emergency latest snapshot: `blocked_edges = 0`, `queue = 0.05`, `corridor_efficiency ~= 0.93`.
- Multi-emergency summary: `blocked_edges = 0`, `active_corridors = 4`, `corridor_efficiency = 0.6`.
- Comparison output in `visualizations/comparison_charts/final_evaluation_results.csv`:
  - Fixed-Time: wait `1.7067`, queue `0.4241`
  - Hybrid DDQN: wait `727.3342`, queue `43.2272`
  - Hybrid PPO: wait `315.4059`, queue `56.8118`

## Current Status
### Working
- Core SUMO/TraCI control loop
- PINN deserialization and model loading
- Emergency routing and corridor activation
- Metric collection and output generation
- Shutdown race suppression in TraCI snapshot capture

### Partially Working / Needs Review
- One comparison artifact shows hybrid runs performing worse than baseline under that evaluation path, so that path should be interpreted carefully before using it as a performance claim.

### No Critical Blockers Observed
- No active crash in the latest short controller smoke run.
- No missing core backend entry point found during validation.

## Recommendation
Proceed with a broader scenario sweep next:
- rush-hour stress
- high-density traffic
- accident injection
- ambulance/emergency response
- multi-emergency coordination
- rerouting stress
- RL-only vs RL+PINN vs RL+PINN+ACO comparisons

## Notes
- Dashboard work remains paused.
- The backend controller is the current stability boundary for the system.
