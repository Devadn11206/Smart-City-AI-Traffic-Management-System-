# Emergency Corridor Design Audit (Current Codebase)

Date: 2026-06-05

This audit inspects the current emergency-corridor implementation and its integration points across:
- `smart_city_ai/integration/hybrid_smart_city_controller.py`
- `smart_city_ai/integration/rl_pinn_aco_integration.py`
- `smart_city_ai/integration/full_hybrid_system.py`

It documents:
1) current emergency detection
2) current corridor activation and preemption
3) current ACO interaction points
4) current RL override logic
5) missing components preventing zero-stop behavior, green-wave progression, corridor protection, lane clearing, and emergency KPI reporting.

---

## 1. Current Emergency Detection Mechanism

### Where detection happens
**Primary detection loop:**
- `smart_city_ai/integration/rl_pinn_aco_integration.py`

In the main simulation loop, each tick:
- collects `all_ids = set(traci.vehicle.getIDList())`
- filters vehicles where `traci.vehicle.getTypeID(vid).lower()` is one of:
  - `emergency`, `ambulance`, `police`, `firetruck`, `fire_truck`
- for each such vehicle `vid`, fetches its route via `traci.vehicle.getRoute(vid)`.

### What is (and isn’t) tracked
- There is **no persistent registry** of active emergency vehicles.
- Detection is **stateless per tick** (recomputed each step from `traci.vehicle.getIDList()`).
- No explicit “spawn immediately” handling: it is effectively “immediate” only insofar as the next simulation tick sees the vehicle.

**Consequence:** emergency corridor activation can be delayed or oscillate if detection misses a tick (e.g., vehicle spawned between checks, type mismatch, transient ID list changes).

---

## 2. Current Corridor Activation Logic

### Activation entrypoint
When an emergency vehicle is detected, the integration computes a set of TLS that should be coordinated:
- iterates `tls_list = traci.trafficlight.getIDList()`
- for each TLS `ts`:
  - gets controlled lanes: `traci.trafficlight.getControlledLanes(ts)`
  - maps lanes to edges: `edges = set(traci.lane.getEdgeID(l) for l in lanes)`
  - checks overlap with the emergency vehicle’s route edges using:
    - `if any(e in edges for e in route[:6]): corridor.append(ts)`

If `corridor` is non-empty, it calls:
- `HybridSmartCityController.apply_coordinated_emergency_green(...)`
  - `sim_step=sim_step`
  - `hold_steps=args.emergency_hold_steps`
  - `route_edges=route`

### How HybridSmartCityController builds the TLS plan
In `smart_city_ai/integration/hybrid_smart_city_controller.py`:
- `apply_coordinated_emergency_green()`:
  - filters corridor TLS IDs that exist in `traci.trafficlight.getIDList()`
  - for each TLS `ts` assigns a phase plan using:
    - `cls._emergency_phase_plan[ts] = cls._safe_phase_index(ts, select_best_phase_for_tls(ts, route_edges or []))`
  - then calls `synchronize_emergency_corridor(corridor, sim_step, hold_steps)`

- `synchronize_emergency_corridor()`:
  - sets `cls._emergency_corridor` to the corridor TLS list (de-duplicated)
  - sets each TLS phase to the computed emergency phase
  - stores release timing in:
    - `cls.emergency_state[ts] = {'release_step': sim_step + hold_steps, 'hold_steps': hold_steps}`

### Phase restoration
- On every controller tick, `_release_expired_emergency_phases(sim_step)`:
  - restores prior phase for each expired TLS using cached values in `_phase_cache`
  - removes TLS from `_emergency_corridor`.

### What is missing for real green-wave progression
- Corridor activation uses only the **first 6 edges** of the emergency vehicle route (`route[:6]`).
- There is **no proactive reservation** that ensures:
  - 3–5 intersections ahead are already green/queued for progression
  - progression continues until the emergency vehicle fully clears those intersections
- There is **no ETA-based maintenance**; activation relies on a fixed `hold_steps` timeout.

---

## 3. Current Signal Preemption Logic (RL Override Protection)

### RL override guard
In `HybridSmartCityController.step_traci(traci_module, sim_step)`:
- when applying RL actions for each TLS:
  - it explicitly checks:
    - `if ts in cls.emergency_state: continue`

Meaning:
- RL will not change the phase of a TLS that is currently in emergency_state.

This is a correct and explicit “emergency overrides RL” protection.

### How phases are set during emergency
- Emergency phase is set directly by `traci.trafficlight.setPhase(ts, phase_index)`.
- It uses a helper to pick best aligned phase:
  - `select_best_phase_for_tls(ts_id, route_edges)`

### What is missing for “zero-stop” and reduced red-signal encounters
- `hold_steps` is fixed and not tied to emergency arrival time at each intersection.
- There is no coordination that prevents **late switching** as the vehicle approaches successive intersections.
- No check that the emergency vehicle is actually approaching that TLS’s controlled lanes at the moment of switch.

---

## 4. Current ACO Interaction Points

### Where ACO is used
In `rl_pinn_aco_integration.py`:
- ACO is always created when `--enable-aco` is truthy.
- `optimizer.step(sim_step, pinn_predictions=preds)` is called each tick.

### How ACO is influenced by emergency corridor
In the inspected code:
- **No corridor edge protection** exists.
- There is no logic that:
  - marks corridor edges as protected
  - prevents normal traffic rerouting into corridor segments
  - increases routing cost for protected edges

### Related but insufficient: phase coordination only
Emergency corridor coordination currently only affects RL-controlled TLS phases.
ACO routing remains unaware of “protected corridor edges”.

**Consequence:** surrounding traffic may still route into (or build congestion on) edges that the emergency vehicle will need.

---

## 5. Current RL Override Logic

### Implemented behavior
- RL TLS decisions are skipped if TLS is in emergency_state.

### Not implemented behavior
- RL is not “prevented from reserving” beyond the TLS already in emergency_state.
- There is no global gating that prevents RL from issuing new TLS changes in anticipation of an emergency green wave.

---

## 6. Missing Components Preventing the Target Capabilities

Below is a direct mapping to your requested missing capabilities.

### A) Predictive Green Wave (Missing)
Current system:
- sets emergency phases when it detects overlap with `route[:6]`
- uses fixed `hold_steps`

Missing:
1. Reserve **3–5 intersections ahead** using time/ETA.
2. Activate green **before** arrival.
3. Maintain progression through successive intersections until cleared.
4. Avoid late switching as the vehicle advances.

### B) Corridor Edge Protection (Missing)
Current system:
- does not produce a protected set of corridor edges.

Missing:
1. corridor edges should be marked “protected”
2. ACO should apply higher costs on protected edges
3. normal rerouting should be discouraged from protected segments

### C) Cooperative Traffic Clearing (Missing)
Current system:
- no lane-clearing behavior for blocking vehicles.

Missing:
1. detect blocking vehicles on the emergency route
2. attempt lane changes
3. if lane change fails, reroute via ACO
4. record clearing metrics

### D) Emergency KPI Reporting (Missing)
Current metrics:
- `snapshot_metrics()` logs basic counts like `emergency_vehicle_count`, `active_corridor_tls`, `emergency_phase_count`.

Missing:
- writing `results/emergency_optimization_report.json`
- full KPI set:
  - `emergency_detected`
  - `emergency_stop_count` / rate
  - `average_emergency_delay`
  - `corridor_activation_count` and success rates
  - `green_wave_success_rate`
  - lane clearing actions and success rate
  - `reroute_count`
  - `emergency_eta_improvement`

---

## Summary of Root Causes for Observed Issues

The observed emergency failures (stops at red, trapped behind traffic, delayed activation) are consistent with:
- fixed-hold emergency preemption (not ETA-based)
- limited lookahead (`route[:6]`) without predictive reservation
- no corridor edge protection in ACO routing
- no lane-clearing cooperation

These are architecture-level gaps rather than minor parameter tuning.

