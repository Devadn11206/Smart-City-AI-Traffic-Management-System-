# emergency_detection_audit.md

## Summary
Emergency vehicle types and explicit trips are present in the SUMO scenario, but the runtime KPIs in the integration runner can still produce `emergency_detected = 0` if the runner’s detection logic never finds an emergency vehicle matching its filters.

This audit verifies **scenario definitions/spawns** first, then points to the **runner-side detection conditions** that can cause `emergency_detected` to remain 0.

## 1) Emergency vehicle definitions (vehicle_types.add.xml)
File: `smart_city_ai/sumo_network/vehicle_types.add.xml`

Defined `vType` entries:
- `ambulance` with `vClass="emergency"`
- `police` with `vClass="emergency"`
- `fire_truck` with `vClass="emergency"`

## 2) Emergency vehicle trips (routes.rou.xml)
File: `smart_city_ai/sumo_network/routes.rou.xml`

Explicit emergency trips:
- Trip `amb1`: type=`ambulance`, depart=`100`, from=`cz1_to_cz3` to=`hw5_to_rb2`
- Trip `pol1`: type=`police`, depart=`300`, from=`hw1_to_hw2` to=`cz2_to_cz4`
- Trip `fire1`: type=`fire_truck`, depart=`500`, from=`rz1_to_rz3` to=`rr3_to_rr4`

These depart times fall within a 0–600 simulation window.

## 3) Scenario loading (config.sumocfg)
File: `smart_city_ai/sumo_network/config.sumocfg`

The configuration loads:
- `route-files value="routes.rou.xml,random_trips.trips.xml"`
- `additional-files value="vehicle_types.add.xml"`

So emergency trips from `routes.rou.xml` should be present in TraCI.

## 4) Why `emergency_detected` can still be 0 in `rl_pinn_aco_integration.py`
In `smart_city_ai/integration/rl_pinn_aco_integration.py`, the Phase-2 reserved-corridor detection uses:

- Emergency filter:
  - `vtype = traci.vehicle.getTypeID(vid).lower()`
  - then `any(tok in vtype for tok in ('emergency','ambulance','police','firetruck','fire_truck','rescue'))`

- Route requirement:
  - `route = list(traci.vehicle.getRoute(vid) or [])`
  - then `if not route: continue`

- Corridor reserved edges are computed with a ~120m edge-length lookahead; lifecycle logs occur only for emergencies that enter this map.

- The KPIs are computed as:
  - `emergency_detected = len(_emergency_corridor_logs)`
  - `_emergency_corridor_logs` is populated only when a previously stored emergency disappears from the active set.

### Critical KPI implication
If the simulation ends before an emergency vehicle disappears from TraCI (or if reserved-edge computation never happens due to `getRoute` returning empty), then `_emergency_corridor_logs` can remain empty, yielding `emergency_detected = 0`.

## 5) What to check next (temporary diagnostics; no KPI/ACO/Clearing modifications)
Run `rl_pinn_aco_integration.py` and add temporary logging (best-effort) whenever a vehicle meets the emergency-identification predicate:

Log per step (or every N steps):
- `vid`
- `typeID = traci.vehicle.getTypeID(vid)`
- `vClass` if available (e.g., `traci.vehicle.getVehicleClass` if supported)
- `route_len = len(traci.vehicle.getRoute(vid))`
- `first_route_edge` and `last_route_edge`
- `depart_time` if available (e.g., via route/trip params if TraCI supports)

Then compare:
- expected emergency trips: amb1 (100s), pol1 (300s), fire1 (500s)
- observed `vehicle_id`s in TraCI across time
- whether `getRoute(vid)` is non-empty when they appear.

## 6) Current observed KPI snapshot
Example current report: `results/emergency_optimization_report.json`
- `emergency_detected = 0`
- `corridor_activation_count = 0`

Given that scenario spawns are present, remaining causes are expected to be:
1) runtime detection mismatch (typeID string differences vs filter), or
2) `getRoute` empty at runtime during the detection cycle, or
3) KPI logging lifecycle condition never triggers before end of simulation.


