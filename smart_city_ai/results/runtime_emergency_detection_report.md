# Runtime Emergency Detection Diagnostics

- emergency_detected (corridor log proxy): 0
- corridor_activation_count (corridor log proxy): 0

## Detected emergency vehicles
- flow_hw_wb.0
- flow_rr_ccw.0

## Missed emergency vehicles (filter FAIL events)
- flow_hw_eb.0
- flow_hw_eb_trk.0
- flow_hw_wb.1
- flow_rr_cw.0
- rand_veh0
- rand_veh1
- rand_veh2
- rand_veh3

## First detection time (step)
- flow_hw_wb.0: step 1
- flow_rr_ccw.0: step 1

## Active emergency counts (every 10 steps)
- step 1: active_emergency_count=2, total_vehicle_count=5
- step 2: active_emergency_count=2, total_vehicle_count=5
- step 3: active_emergency_count=2, total_vehicle_count=6
- step 4: active_emergency_count=2, total_vehicle_count=7
- step 5: active_emergency_count=2, total_vehicle_count=8
- step 6: active_emergency_count=2, total_vehicle_count=9
- step 7: active_emergency_count=2, total_vehicle_count=10

## Root cause (emergency_detected == 0)
- exact_root_cause: no_token_match
