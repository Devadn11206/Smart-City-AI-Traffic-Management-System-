# PINN Ablation Study: Complete A/B Experiment Design

**Date**: June 2, 2026  
**Status**: ✅ Ready to Execute  

---

## Executive Summary

This document describes a rigorous controlled A/B experiment to measure the **isolated impact** of Physics-Informed Neural Networks (PINN) on smart city traffic management system performance.

**Objective**: Determine if PINN provides statistically meaningful performance improvements across congestion management, route selection, and emergency response domains.

---

## 1. Experimental Design

### Overview

Two parallel simulations run under **identical conditions** except for PINN enablement:

| Aspect | Experiment A | Experiment B |
|--------|-------------|-------------|
| **PINN** | ❌ Disabled | ✅ Enabled |
| **ACO** | ✅ Enabled | ✅ Enabled |
| **Emergency Corridor** | ✅ Enabled | ✅ Enabled |
| **Predictive Green Wave** | ✅ Enabled | ✅ Enabled |
| **Traffic Clearing** | ✅ Enabled | ✅ Enabled |

### Controlled Parameters (Fixed for Both)

```python
EXPERIMENT_CONFIG = {
    'duration_seconds': 100,           # Simulation duration (100 sec)
    'num_steps': 1000,                 # 0.1s per step = 1000 steps
    'step_length': 0.1,                # Each step = 0.1 seconds (10Hz)
    'random_seed': 42,                 # Fixed seed for reproducibility
    'traffic_demand_multiplier': 1.0,  # Vehicle generation rate
}
```

### Network & Traffic Properties

- **Network File**: `smart_city_ai/sumo_network/smart_city.sumocfg`
- **Vehicle Generation**: Fixed schedule from same route file
- **Emergency Schedule**: Same emergency vehicle triggers at same times
- **Lane Configuration**: Fixed across both experiments
- **Signal Timing**: RL-based (same policy used in both)

---

## 2. Metrics Collected

### Category 1: Congestion Management (Lower is Better)

1. **Average Waiting Time** (seconds)
   - Definition: Mean time vehicles wait at intersections
   - Good value: < 10 seconds
   - Impact: Directly affects passenger experience

2. **Average Queue Length** (vehicles)
   - Definition: Mean number of vehicles in queues
   - Good value: < 5 vehicles
   - Impact: Indicates congestion level

3. **Average Speed** (m/s) *Higher is Better*
   - Definition: Mean speed of all vehicles
   - Good value: > 8 m/s
   - Impact: Traffic flow efficiency

4. **Throughput** (vehicles/step) *Higher is Better*
   - Definition: Vehicles exiting system per time step
   - Good value: > 0.5 vehicles/step
   - Impact: System capacity utilization

### Category 2: Emergency Response (Lower is Better)

5. **Emergency Travel Time** (seconds)
   - Definition: Time for emergency vehicle to reach destination
   - Good value: < 180 seconds
   - Impact: Life-saving metric for emergency response

6. **Emergency Vehicle Stops** (count)
   - Definition: Number of times emergency vehicle stops completely
   - Good value: 0 stops
   - Impact: Unimpeded passage through network

### Category 3: System Activity (Context Dependent)

7. **ACO Reroutes** (count) *Higher is Better*
   - Definition: Number of dynamic route recalculations
   - Interpretation: More reroutes = better adaptation
   - Impact: System responsiveness to congestion

8. **Corridor Activations** (count) *Higher is Better*
   - Definition: Number of active emergency corridors
   - Interpretation: More corridors = more proactive response
   - Impact: Emergency preparedness level

---

## 3. Hypothesis Testing

### Statistical Framework

**Null Hypothesis (H0)**
- PINN provides no significant performance difference
- Any improvements are within noise margin

**Alternative Hypothesis (H1)**
- PINN provides meaningful improvements
- At least 5 out of 8 metrics improve

### Decision Criterion

```
If metrics_improved >= 5 out of 8:
    → Reject H0 ✓ (PINN is beneficial)
Else:
    → Fail to reject H0 (insufficient evidence)
```

---

## 4. Execution Steps

### Step 1: Run Experiment A (Without PINN)

```bash
python run_pinn_ablation_study.py
```

Configuration: `enable_pinn=False`

**Expected Output**:
- Baseline metrics without PINN predictions
- ACO reroutes using only current traffic state
- Emergency corridors without predictive lookahead

### Step 2: Run Experiment B (With PINN)

Automatic after Experiment A completes

Configuration: `enable_pinn=True`

**Expected Output**:
- Metrics with PINN congestion predictions
- ACO informed by 10-step lookahead predictions
- Proactive emergency corridor activation

### Step 3: Generate Reports

Automatic after both experiments complete

**Outputs**:
1. `pinn_ablation_report.json` - Complete data + comparison
2. `pinn_comparison_table.csv` - Formatted metrics table
3. `pinn_ablation_summary.txt` - Executive summary

### Step 4: Analyze Results

Open Jupyter notebook:

```bash
jupyter notebook pinn_ablation_study.ipynb
```

Review:
- Detailed metric comparisons
- Category-specific conclusions
- Hypothesis test results
- Deployment recommendations

---

## 5. Expected Results & Interpretations

### Scenario A: Strong PINN Benefit (6-8 metrics improved)

**Interpretation**: PINN is valuable

**Metrics likely to improve**:
- ✓ Emergency travel time (PINN enables predictive corridors)
- ✓ Waiting time (congestion prediction enables preemptive control)
- ✓ Corridor activations (PINN enables proactive decisions)

**Action**: Deploy with PINN enabled

### Scenario B: Moderate PINN Benefit (4-5 metrics improved)

**Interpretation**: PINN helps but not critical

**Metrics likely to improve**:
- ✓ Emergency response metrics
- ~ Traffic metrics mixed

**Action**: Deploy with PINN as optional/configurable

### Scenario C: Limited PINN Benefit (<4 metrics improved)

**Interpretation**: PINN needs tuning

**Potential Issues**:
- PINN predictions inaccurate for this network
- ACO not utilizing predictions effectively
- Network topology doesn't benefit from predictions

**Action**: Retrain PINN on network-specific data

---

## 6. Reproducibility

### Why This Matters

- **Scientific Rigor**: Same seed ensures exact reproducibility
- **Fair Comparison**: No algorithmic randomness differences
- **Peer Review**: Other researchers can verify results

### Reproducibility Features

1. **Fixed Random Seed**: `random_seed=42`
2. **Deterministic SUMO**: Fixed vehicle generation schedule
3. **Same Network**: Identical topology for both runs
4. **Same Emergency Events**: Same emergency vehicle times
5. **Logged Configuration**: All params saved with results

### Running Again with Different Seed

To test robustness:

```python
EXPERIMENT_CONFIG['random_seed'] = 123  # Different seed
# Run again to see if results remain consistent
```

---

## 7. Metrics Comparison Template

The system generates this table automatically:

```
┌──────────────────────────┬────────────────┬────────────┬──────────────┬──────────────┬───────────┬───────────┐
│ Metric                   │ Baseline (No   │ With PINN  │ Absolute     │ Percentage   │ Improved  │ Direction │
│                          │ PINN)          │            │ Difference   │ Change       │           │           │
├──────────────────────────┼────────────────┼────────────┼──────────────┼──────────────┼───────────┼───────────┤
│ avg_waiting_time         │ 12.34          │ 11.56      │ -0.78        │ -6.33%       │ ✓ YES     │ lower     │
│ avg_queue_length         │ 5.67           │ 4.89       │ -0.78        │ -13.77%      │ ✓ YES     │ lower     │
│ avg_speed                │ 7.89           │ 8.45       │ +0.56        │ +7.10%       │ ✓ YES     │ higher    │
│ avg_throughput           │ 0.34           │ 0.41       │ +0.07        │ +20.59%      │ ✓ YES     │ higher    │
│ avg_emergency_eta        │ 187.23         │ 156.34     │ -30.89       │ -16.49%      │ ✓ YES     │ lower     │
│ total_emergency_stops    │ 12             │ 8          │ -4           │ -33.33%      │ ✓ YES     │ lower     │
│ total_aco_reroutes       │ 45             │ 67         │ +22          │ +48.89%      │ ✓ YES     │ higher    │
│ avg_corridor_activations │ 2.3            │ 3.8        │ +1.5         │ +65.22%      │ ✓ YES     │ higher    │
└──────────────────────────┴────────────────┴────────────┴──────────────┴──────────────┴───────────┴───────────┘
```

---

## 8. Interpreting Results

### For Each Metric

**Lower is Better Metrics**:
- Waiting Time, Queue Length, Emergency ETA, Emergency Stops
- Good = Negative change (PINN shows lower values)
- Example: -6.33% improvement ✓

**Higher is Better Metrics**:
- Average Speed, Throughput, ACO Reroutes, Corridor Activations
- Good = Positive change (PINN shows higher values)
- Example: +7.10% improvement ✓

### Statistical Significance

With 8 metrics:
- **6-8 improved**: Strong evidence for PINN ✓✓✓
- **5 improved**: Moderate evidence for PINN ✓✓
- **4 improved**: Weak evidence, mixed results ✓
- **<3 improved**: Insufficient evidence for PINN ✗

---

## 9. Output Files Structure

```
results/
└── pinn_ablation/
    ├── pinn_ablation_report.json        # Complete data + metadata
    ├── pinn_comparison_table.csv        # Formatted metrics table
    └── pinn_ablation_summary.txt        # Executive summary
```

### pinn_ablation_report.json Structure

```json
{
  "metadata": {
    "experiment_type": "PINN Ablation Study",
    "timestamp": "2026-06-02T...",
    "configuration": { ... }
  },
  "experiment_a": {
    "experiment": "Baseline (No PINN)",
    "pinn_enabled": false,
    "avg_waiting_time": 12.34,
    "avg_queue_length": 5.67,
    ...
  },
  "experiment_b": {
    "experiment": "With PINN",
    "pinn_enabled": true,
    "avg_waiting_time": 11.56,
    "avg_queue_length": 4.89,
    ...
  },
  "comparison": {
    "experiment_a": "Baseline (No PINN)",
    "experiment_b": "With PINN",
    "comparison": {
      "avg_waiting_time": {
        "baseline": 12.34,
        "with_pinn": 11.56,
        "absolute_difference": -0.78,
        "percentage_change": -6.33,
        "improved": true,
        "direction": "lower_is_better"
      },
      ...
    },
    "overall_assessment": {
      "metrics_improved": 8,
      "total_metrics": 8,
      "improvement_rate": 100.0
    }
  }
}
```

---

## 10. Troubleshooting

| Issue | Solution |
|-------|----------|
| PINN model not found | Verify `smart_city_ai/pinn_models/trained_pinn.pth` exists |
| SUMO config not found | Check `smart_city_ai/sumo_network/smart_city.sumocfg` path |
| Simulation too slow | Normal - allow ~5-10 mins per experiment |
| Different results on re-run | Expected - different seed used. Set seed=42 for reproduction |
| Experiment B missing | Check logs - Experiment A may have failed |

---

## 11. Publication Quality Results

This ablation study generates publication-ready evidence for:

✓ PINN effectiveness claims  
✓ Component importance analysis  
✓ System design validation  
✓ Performance benchmarking  
✓ Deployment recommendations  

**Example Paper Claim**:
> "An ablation study comparing systems with and without PINN showed 75% of traffic metrics improved, demonstrating PINN's significant contribution to congestion reduction and emergency response efficiency."

---

## 12. Next Steps After Results

### If PINN is Beneficial (H1 accepted)

1. ✓ Include PINN in production deployment
2. ✓ Monitor PINN prediction accuracy in real-world
3. ✓ Plan periodic PINN model retraining
4. ✓ Document PINN as critical component

### If PINN Impact is Limited (H0 not rejected)

1. ⚠ Investigate PINN prediction accuracy
2. ⚠ Analyze ACO-PINN integration
3. ⚠ Consider network-specific PINN retraining
4. ⚠ Re-run ablation after improvements
5. ⚠ Evaluate alternative prediction methods

---

## 13. References

- Raissi et al. (2019): Physics-informed neural networks
- Dorigo & Stützle (2004): Ant Colony Optimization
- SUMO Documentation: Traffic Simulation
- Emergency Response Standards: NFPA 1010

---

**Last Updated**: June 2, 2026  
**Status**: ✅ Ready to Execute  
**Expected Runtime**: ~15 minutes per experiment  
**Total Time**: ~30 minutes for full analysis  
