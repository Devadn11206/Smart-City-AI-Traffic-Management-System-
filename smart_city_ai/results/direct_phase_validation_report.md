# Direct Phase Selection Validation Report

| Agent | Reward | Throughput | Avg Queue | Avg Wait | Switches | Emergency Delay | Actions Distribution |
|-------|--------|------------|-----------|----------|----------|-----------------|----------------------|
| **DQN** | -62507.84 | 355 | 2.96 | 16.41 | 70 | 5625 | {0: 317, 1: 61, 2: 48, 3: 280} |
| **DDQN** | -1271392.25 | 328 | 53.01 | 272.01 | 98 | 9770347 | {0: 39, 1: 0, 2: 645, 3: 17} |
| **PPO** | -1087742.44 | 325 | 30.70 | 224.39 | 110 | 727138 | {0: 624, 1: 3, 2: 71, 3: 0} |

## Reward Component Breakdown
Total sums accumulated during evaluation:

### DQN
| Component | Raw Contribution | % Contribution | Dominance | Flag |
|-----------|------------------|----------------|-----------|------|
| `worst_ev_wait` | -78895.00 | 73.9% | 1 | ⚠️ **>40% DOMINANCE** |
| `ev_clearance` | 8200.00 | 7.7% | 2 |  |
| `ev_complete` | 7000.00 | 6.6% | 3 |  |
| `fairness` | -5643.92 | 5.3% | 4 |  |
| `throughput` | 5325.00 | 5.0% | 5 |  |
| `starvation` | -1141.47 | 1.1% | 6 |  |
| `stale_penalty` | -507.00 | 0.5% | 7 |  |
| `queue_delta` | -0.11 | 0.0% | 8 |  |
| `wait_delta` | -0.05 | 0.0% | 9 |  |
| `gridlock` | 0.00 | 0.0% | 10 |  |

### DDQN
| Component | Raw Contribution | % Contribution | Dominance | Flag |
|-----------|------------------|----------------|-----------|------|
| `worst_ev_wait` | -3162230.50 | 77.8% | 1 | ⚠️ **>40% DOMINANCE** |
| `fairness` | -609030.33 | 15.0% | 2 |  |
| `starvation` | -269744.34 | 6.6% | 3 |  |
| `ev_complete` | 8000.00 | 0.2% | 4 |  |
| `ev_clearance` | 6800.00 | 0.2% | 5 |  |
| `throughput` | 4920.00 | 0.1% | 6 |  |
| `wait_delta` | -1443.23 | 0.0% | 7 |  |
| `queue_delta` | -10.68 | 0.0% | 8 |  |
| `gridlock` | 0.00 | 0.0% | 9 |  |
| `stale_penalty` | 0.00 | 0.0% | 10 |  |

### PPO
| Component | Raw Contribution | % Contribution | Dominance | Flag |
|-----------|------------------|----------------|-----------|------|
| `worst_ev_wait` | -2557178.50 | 89.6% | 1 | ⚠️ **>40% DOMINANCE** |
| `fairness` | -200882.52 | 7.0% | 2 |  |
| `starvation` | -78893.34 | 2.8% | 3 |  |
| `ev_clearance` | 7000.00 | 0.2% | 4 |  |
| `throughput` | 4875.00 | 0.2% | 5 |  |
| `ev_complete` | 4000.00 | 0.1% | 6 |  |
| `wait_delta` | -163.87 | 0.0% | 7 |  |
| `queue_delta` | -4.66 | 0.0% | 8 |  |
| `gridlock` | 0.00 | 0.0% | 9 |  |
| `stale_penalty` | 0.00 | 0.0% | 10 |  |

## Critical Policy Analysis
> [!INFO]
> Evaluated against the new Progress-Based reward design. Review the component breakdowns above to ensure no single factor contributes >40%.
