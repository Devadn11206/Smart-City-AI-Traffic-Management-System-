# Reward Balance Validation Report

| Agent | Reward | Throughput | Avg Queue | Avg Wait | Switches | Emergency Delay | Actions Distribution |
|-------|--------|------------|-----------|----------|----------|-----------------|----------------------|
| **DQN** | 261.77 | 342 | 2.72 | 14.13 | 52 | 16791 | {0: 141, 1: 358, 2: 146, 3: 65} |
| **DDQN** | -1080692.93 | 326 | 46.35 | 237.67 | 100 | 7574932 | {0: 84, 1: 0, 2: 598, 3: 18} |
| **PPO** | -113753.11 | 331 | 10.03 | 85.78 | 105 | 73921 | {0: 629, 1: 70, 2: 0, 3: 0} |

## Reward Component Breakdown
Total sums accumulated during evaluation:

### DQN
| Component | Raw Contribution | % Contribution | Dominance | Flag |
|-----------|------------------|----------------|-----------|------|
| `ev_complete` | 22000.00 | 27.2% | 1 |  |
| `worst_ev_wait` | -19642.00 | 24.3% | 2 |  |
| `fairness` | -19279.83 | 23.8% | 3 |  |
| `throughput` | 10260.00 | 12.7% | 4 |  |
| `ev_clearance` | 8400.00 | 10.4% | 5 |  |
| `starvation` | -1079.85 | 1.3% | 6 |  |
| `stale_penalty` | -287.00 | 0.4% | 7 |  |
| `queue_delta` | 0.28 | 0.0% | 8 |  |
| `wait_delta` | 0.17 | 0.0% | 9 |  |
| `gridlock` | 0.00 | 0.0% | 10 |  |

### DDQN
| Component | Raw Contribution | % Contribution | Dominance | Flag |
|-----------|------------------|----------------|-----------|------|
| `worst_ev_wait` | -1842876.00 | 68.0% | 1 | ⚠️ **>40% DOMINANCE** |
| `fairness` | -535290.12 | 19.8% | 2 |  |
| `starvation` | -290763.57 | 10.7% | 3 |  |
| `ev_complete` | 24000.00 | 0.9% | 4 |  |
| `throughput` | 9780.00 | 0.4% | 5 |  |
| `ev_clearance` | 7000.00 | 0.3% | 6 |  |
| `wait_delta` | 500.33 | 0.0% | 7 |  |
| `queue_delta` | 4.89 | 0.0% | 8 |  |
| `gridlock` | 0.00 | 0.0% | 9 |  |
| `stale_penalty` | 0.00 | 0.0% | 10 |  |

### PPO
| Component | Raw Contribution | % Contribution | Dominance | Flag |
|-----------|------------------|----------------|-----------|------|
| `worst_ev_wait` | -88218.00 | 37.8% | 1 |  |
| `fairness` | -71658.06 | 30.7% | 2 |  |
| `ev_complete` | 42000.00 | 18.0% | 3 |  |
| `starvation` | -13893.48 | 5.9% | 4 |  |
| `throughput` | 9930.00 | 4.3% | 5 |  |
| `ev_clearance` | 7900.00 | 3.4% | 6 |  |
| `wait_delta` | 0.45 | 0.0% | 7 |  |
| `queue_delta` | 0.40 | 0.0% | 8 |  |
| `gridlock` | 0.00 | 0.0% | 9 |  |
| `stale_penalty` | 0.00 | 0.0% | 10 |  |

## Critical Policy Analysis
> [!INFO]
> Evaluated against the new Progress-Based reward design. Review the component breakdowns above to ensure no single factor contributes >40%.
