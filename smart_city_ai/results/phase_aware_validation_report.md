# Phase-Aware Validation Report

| Agent | Reward | Throughput | Avg Queue | Avg Wait | Switches | Emergency Delay |
|-------|--------|------------|-----------|----------|----------|-----------------|
| **DQN** | -9866.17 | 319 | 1.38 | 7.00 | 20 | 1890 |
| **DDQN** | -12099.57 | 382 | 1.54 | 6.84 | 0 | 2194 |
| **PPO** | -12099.57 | 382 | 1.54 | 6.84 | 0 | 2194 |

## Reward Component Breakdown
Total sums accumulated during evaluation:

### DQN
| Component | Raw Contribution | % Contribution | Dominance | Flag |
|-----------|------------------|----------------|-----------|------|
| `worst_ev_wait` | -17035.50 | 33.8% | 1 |  |
| `stale_penalty` | -10730.00 | 21.3% | 2 |  |
| `ev_clearance` | 8500.00 | 16.8% | 3 |  |
| `ev_complete` | 7000.00 | 13.9% | 4 |  |
| `throughput` | 4785.00 | 9.5% | 5 |  |
| `fairness` | -2135.42 | 4.2% | 6 |  |
| `starvation` | -276.48 | 0.5% | 7 |  |
| `queue_delta` | -0.23 | 0.0% | 8 |  |
| `wait_delta` | -0.19 | 0.0% | 9 |  |
| `gridlock` | 0.00 | 0.0% | 10 |  |

### DDQN
| Component | Raw Contribution | % Contribution | Dominance | Flag |
|-----------|------------------|----------------|-----------|------|
| `worst_ev_wait` | -22787.00 | 36.5% | 1 |  |
| `stale_penalty` | -12330.00 | 19.8% | 2 |  |
| `ev_complete` | 11000.00 | 17.6% | 3 |  |
| `ev_clearance` | 8400.00 | 13.5% | 4 |  |
| `throughput` | 5730.00 | 9.2% | 5 |  |
| `fairness` | -1822.92 | 2.9% | 6 |  |
| `starvation` | -309.60 | 0.5% | 7 |  |
| `wait_delta` | -0.83 | 0.0% | 8 |  |
| `queue_delta` | -0.45 | 0.0% | 9 |  |
| `gridlock` | 0.00 | 0.0% | 10 |  |

### PPO
| Component | Raw Contribution | % Contribution | Dominance | Flag |
|-----------|------------------|----------------|-----------|------|
| `worst_ev_wait` | -26306.50 | 30.0% | 1 |  |
| `ev_complete` | 23500.00 | 26.8% | 2 |  |
| `stale_penalty` | -16584.00 | 18.9% | 3 |  |
| `ev_clearance` | 11000.00 | 12.5% | 4 |  |
| `throughput` | 7470.00 | 8.5% | 5 |  |
| `fairness` | -2408.75 | 2.7% | 6 |  |
| `starvation` | -407.94 | 0.5% | 7 |  |
| `wait_delta` | -1.51 | 0.0% | 8 |  |
| `queue_delta` | -0.91 | 0.0% | 9 |  |
| `gridlock` | 0.00 | 0.0% | 10 |  |

## Critical Policy Analysis
> [!INFO]
> Evaluated against the new Progress-Based reward design. Review the component breakdowns above to ensure no single factor contributes >40%.
