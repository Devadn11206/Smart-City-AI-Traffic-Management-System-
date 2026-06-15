# Reward Hacking Validation Report

| Agent | Reward | Throughput | Avg Queue | Avg Wait | Switches | Emergency Delay |
|-------|--------|------------|-----------|----------|----------|-----------------|
| **DQN** | -58287.11 | 346 | 2.58 | 14.43 | 20 | 6210 |
| **DDQN** | -12099.57 | 382 | 1.54 | 6.84 | 0 | 2194 |
| **PPO** | -12099.57 | 382 | 1.54 | 6.84 | 0 | 2194 |

## Reward Component Breakdown
Total sums accumulated during evaluation:

### DQN
| Component | Raw Contribution | % Contribution | Dominance | Flag |
|-----------|------------------|----------------|-----------|------|
| `worst_ev_wait` | -63225.00 | 61.6% | 1 | ⚠️ **>40% DOMINANCE** |
| `stale_penalty` | -10610.00 | 10.3% | 2 |  |
| `ev_complete` | 8500.00 | 8.3% | 3 |  |
| `ev_clearance` | 8300.00 | 8.1% | 4 |  |
| `fairness` | -6216.83 | 6.1% | 5 |  |
| `throughput` | 5190.00 | 5.1% | 6 |  |
| `starvation` | -550.26 | 0.5% | 7 |  |
| `wait_delta` | -0.38 | 0.0% | 8 |  |
| `queue_delta` | -0.34 | 0.0% | 9 |  |
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

## Critical Policy Analysis
> [!INFO]
> Evaluated against the new Progress-Based reward design. Review the component breakdowns above to ensure no single factor contributes >40%.
