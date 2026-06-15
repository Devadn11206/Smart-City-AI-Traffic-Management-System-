# Absolute Reward Validation Report

| Agent | Reward | Throughput | Avg Queue | Avg Wait | Switches | Emergency Delay | Actions Distribution |
|-------|--------|------------|-----------|----------|----------|-----------------|----------------------|
| **DQN** | 3159.12 | 335 | 3.25 | 18.86 | 118 | 13697 | {0: 298, 1: 65, 2: 267, 3: 67} |
| **DDQN** | 32068.22 | 338 | 3.33 | 23.25 | 75 | 6122 | {0: 321, 1: 0, 2: 0, 3: 384} |
| **PPO** | -515815.19 | 315 | 17.72 | 177.18 | 94 | 1269270 | {0: 32, 1: 0, 2: 608, 3: 62} |

## Reward Component Breakdown
Total sums accumulated during evaluation:

### DQN
| Component | Raw Contribution | % Contribution | Dominance | Flag |
|-----------|------------------|----------------|-----------|------|
| `ev_complete` | 22000.00 | 28.2% | 1 |  |
| `worst_ev_wait` | -19742.00 | 25.3% | 2 |  |
| `fairness` | -14070.08 | 18.0% | 3 |  |
| `throughput` | 10050.00 | 12.9% | 4 |  |
| `ev_clearance` | 8600.00 | 11.0% | 5 |  |
| `wait` | -1666.98 | 2.1% | 6 |  |
| `starvation` | -1403.13 | 1.8% | 7 |  |
| `queue` | -566.75 | 0.7% | 8 |  |
| `gridlock` | 0.00 | 0.0% | 9 |  |
| `stale_penalty` | 0.00 | 0.0% | 10 |  |

### DDQN
| Component | Raw Contribution | % Contribution | Dominance | Flag |
|-----------|------------------|----------------|-----------|------|
| `ev_complete` | 34000.00 | 47.2% | 1 | ⚠️ **>40% DOMINANCE** |
| `worst_ev_wait` | -10194.00 | 14.1% | 2 |  |
| `throughput` | 10140.00 | 14.1% | 3 |  |
| `ev_clearance` | 8000.00 | 11.1% | 4 |  |
| `fairness` | -5293.75 | 7.3% | 5 |  |
| `wait` | -2084.28 | 2.9% | 6 |  |
| `starvation` | -1752.21 | 2.4% | 7 |  |
| `queue` | -587.00 | 0.8% | 8 |  |
| `stale_penalty` | -55.00 | 0.1% | 9 |  |
| `gridlock` | 0.00 | 0.0% | 10 |  |

### PPO
| Component | Raw Contribution | % Contribution | Dominance | Flag |
|-----------|------------------|----------------|-----------|------|
| `worst_ev_wait` | -479928.00 | 48.6% | 1 | ⚠️ **>40% DOMINANCE** |
| `fairness` | -270643.75 | 27.4% | 2 |  |
| `wait` | -139290.34 | 14.1% | 3 |  |
| `starvation` | -43651.08 | 4.4% | 4 |  |
| `ev_complete` | 34000.00 | 3.4% | 5 |  |
| `throughput` | 9450.00 | 1.0% | 6 |  |
| `ev_clearance` | 8400.00 | 0.8% | 7 |  |
| `queue` | -3110.25 | 0.3% | 8 |  |
| `stale_penalty` | -33.00 | 0.0% | 9 |  |
| `gridlock` | 0.00 | 0.0% | 10 |  |

## Critical Policy Analysis
> [!INFO]
> Evaluated against the Absolute Penalties reward design. Review the component breakdowns above to ensure no single factor contributes >40%.
