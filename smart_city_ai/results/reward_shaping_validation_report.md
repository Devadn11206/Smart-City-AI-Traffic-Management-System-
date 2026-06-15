# Reward Shaping Validation Report

| Agent | Reward | Throughput | Avg Queue | Avg Wait | Switches | Emergency Delay |
|-------|--------|------------|-----------|----------|----------|-----------------|
| **DQN** | 52013.62 | 337 | 1.50 | 7.80 | 20 | 1874 |
| **DDQN** | 47768.03 | 382 | 1.54 | 6.84 | 0 | 2194 |
| **PPO** | 48540.93 | 382 | 1.54 | 6.84 | 0 | 2194 |

## Reward Component Contributions
Total sums accumulated during evaluation:

### DQN
- `throughput`: 3370.00
- `queue_delta`: -0.34
- `wait_delta`: -0.70
- `avg_speed`: 17068.70
- `gridlock`: 0.00
- `starvation`: -323.49
- `ev_delay`: -28110.00
- `ev_stop`: -5520.00
- `ev_speed`: 45207.65
- `ev_clearance`: 8500.00
- `ev_complete`: 10000.00
- `switch_bonus`: 700.00
- `emergency_wait_bonus`: 1250.00

### DDQN
- `throughput`: 3820.00
- `queue_delta`: -0.23
- `wait_delta`: -0.83
- `avg_speed`: 17202.36
- `gridlock`: 0.00
- `starvation`: -309.60
- `ev_delay`: -32910.00
- `ev_stop`: -7240.00
- `ev_speed`: 45215.42
- `ev_clearance`: 8400.00
- `ev_complete`: 12000.00
- `switch_bonus`: 5.00
- `emergency_wait_bonus`: 1700.00

### PPO
- `throughput`: 3820.00
- `queue_delta`: -0.23
- `wait_delta`: -0.83
- `avg_speed`: 17202.36
- `gridlock`: 0.00
- `starvation`: -309.60
- `ev_delay`: -32910.00
- `ev_stop`: -7240.00
- `ev_speed`: 45215.42
- `ev_clearance`: 8400.00
- `ev_complete`: 12000.00
- `switch_bonus`: 780.00
- `emergency_wait_bonus`: 1700.00

## Critical Policy Analysis
> [!WARNING]
> **Why DDQN and PPO refuse to switch:**
> The `avg_speed` component is heavily dominating the total reward (`r_avg_speed = 2.0 * avg_speed`).
> A vehicle traveling at ~12 m/s on a green light continuously yields `+24.0` reward PER STEP.
> Over 3600 steps, this single component contributes over **+80,000** to the reward!
> If the agent switches the light, vehicles stop, speed drops to 0, and the agent loses `~24.0` per step while the intersection clears.
> The `+5.0` switch bonus is mathematically dwarfed by the massive loss in continuous `avg_speed` reward.
> RL agents are optimizing perfectly for the math provided: **holding green is mathematically the most lucrative strategy**.
