# Reward Shaping V2 Validation Report

| Agent | Reward | Throughput | Avg Queue | Avg Wait | Switches | Emergency Delay |
|-------|--------|------------|-----------|----------|----------|-----------------|
| **DQN** | 27310.50 | 337 | 1.50 | 7.80 | 20 | 1874 |
| **DDQN** | 22797.86 | 382 | 1.54 | 6.84 | 0 | 2194 |
| **PPO** | 23004.36 | 382 | 1.54 | 6.84 | 0 | 2194 |

## Reward Component Contributions
Total sums accumulated during evaluation:

### DQN
- `throughput`: 5055.00
- `queue_delta`: -0.68
- `wait_delta`: -0.70
- `avg_speed`: 853.43
- `gridlock`: 0.00
- `starvation`: -323.49
- `stale_penalty`: -11430.00
- `ev_delay`: -28110.00
- `ev_stop`: -5520.00
- `ev_speed`: 45207.65
- `ev_clearance`: 8500.00
- `ev_complete`: 10000.00
- `switch_bonus`: 605.00
- `emergency_wait_bonus`: 2500.00

### DDQN
- `throughput`: 5730.00
- `queue_delta`: -0.45
- `wait_delta`: -0.83
- `avg_speed`: 860.12
- `gridlock`: 0.00
- `starvation`: -309.60
- `stale_penalty`: -12330.00
- `ev_delay`: -32910.00
- `ev_stop`: -7240.00
- `ev_speed`: 45215.42
- `ev_clearance`: 8400.00
- `ev_complete`: 12000.00
- `switch_bonus`: 0.00
- `emergency_wait_bonus`: 3400.00

### PPO
- `throughput`: 5730.00
- `queue_delta`: -0.45
- `wait_delta`: -0.83
- `avg_speed`: 860.12
- `gridlock`: 0.00
- `starvation`: -309.60
- `stale_penalty`: -12330.00
- `ev_delay`: -32910.00
- `ev_stop`: -7240.00
- `ev_speed`: 45215.42
- `ev_clearance`: 8400.00
- `ev_complete`: 12000.00
- `switch_bonus`: 210.00
- `emergency_wait_bonus`: 3400.00

## Critical Policy Analysis
> [!WARNING]
> **Why DDQN and PPO STILL refuse to switch:**
> The `ev_speed` component is now the heavily dominating factor (`r_ev_speed = 8.0 * ev_speed`).
> While we successfully nerfed `avg_speed` and added the `stale_penalty` (which correctly applied a massive `-12,330` penalty to the holding agents), the agents mathematically calculated that holding the green light is STILL the optimal policy.
> Why? Because holding the green light keeps the emergency vehicles on the main corridor moving continuously at high speeds (~12 m/s). This yields roughly `+96.0` reward PER STEP for every moving EV!
> Over the course of the episode, `ev_speed` alone generated **+45,215** reward.
> The agent willingly absorbs the `-32,910` penalty for the single stuck cross-street EV and the `-12,330` stale phase penalty, because the continuous +45k reward from keeping the main-corridor EVs moving at top speed massively outweighs the delayed benefits of switching.
> We have successfully fixed the PPO action-switching loophole, but the math still dictates that "never switch" is the most lucrative policy!
