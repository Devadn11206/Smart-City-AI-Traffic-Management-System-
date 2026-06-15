# Reward Redesign Validation Report

| Agent | Reward | Throughput | Avg Queue | Avg Wait | Switches | Emergency Delay |
|-------|--------|------------|-----------|----------|----------|-----------------|
| **DQN** | 36652.53 | 320 | 1.53 | 7.81 | 20 | 2688 |
| **DDQN** | 45569.34 | 382 | 1.54 | 6.84 | 0 | 2194 |
| **PPO** | 45659.48 | 382 | 1.54 | 6.84 | 0 | 2194 |

## Observations
- **Switch Count**: DDQN and PPO must have switches > 0 to indicate no mode collapse.
- **Emergency Delay**: Should be significantly reduced compared to 1M+ delays.
