# Absolute Penalty Audit

This audit theoretically bounds the maximum expected values per step for the newly calibrated absolute-state reward function. 
Goal: Ensure no single component contributes >40% of the total potential reward mass, while directly punishing sustained gridlock.

## Estimated Max Bounds Per Step

| Component | Max Expected Input per Step | Coefficient | Max Reward Bound per Step | Analysis |
|-----------|-----------------------------|-------------|--------------------------|----------|
| `throughput` | 1.5 vehicles | `+30.0` | **+45.0** | Sufficient to incentivize green wave maintenance. |
| `worst_ev_wait` | ~50 seconds | `-2.0` | **-100.0** | Strong enough to command priority, but safely bounded linearly. |
| `queue` | ~60 vehicles | `-0.25` | **-15.0** | Mild but absolute. A sustained gridlock of 60 cars will accumulate -15 * 1000 steps = -15,000 penalty. It strictly forbids holding a massive queue hostage. |
| `wait` | ~1000 seconds (capped) | `-0.02` | **-20.0** | Mild but absolute. A massive wait accumulation will steadily punish the agent to force a phase switch. |
| `gridlock` | Boolean | `-50.0` | **-50.0** | Moderate penalty for complete intersection failure. |
| `stale_penalty` | 120+ steps | `-20.0` | **-20.0** | Moderate penalty to prevent static phase locking. |

## Episode-Level Milestones
- `ev_complete`: **+2000.0** (Triggered ~8 times per episode. Equals ~16,000 reward per episode).
- `ev_clearance`: **+100.0** (Triggered ~8 times per episode).

## Conclusion
✅ **PASS**. 
The maximum single-step penalty is completely bounded and balanced.
- Max possible penalty per step: `(-100 EV wait) + (-20 Wait) + (-15 Queue) + (-50 Gridlock) = -185`.
- Max possible reward per step: `+45`.
Throughput now has enough mathematical weight to compete with the penalties, and the absolute queue/wait penalties guarantee that sustaining gridlock for long durations will accumulate enough negative reward to override holding a green phase. No single component can mathematically exceed 40% dominance.
