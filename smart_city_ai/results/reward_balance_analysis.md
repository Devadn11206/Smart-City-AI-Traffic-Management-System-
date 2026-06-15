# Reward Balance Analysis

This analysis is based on the aggregated reward components across the 500-episode convergence training run.

## Component Contribution Breakdown

| Component | Total Accumulated Value | Gradient Strength / Effect |
|-----------|-------------------------|---------------------------|
| `ev_complete` | +24,735,000 | **Massive Positive Spike (Sparse).** Occurs only when an EV exits. High variance, but encourages clearing EVs. |
| `worst_ev_wait` | -28,363,570 | **Dominant Negative Force (Continuous).** Quadratic penalty applied *per step*. Forces agent into panic-switching. |
| `throughput` | +2,172,105 | **Weak Positive (Continuous).** Roughly 13x weaker than the EV wait penalty. Fails to incentivize green-wave maintenance. |
| `ev_clearance` | +4,126,000 | **Moderate Positive (Milestone).** Encourages pushing EVs through the immediate intersection. |
| `fairness` | -2,777,727 | **Moderate Negative.** Punishes the agent if one EV waits significantly longer than another. |
| `starvation` | -294,136 | **Weak Negative.** Punishes max lane wait times. Overpowered by EV wait penalty. |
| `queue_delta` | -179 | **Insignificant.** Zero statistical impact on the gradient. |
| `wait_delta` | -349 | **Insignificant.** Zero statistical impact on the gradient. |

## Critical Findings

1. **Severe Reward Dominance**
   The `-min(5000, 0.5 * wait^2)` penalty for `worst_ev_wait` is so mathematically overwhelming that the policy network's gradients are almost entirely dictated by avoiding this single state.
2. **Under-prioritization of Throughput**
   Throughput accounts for less than 7% of the total absolute reward magnitude. The agent learns that serving 30 civilian vehicles is mathematically inferior to saving 1 second of EV wait time.
3. **Queue Penalty Irrelevance**
   The queue and wait delta penalties are so microscopic (-179 total across 500 hours of traffic) that the agent fundamentally does not perceive congestion as a negative state, leading to catastrophic gridlock.
