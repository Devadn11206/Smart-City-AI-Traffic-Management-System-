# Convergence Failure Analysis

## 1. Why DQN Reward Collapses After Episode 50
DQN utilizes an epsilon-greedy exploration strategy where `epsilon` decays from 1.0 down to `epsilon_min` over the first ~50 episodes. As epsilon decays, the agent transitions from taking random exploratory actions to relying purely on its Q-values. Because of the massive disparity in the reward function (where emergency wait penalties can reach -5,000 per step), the Q-network suffers from extreme overestimation bias for actions that "hold" the green phase for an emergency vehicle. Once exploration stops, DQN strictly exploits this suboptimal policy, causing throughput to crash and delays to spike as it gets stuck holding phases for EVs that cannot move due to congestion.

## 2. Why PPO Throughput Remains Capped Around 330-360
PPO's throughput is artificially capped because it has learned to **break green waves to appease the Emergency Vehicle penalty**. 
Throughput requires maintaining steady, long green phases on main corridors. However, because the agent incurs massive per-step penalties the moment an EV stops, PPO is mathematically incentivized to interrupt main traffic flows immediately. The resulting erratic, high-frequency switching (`84 switches` by episode 200) disrupts the flow of the 350+ civilian vehicles, resulting in gridlock that caps total throughput.

## 3. Why Emergency Delay Remains Above 2000
While PPO successfully lowered Emergency Delay from 727,000 down to 2,100, it cannot lower it further due to **physical traffic congestion**.
The agent switches the light green for the EV instantly. But because the agent has been ignoring throughput and creating stop-and-go traffic across the entire map, the intersection is physically blocked by queued civilian vehicles. The EV gets a green light, but has to wait for the gridlock ahead to clear before it can proceed.

## 4. Does Reward Scaling Cause Over-Prioritization?
**Yes. The reward scaling is fundamentally broken.**
- **Throughput:** `+15.0 * throughput`. Even with a perfect throughput of 400 vehicles, the agent only earns `+6,000` reward across the entire 1-hour episode.
- **Emergency Penalty:** `r_worst_ev_wait` applies `-min(5000.0, 0.5 * wait^2)` **PER STEP**. If an EV waits 50 seconds, the agent is penalized `-1,250` at step 1, `-1,250` at step 2, etc. A single EV waiting 10 seconds can wipe out the entire throughput reward for the episode. 
The agent is literally terrified of the EV penalty and will sacrifice the entire city's traffic to avoid it.

## 5. Are Queue-Reduction Rewards Too Weak?
**Yes.**
`r_queue_delta = -1.0 * queue_delta`. 
If a massive queue of 30 cars builds up because the agent is holding a phase for an EV, the agent only receives a `-30` penalty. The agent gladly accepts a `-30` penalty to avoid a `-1,250` EV penalty. Queue reduction has no voice in the reward signal.

## 6. Are Emergency Completion Bonuses Too Sparse?
**Yes.**
`r_ev_complete = 500.0`.
This only triggers once per episode per EV (when the EV successfully exits the map). 8 EVs = `4,000` total completion reward. The agent receives thousands of negative reward signals (per-step wait penalties) but almost no positive reinforcement when the EV actually finishes its route.

---

## Minimum Set of Changes Required

To achieve the strict target metrics (Throughput > 400, Wait < 8, Delay < 1000), we must balance the reward scale so that throughput and queue management are competitive with EV management.

1. **Boost Throughput Reward:**
   Increase from `15.0` to `50.0`.
   *(Max episode potential: ~20,000)*

2. **Linearize and Cap EV Wait Penalty:**
   Change from quadratic `-0.5 * wait^2` to linear `-2.0 * worst_ev_wait` to prevent the gradient from exploding and completely dominating the network.

3. **Boost Queue & Wait Penalties:**
   Increase `r_queue_delta` from `-1.0` to `-5.0`.
   Increase `r_wait_delta` from `-0.5` to `-2.0`.
   *(Forces the agent to respect civilian traffic flow while the EV is moving)*

4. **Boost EV Completion:**
   Increase `r_ev_complete` from `500.0` to `2000.0`.
   *(Strongly reinforces the actual goal rather than just punishing the waiting state)*
