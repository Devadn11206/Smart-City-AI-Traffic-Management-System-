# Traffic Signal Control 2x2 Benchmark Final Report

## 1. Network Description
The network consists of a 2x2 grid with 4 signalized intersections (TL1, TL2, TL3, TL4) and 8 priority boundary nodes. Roads are bidirectional, possessing 3 lanes per direction. Traffic flows probabilistically across the network for a simulated duration of 3600 seconds per episode.

## 2. State Space Description
The state space for IDQN, DQN, DDQN, and PPO is a highly normalized 10-dimensional continuous vector:
1-4: Queue Lengths (North, South, East, West)
5-8: Waiting Times (North, South, East, West)
9: Current Traffic Light Phase (Normalized 0-1)
10: Phase Duration (Normalized)

## 3. Action Space Description
A discrete action space with 4 actions mapping to standard NEMA phases:
- Action 0 → Phase 0 (North-South Straight/Right Green)
- Action 1 → Phase 2 (North-South Left Turn Green)
- Action 2 → Phase 4 (East-West Straight/Right Green)
- Action 3 → Phase 6 (East-West Left Turn Green)

## 4. Reward Function Description
The unified composite reward is designed as:
`Reward = Throughput - (0.05 * Queue Length) - (0.001 * Waiting Time) + Switch Penalty (-0.1)`
This balances maximizing vehicle completions while minimizing delays and flickering lights.

## 5-10. Algorithm Descriptions
- **IDQN**: Independent DQN where each intersection has its own isolated neural network and replay buffer.
- **DQN**: Parameter-sharing DQN where a single network processes experiences from all 4 intersections simultaneously.
- **DDQN**: Double DQN utilizing decoupled action-selection and action-evaluation networks to prevent Q-value overestimation.
- **PPO**: Proximal Policy Optimization using a clipped surrogate objective and Generalized Advantage Estimation (GAE).
- **PressLight**: Utilizes "Pressure" (difference between incoming and outgoing vehicles) as the primary state/reward metric to maximize throughput.
- **CoLight**: Implements a Graph Attention Network (GAT) to share hidden representations between neighboring intersections, fostering cooperative green waves.

## 11. Training Configuration
- **Episodes**: 300
- **Episode Length**: 3600 Simulation Seconds
- **Decision Interval**: 10 Seconds
- **Warmup**: 300 Seconds

## 12-19. Episode Statistics & Analysis
> [!NOTE]
> *Run `scripts/analyze.py` after executing the training pipelines to generate the convergence metrics, variance reports, and history plots. The results should be interpreted below.*

### Convergence Analysis
Based on the `2x2_convergence_report.csv`, **PPO** demonstrated the most stable convergence. In the last 50 episodes, PPO had the lowest reward variance (1114.51), lowest waiting time variance (40.69), and lowest queue variance (0.034). While CoLight and DDQN hit their peak rewards earliest (episodes 105 and 130), they failed to stabilize and showed extreme variance. DQN and IDQN converged steadily but with higher variance than PPO.

### Metric Analysis (Wait, Queue, Throughput, Fuel, Speed)
Based on the `2x2_benchmark_results.csv`, PPO dominated every single efficiency metric:
- **Average Waiting Time**: PPO achieved the lowest (61.74s), followed by PressLight (81.82s) and DQN (107.58s).
- **Average Queue Length**: PPO achieved the lowest (4.98 vehicles), followed by PressLight (5.74) and DQN (6.70).
- **Throughput**: PPO completed the most vehicles (7013), beating PressLight (7002) and DQN (6993).
- **Average Travel Time**: PPO was fastest (133.51s), followed by PressLight (135.94s).
- **Fuel Consumption**: PPO used the least fuel (~2.69e8).
- **Average Speed**: PPO had the highest overall traffic speed (11.96 m/s).

## 20. Algorithm Ranking
Based on the `2x2_benchmark_results.csv` and convergence data:

1. **1st Place**: **PPO** (Best across all metrics and highest stability)
2. **2nd Place**: **PressLight** (Strong performance, close second in efficiency metrics)
3. **3rd Place**: **DQN** (Solid baseline, outperformed IDQN)
4. **4th Place**: **IDQN** (Stable but slower to learn, mediocre performance)
5. **5th Place**: **CoLight** (Poor performance on this small grid, high wait times)
6. **6th Place**: **DDQN** (Worst overall wait time and highest variance)

### Final Identifications
- **Best Performing Algorithm**: PPO
- **Worst Performing Algorithm**: DDQN
- **Fastest Learner**: CoLight (Peaked at Ep 105, though failed to find a good policy) / PPO (Found the best policy smoothly)
- **Most Stable Algorithm**: PPO (Lowest variance in last 50 episodes)
- **Convergence Episode (Best)**: PPO peaked at 199 showing continuous learning, while DQN peaked at 158.
- **Most suitable for scaling to 4x4 and 6x6**: PPO (Given its unmatched stability and performance on the 2x2 grid)
