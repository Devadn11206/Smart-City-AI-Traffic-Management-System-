# Traffic Signal Control 4x4 Benchmark Final Report

This report summarizes the performance and convergence of IDQN, DQN, DDQN, PPO, PressLight, and CoLight algorithms on the 4x4 grid under standard Reward Version A.

## 1. Executive Summary
Under Reward Version A:
$$\text{Reward} = \frac{\text{Throughput} - 0.1 \times \text{QueueLength} - 0.1 \times \text{SwitchPenalty}}{100}$$
the waiting time penalty has been completely removed for RL agents (DQN, IDQN, DDQN, PPO), leading to stable, bounded reward signals and preventing reward explosion. Multi-criteria rankings evaluate algorithms based on throughput, queue lengths, average wait times, travel times, fuel efficiency, and speed.

## 2. Final Algorithm Rankings (Last 50 Episodes Average)

| Rank | Algorithm | Score | Avg Wait (s) | Avg Queue | Throughput | Avg Travel Time (s) | Fuel (ml) | Avg Speed (m/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | DQN | 16.0 | 15266.84 | 126.72 | 16385.88 | 871.76 | 4.23e+09 | 2.79 |
| 2 | PRESSLIGHT | 16.0 | 13739.67 | 125.00 | 17926.24 | 896.50 | 4.23e+09 | 2.44 |
| 3 | PPO | 20.0 | 12175.79 | 127.79 | 16458.70 | 992.29 | 4.21e+09 | 2.59 |
| 4 | DDQN | 20.0 | 15267.57 | 127.48 | 16446.38 | 874.97 | 4.27e+09 | 2.76 |
| 5 | IDQN | 21.0 | 15501.24 | 128.45 | 16751.48 | 862.16 | 4.29e+09 | 2.76 |
| 6 | COLIGHT | 33.0 | 17035.10 | 134.03 | 14235.32 | 913.58 | 4.32e+09 | 2.63 |

## 3. Metric Highlights
*   **Best Performing Algorithm**: DQN
*   **Most Stable Algorithm**: PPO
*   **Fastest Converging Algorithm**: PPO
*   **Lowest Queue Length**: PRESSLIGHT
*   **Lowest Travel Time**: IDQN
*   **Highest Throughput**: PRESSLIGHT
*   **Lowest Fuel Consumption**: PPO

## 4. Convergence Analysis (Last 50 Episodes Variance)

| Algorithm | Best Episode | Reward Var (50) | Wait Var (50) | Queue Var (50) |
| --- | --- | --- | --- | --- |
| IDQN | 45 | 3433.06 | 9331049.46 | 52.6593 |
| DQN | 149 | 3738.57 | 9081109.65 | 58.4834 |
| DDQN | 117 | 3131.50 | 10092613.09 | 57.4733 |
| PPO | 1 | 2456.12 | 6762622.89 | 38.8437 |
| PRESSLIGHT | 159 | 153122214.48 | 16217749.98 | 90.8269 |
| COLIGHT | 20 | 43942027.29 | 6193836.74 | 30.8346 |

## 5. Scaling Recommendations

### Scaling from 4x4 to 6x6 Grid
- **Parameter Sharing**: DQN and PPO utilize parameter sharing, enabling them to scale naturally as they process intersections independently using identical weights.
- **Graph Attention (CoLight)**: CoLight's Graph Attention Network (GAT) becomes critical on a 6x6 grid. It coordinates neighboring intersections to create "green waves" and prevent traffic bottlenecks.
- **Action Decoupling**: Introduce minor action offsets or coordination protocols to avoid concurrent left-turn locks.

### Scaling from 6x6 to City-Scale Network
- **Hierarchical MARL**: Divide city-scale networks into regional subgrids (e.g., zones/districts). Regional subnets (DQN/PPO) control local lights, while a high-level router manages macro flow.
- **Asynchronous Optimization**: Distribute TraCI simulation instances across parallel worker nodes to handle computational loads.
- **Memory Buffer Management**: Implement prioritized replay buffers to focus learning on congested bottlenecks.

### Integration with the Hybrid PPO + PINN + ACO + Emergency Corridor System
- **PINN Traffic Prediction**: Use Physics-Informed Neural Networks (PINN) to predict short-term traffic state transitions ($s 	o s'$) and feed predictions as auxiliary states to PPO.
- **ACO Routing Coordination**: Ant Colony Optimization (ACO) routes vehicles dynamically. The RL traffic lights should prioritize green corridors along high-pheromone ACO paths.
- **Emergency Vehicle Priority**: Dynamically adjust the reward function weight (e.g. increase QueueLength penalty on EV approach lanes) and clear adjacent lane queues (Emergency Corridor) to allow visual lane yielding.
