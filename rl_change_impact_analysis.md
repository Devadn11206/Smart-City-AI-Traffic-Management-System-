# RL Change Impact Analysis

## 1. DQN Epsilon-Decay Fix
**Modification**: Move epsilon decay from the per-step `replay()` method to the per-episode level in `train_dqn.py`.
**Expected Effect**: The agent will explore over a longer period (150 episodes) instead of decaying exploration to zero within the first 1000 steps. This will allow it to discover higher-throughput policies rather than converging on a premature local minimum.
**Risk Level**: Low.
**Rollback Strategy**: Revert `dqn_agent.py` to call `self.epsilon *= self.epsilon_decay` inside `replay()`.

## 2. DDQN Exploration Fix
**Modification**: Change `epsilon_decay` from 0.999 to 0.98. The Double-Q target calculation is already correct in code. We will also add action distribution logging.
**Expected Effect**: The agent will properly exploit learned Q-values rather than randomly exploring throughout the entire run. This will prevent DDQN from performing identically to Fixed-Time control. Action logging will provide observability into mode collapse.
**Risk Level**: Low.
**Rollback Strategy**: Change `epsilon_decay` back to 0.999.

## 3. PPO Stabilization
**Modification**: Implement mini-batch updates in `replay()` (batch size 256), and call `agent.replay()` every 2000 steps instead of every 10,000 steps.
**Expected Effect**: Massive improvement in PPO stability. Avoids catastrophic forgetting and NaN loss values. The agent will no longer produce large negative rewards.
**Risk Level**: Medium (PPO math is sensitive to dimensions and shapes during mini-batching).
**Rollback Strategy**: Revert `ppo_agent.py` to process the entire memory array at once. Revert `train_ppo.py` to call `replay()` at the end of the episode.

## 4. Reward Redesign
**Modification**: Replace the deeply normalized reward structure with `2 * throughput - 3 * queue_delta - 2 * wait_delta - 10 * emergency_delay - congestion_penalty`.
**Expected Effect**: Clearer gradients for all RL agents. Using deltas prevents massive absolute negative rewards early on, while explicitly penalizing emergency vehicle delays ensures compliance with emergency routing logic.
**Risk Level**: Medium (Magnitude of reward might require tuning if `queue_delta` dwarfs `throughput`).
**Rollback Strategy**: Revert `_compute_reward` in `sumo_environment.py` to the previous complex mathematical expression.
