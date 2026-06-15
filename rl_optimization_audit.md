# RL Optimization Audit

## 1. Executive Summary
This audit evaluates the performance issues in the RL agents (DQN, DDQN, PPO) deployed for the Smart City Traffic Management System. The review covered the training loops, state/action spaces, reward functions, exploration strategies, checkpoint loading, and GPU optimization.

## 2. Agent-Specific Analysis

### 2.1 DQN (Best Performing but Reduces Throughput)
**Observation:** Reduced waiting time and queue length, but slight throughput reduction.
**Root Cause Analysis:** 
The reward function relies heavily on negative delta scaling for wait times and queues. Because the network learns that heavy traffic yields poor rewards regardless of actions, it converges on a policy that keeps secondary roads green indefinitely. This prevents small queues from forming on minor arteries but starves the main arteries, leading to a net reduction in overall vehicle throughput.
**Recommended Fix:** 
- Increase the positive reward coefficient for `throughput` to 2.0.
- Introduce a heavy `congestion_penalty` (e.g., -5.0) that punishes the agent when `total_queue > num_lanes * 10` to prevent starvation of main arteries.

### 2.2 DDQN (Behaves like Fixed-Time / Undertrained)
**Observation:** Produces results similar to Fixed-Time control; appears undertrained or not learning effectively.
**Root Cause Analysis:**
1. **Checkpointing Issue:** The `DDQNAgent.save()` method only dumps `self.model.state_dict()`. The target network weights (`self.target_model`) are not explicitly serialized. When reloading, the target network loses synchronization, causing massive TD-error spikes.
2. **Exploration vs. Episode Length:** `train_ddqn.py` runs for `num_seconds=10000` (almost 3 hours of simulation) per episode but only decays epsilon once per episode. This causes the agent to spend massive amounts of wall-clock time in high-exploration regimes without adequately exploiting learned policies.
**Recommended Fix:**
- Add `target_model_state_dict` to the serialization payload in the training loop.
- Reduce episode length to `3600` seconds (1 hour) to accelerate epsilon decay and increase the frequency of target updates relative to wall-clock time.

### 2.3 PPO (Worst Performing)
**Observation:** Higher waiting times and queues, negative rewards.
**Root Cause Analysis:**
1. **Value Truncation in Rollouts:** In `PPOAgent.replay()`, generalized advantage estimation (GAE) computes advantages over a rigid 2000-step batch. If the episode is not done at step 2000, it assumes the next state value is 0 (`next_values = 0.0`), heavily corrupting the value target for states near the end of the batch.
2. **Suboptimal Optimization Hyperparameters:** PPO clears its memory after every `replay()` but trains with only `k_epochs=4` and `lr=1e-4`. It extracts too little information from the gathered 2000 steps before discarding them.
**Recommended Fix:**
- Correct the GAE bootstrap logic: `next_values = state_values_all[t+1]` if `t == len(rewards) - 1`.
- Increase `k_epochs` to `10`.
- Increase learning rate to `3e-4`.

## 3. General Pipeline Evaluation

### Reward Function Shaping
The current objective is to minimize waiting time, queue length, emergency delay, while maximizing throughput.
**Current Mathematical Formulation:**
Delta representations (e.g., $queue\_delta = prev\_queue - current\_queue$) are correct because they bound the return. However, weights need precise alignment to match the 2:3:2:10 ratio requested by the system goals.

**Optimized Reward:**
$$ R = 2.0 \times \text{throughput} + 3.0 \times \text{queue\_delta} + 2.0 \times \text{wait\_delta} - 10.0 \times \text{emergency\_delay} - 5.0 \times \text{congestion\_penalty} $$

### GPU Optimization
- **CUDA Validation:** PyTorch implicitly falls back to CUDA if available, but some tensor operations inside the agents were causing DtoH (Device-to-Host) synchronization bottlenecks during memory extraction. Using `.to(self.device)` directly on batched `np.array` avoids scalar loop overhead. Mixed precision (AMP) is not strictly necessary for simple MLPs but can be added if batch sizes exceed 1024.
- **Fast Training Strategy:** Reducing episode durations to 3600 steps while increasing the total number of episodes (100 for DQN/DDQN, 150 for PPO) provides faster policy iteration and better validation monitoring.
