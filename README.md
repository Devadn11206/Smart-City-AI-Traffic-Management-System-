# Physics-Guided Reinforcement Learning for Adaptive Traffic Signal Optimization

Research project: Adaptive traffic signal control using Reinforcement Learning (RL), Physics-Informed Neural Networks (PINNs), SUMO simulation, and an MLOps workflow.

---

## Overview

Urban traffic conditions change continuously as vehicle demand, queues, speeds, and traffic density vary across intersections. Fixed-time traffic signals cannot always respond effectively to these changes. Reinforcement Learning (RL) can learn signal-control policies through interaction with a traffic simulator, but learning can be data-intensive and may not explicitly use the physical principles governing traffic flow.

This project investigates a physics-guided RL framework for adaptive traffic signal optimization. Traffic networks are simulated in SUMO, connected to Python through TraCI, and controlled using several RL algorithms. A Physics-Informed Neural Network (PINN) is investigated as a way to predict future traffic states using traffic data and traffic-flow physics, providing additional information to the RL learning process.

An MLOps workflow supports experiment tracking, hyperparameter tuning, training monitoring, and dataset/model versioning.

---

## Objectives

- Simulate and evaluate adaptive traffic signal control in SUMO.
- Benchmark six traffic-control algorithms across multiple network sizes.
- Investigate PINN integration with DQN, DDQN, and PPO.
- Focus research development on a physics-guided DDQN approach.
- Evaluate traffic performance using waiting time, queue length, throughput, travel time, and learning behavior.
- Extend evaluation toward larger and heterogeneous traffic networks.
- Improve experiment reproducibility and management with MLOps tools.

---

## Problem Statement

Traditional fixed-time and rule-based traffic signal systems use predefined timings and may not adapt well to changing traffic demand. RL-based systems can learn signal policies from experience, but often rely heavily on trial-and-error interaction and reward signals. This project investigates whether incorporating traffic-flow physics through a PINN can provide useful future-state information and improve the learning and performance of RL-based traffic signal control, particularly as network size and complexity increase.

---

## Existing System and Motivation

Common approaches include:
- **Fixed-time or rule-based control**: Uses predefined signal timings and has limited responsiveness to changing demand.
- **RL-based control**: Learns signal actions through interaction with a traffic environment and feedback from a reward function.
- **Network-level RL**: Approaches such as PressLight and CoLight consider traffic pressure or coordination among intersections.

The project explores a complementary direction: adding traffic-flow physics to RL learning and evaluating the approach across multiple algorithms and network sizes.

---

## Proposed Approach

The high-level workflow is:

```
                      SUMO Traffic Simulation
                                |
                                v
                    Traffic-State Observations
     (vehicle count, queue, density, speed, phase, lane features)
                                |
                +---------------+---------------+
                |                               |
                v                               v
            RL Agent                    Action-Aware PINN
     (DQN / DDQN / PPO /           (Traffic data + flow physics)
  IDQN / PressLight / CoLight)                  |
                |                               v
                |                 Predicted future traffic state
                |              (e.g., density, speed, queue, flow)
                +---------------+---------------+
                                |
                                v
                     Traffic-signal decision
                                |
                                v
                         SUMO environment
                                |
                                v
                    Metrics and reward feedback
                                |
                                v
                          MLOps workflow
```

> **Important**: The precise PINN-to-RL integration differs by algorithm. The project investigates PINN integrations with DQN, DDQN, and PPO; the design should not be interpreted as one identical integration mechanism for every agent.

---

## Methodology

### 1. Traffic Simulation — SUMO and TraCI
- Build and run traffic networks in SUMO.
- Use TraCI to exchange traffic states and signal-control actions between Python and SUMO.
- Collect traffic observations such as vehicle counts, queue lengths, density, speed, traffic-light phase, and lane features.
- Simulation duration: **3600 seconds per episode** across all evaluated experiments.
- Networks: 2x2 (4 intersections), 4x4 (16 intersections), 6x6 (36 intersections), 10x10 (100 intersections).

### 2. RL Benchmarking
The project benchmarks six approaches:

| Algorithm | Role in the project |
| :--- | :--- |
| **DQN** | Value-based deep reinforcement learning baseline |
| **DDQN** | Double DQN baseline designed to reduce Q-value overestimation |
| **PPO** | Policy-optimization baseline |
| **IDQN** | Independent DQN — each intersection has its own isolated network and replay buffer |
| **PressLight** | Traffic-signal control method based on the max-pressure concept |
| **CoLight** | Cooperative, network-level traffic signal control using Graph Attention Networks |

**Reward function (2x2 benchmark):** `Reward = Throughput - 0.05 x Queue - 0.001 x WaitTime + SwitchPenalty(-0.1)`

**Reward function (4x4 benchmark):** `Reward = (Throughput - 0.1 x Queue - 0.1 x SwitchPenalty) / 100`

**Reward function (PINN-integration packages):** Composite multi-agent reward averaged across all traffic signal agents.

### 3. PINN for Traffic-Flow Prediction
A Physics-Informed Neural Network learns from traffic data while incorporating physical constraints or governing relationships. In this project, the relevant traffic concepts include:
- Vehicle density
- Speed and traffic flow
- Queue evolution
- Conservation of vehicles

The Action-Aware PINN concept takes the current traffic state and a candidate signal action as inputs and predicts aspects of the future traffic state. Physics-based loss terms are used alongside data-driven learning so that predictions are encouraged to respect the selected traffic-flow constraints.

Pretrained PINN ensemble weights are available for both 6x6 (`pinn_ensemble_pretrained_6x6.pth`) and 10x10 (`pinn_ensemble_pretrained_10x10.pth`) networks. Training datasets are versioned with DVC (`pinn_dataset_6x6.pt`, `pinn_dataset_10x10.pt`).

### 4. PINN + RL Integration
PINN integration has been evaluated with:
- **DQN + PINN (MBPO)** on the 6x6 network — ~154 episodes completed
- **DDQN + PINN (PI-DDQN)** on the 6x6 network — ~153 episodes completed
- **PPO + PINN** on the 6x6 network — ~150 episodes completed
- **DDQN + PINN (PI-DDQN)** on the 10x10 network — ~157 episodes completed (stopped early due to GPU VRAM limits)

**Primary research direction:** PINN + DDQN.

### 5. Network Scaling
- 2x2, 4x4, and 6x6 benchmarks: six baseline algorithms, all completed.
- 6x6 PINN integrations: DQN+PINN, DDQN+PINN, PPO+PINN — all stopped at ~150 episodes due to resource constraints.
- 10x10 DDQN+PINN: stopped at episode 152 (GPU VRAM limitation confirmed by checkpoints `ckpt_ep150.pt` and `ckpt_ep152.pt`).
- Irregular/heterogeneous networks: code infrastructure exists (`sumo_network_irregular/`); training not yet completed.

---

## Evaluation Metrics

| Metric | Meaning | Preferred direction |
| :--- | :--- | :--- |
| **Avg Waiting Time (s)** | Mean time vehicles spend waiting at intersections per episode | Lower |
| **Avg Queue Length (vehicles)** | Mean number of vehicles waiting or stopped per episode | Lower |
| **Throughput (vehicles)** | Number of vehicles completing trips per episode | Higher |
| **Avg Travel Time (s)** | Mean time per completed trip | Lower |
| **Reward** | RL feedback (reward function definition differs per benchmark; see Methodology) | Interpret with the reward definition |
| **Learning stability** | Variance of reward and metrics in the final 50 episodes | Stable, low variance |

> **Critical note on simulation contexts:** The **benchmark_research** experiments (2x2, 4x4, 6x6) use large SUMO networks with all intersections active and produce waiting times in the **thousands of seconds**. The **PINN integration packages** use a custom lightweight SUMO environment and produce waiting times in the **tens to hundreds of seconds**. Results from these two contexts **cannot be directly compared**.

---

## Experimental Findings

> This section is generated from verified analysis of experiment logs. Results reflect training metrics from single-seed runs; no multi-seed statistical validation is currently available.

### Key Findings

**1. Baseline benchmark (2x2 network):**
PPO achieved the best overall performance: lowest waiting time (61.7 s), lowest queue length (5.0 vehicles), and highest throughput (7013 vehicles/episode) in the final 50 training episodes. PressLight was second-best on most metrics. DDQN and CoLight showed high variance and poor convergence on the small 2x2 grid.

**2. Baseline benchmark (4x4 network):**
PPO achieved the lowest waiting time (12,176 s avg), though all algorithms showed high absolute wait times at this scale. PressLight achieved the highest throughput (17,926 vehicles) and lowest queue length. All algorithms showed high variance at 4x4 scale.

**3. Baseline benchmark (6x6 network):**
PPO remained best overall: lowest waiting time (11,278 s avg), highest throughput (49,920 vehicles). PressLight had the highest waiting time and lowest throughput, performing worst at this scale. DQN and DDQN were comparable on most metrics.

**4. PINN integration results (6x6 and 10x10):**
In all four evaluated PINN-integrated variants, the **PINN-integrated method did not outperform its corresponding baseline** on waiting time, queue length, or travel time in the available training data. Throughput results were mixed: PPO+PINN showed +0.5% throughput improvement; PI-DDQN on 10x10 showed +3.7% throughput improvement. All PINN runs were stopped between episodes 150-157. Results at this stage reflect **training-in-progress snapshots**, not final converged performance.

**5. Single seed only:** All experiments used a single random seed (seed=42). No multi-seed statistical testing has been conducted. Reported standard deviations reflect within-run episode-to-episode variability, not cross-seed uncertainty.

**6. Missing experiments:** 2x2 and 4x4 PINN integrations, irregular/heterogeneous network experiments, and multi-seed validation runs have not been conducted.

> **Scientific conclusion:** Whether PINN-integrated methods outperform baselines **cannot be verified from the currently available data**, since all PINN runs were interrupted before reaching the 200-episode target and no multi-seed comparison exists. The current evidence does not support a claim of improvement.

---

## Table A: Experimental Coverage and Status

| Network Size | Baseline Algorithms | PINN Integrations | Status |
| :--- | :--- | :--- | :--- |
| **2x2** | DQN, DDQN, PPO, IDQN, PressLight, CoLight | None evaluated | Baseline complete (~200 eps each, 1 seed) |
| **4x4** | DQN, DDQN, PPO, IDQN, PressLight, CoLight | None evaluated | Baseline complete (~200 eps each, 1 seed) |
| **6x6** | DQN, DDQN, PPO, IDQN, PressLight, CoLight | DQN+PINN, DDQN+PINN, PPO+PINN | Baseline complete; PINN runs stopped ~150 eps (VRAM) |
| **10x10** | DDQN | DDQN+PINN | Baseline complete (~203 eps); PINN stopped at ep 152 (VRAM - confirmed) |
| **Irregular** | Not run | Not run | Planned — network files exist, no training data |

---

## Table B: Quantitative Results — Benchmark Baselines

All values are means over the **final 50 training episodes** (single seed = 42, 3600 s/episode).

### 2x2 Network
*Reward: `Throughput - 0.05 x Queue - 0.001 x Wait + SwitchPenalty`. Source: `benchmark_research/2x2_benchmark/results/`*

| Algorithm | Eps | Avg Wait (s) +/- sd | Avg Queue +/- sd | Throughput +/- sd | Avg Travel (s) +/- sd | Reward +/- sd |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO | 200 | 61.7 +/- 6.4 | 5.0 +/- 0.2 | 7013 +/- 34 | 133.5 +/- 1.1 | 6472 +/- 33 |
| PressLight | 200 | 81.8 +/- 25.8 | 5.7 +/- 0.5 | 7003 +/- 31 | 135.9 +/- 2.0 | -10404 +/- 719 |
| DQN | 200 | 107.6 +/- 124.9 | 6.7 +/- 1.5 | 6994 +/- 35 | 139.2 +/- 3.8 | 6307 +/- 281 |
| IDQN | 289 | 133.1 +/- 23.8 | 7.6 +/- 0.6 | 6993 +/- 27 | 143.2 +/- 2.6 | 6202 +/- 71 |
| CoLight | 200 | 278.6 +/- 65.8 | 10.8 +/- 1.2 | 6974 +/- 35 | 154.4 +/- 4.3 | 5784 +/- 165 |
| DDQN | 200 | 281.1 +/- 856.8 | 9.2 +/- 8.3 | 6971 +/- 134 | 145.4 +/- 12.0 | 5897 +/- 1799 |

> Note: Reward scales differ across algorithms — PressLight and CoLight use different internal reward definitions. Do not rank algorithms by reward value across different reward functions.

### 4x4 Network
*Reward (RL agents): `(Throughput - 0.1 x Queue - 0.1 x Switch) / 100`. Source: `benchmark_research/4x4_benchmark/results/4x4/`*

| Algorithm | Eps | Avg Wait (s) +/- sd | Avg Queue +/- sd | Throughput +/- sd | Avg Travel (s) +/- sd | Reward +/- sd |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO | 200 | 12176 +/- 2601 | 127.8 +/- 6.2 | 16459 +/- 1949 | 992.3 +/- 60.1 | -513 +/- 50 |
| PressLight | 200 | 13740 +/- 4027 | 125.0 +/- 9.5 | 17926 +/- 2707 | 896.5 +/- 85.9 | -332821 +/- 12374 |
| DQN | 200 | 15267 +/- 3013 | 126.7 +/- 7.6 | 16386 +/- 2359 | 871.8 +/- 71.8 | -509 +/- 61 |
| DDQN | 200 | 15268 +/- 3177 | 127.5 +/- 7.6 | 16446 +/- 1856 | 875.0 +/- 82.7 | -513 +/- 56 |
| IDQN | 230 | 15501 +/- 3055 | 128.5 +/- 7.3 | 16751 +/- 2288 | 862.2 +/- 63.4 | -515 +/- 59 |
| CoLight | 200 | 17035 +/- 2489 | 134.0 +/- 5.6 | 14235 +/- 1615 | 913.6 +/- 74.8 | -46250 +/- 6629 |

### 6x6 Network
*Source: `benchmark_research/6x6_benchmark/results/`*

| Algorithm | Eps | Avg Wait (s) +/- sd | Avg Queue +/- sd | Throughput +/- sd | Avg Travel (s) +/- sd | Reward +/- sd |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO | 260 | 11278 +/- 1502 | 129.5 +/- 5.4 | 49920 +/- 3199 | 977.9 +/- 44.7 | -1048 +/- 89 |
| DQN | 203 | 13208 +/- 1420 | 133.5 +/- 5.4 | 44917 +/- 3215 | 1020.1 +/- 60.5 | -1146 +/- 87 |
| DDQN | 200 | 13013 +/- 1039 | 134.1 +/- 4.0 | 45234 +/- 2533 | 994.3 +/- 56.9 | -1151 +/- 63 |
| CoLight | 247 | 13705 +/- 1649 | 136.1 +/- 5.4 | 43484 +/- 3032 | 1028.3 +/- 47.9 | -1190 +/- 90 |
| IDQN | 200 | 13873 +/- 1579 | 136.5 +/- 5.4 | 44408 +/- 3679 | 1011.7 +/- 48.7 | -1186 +/- 89 |
| PressLight | 213 | 14982 +/- 1472 | 139.1 +/- 4.6 | 40289 +/- 2381 | 1036.9 +/- 65.1 | -8487 +/- 357 |

---

## Table C: PINN Integration Results

> **Important context:** These experiments use a **different, lighter SUMO environment** than the benchmark experiments above. Waiting times are in the range of 40-110 seconds here vs. approximately 11,000-15,000 seconds in the full 6x6 benchmark. The two sets of results are **not directly comparable**. All PINN runs were stopped between episodes 150-157.

### 6x6 Network — PINN Package (single seed, 3600 s/episode)
*Sources: `dqn_pinn_package/training/`, `ddqn_pinn_package/`, `ppo_pinn_package/training/`*

| Algorithm | PINN | Eps | Avg Wait (s) +/- sd | Avg Queue +/- sd | Throughput +/- sd | Avg Travel (s) +/- sd | Reward +/- sd | Status |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| DQN | No | 204 | 96.0 +/- 39.7 | 15.4 +/- 3.2 | 625 +/- 20 | 116.1 +/- 6.0 | -129.9 +/- 35.9 | Complete |
| DQN+PINN (MBPO) | Yes | 154 | 109.4 +/- 26.4 | 16.8 +/- 1.8 | 625 +/- 26 | 116.8 +/- 3.7 | -143.5 +/- 22.0 | Stopped ~ep 154 |
| DDQN | No | 200 | 42.4 +/- 1.5 | 10.8 +/- 0.2 | 634 +/- 23 | 109.3 +/- 3.0 | -79.8 +/- 1.8 | Complete |
| PI-DDQN | Yes | 153 | 55.0 +/- 8.2 | 12.3 +/- 0.4 | 625 +/- 22 | 112.2 +/- 3.0 | -94.0 +/- 5.6 | Stopped ~ep 153 |
| PPO | No | 200 | 37.2 +/- 1.3 | 10.5 +/- 0.3 | 615 +/- 23 | 110.9 +/- 3.1 | -75.7 +/- 1.9 | Complete |
| PPO+PINN | Yes | 150 | 68.5 +/- 1.6 | 13.8 +/- 0.2 | 617 +/- 24 | 113.1 +/- 3.5 | -108.5 +/- 1.6 | Stopped ~ep 150 |

### 10x10 Network — PINN Package (single seed, 3600 s/episode)
*Source: `ddqn_with_pinn_for_10x10/`. Checkpoints confirm last episode = 152.*

| Algorithm | PINN | Eps | Avg Wait (s) +/- sd | Avg Queue +/- sd | Throughput +/- sd | Avg Travel (s) +/- sd | Reward +/- sd | Status |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| DDQN | No | 203 | 45.5 +/- 3.0 | 11.5 +/- 0.5 | 776 +/- 25 | 119.2 +/- 3.8 | -28.5 +/- 1.4 | Complete |
| PI-DDQN | Yes | 157 | 80.2 +/- 6.2 | 17.7 +/- 0.4 | 805 +/- 28 | 131.1 +/- 3.9 | -45.2 +/- 1.7 | Stopped ep 152 (VRAM) |

---

## Table D: Baseline vs PINN Comparison

Relative change calculated over the final 50 episodes of each run. Results are **descriptive only** — single seed, PINN runs incomplete.

- Lower-is-better (wait, queue, travel): Improvement% = (Baseline - PINN) / Baseline x 100
- Higher-is-better (throughput): Improvement% = (PINN - Baseline) / Baseline x 100

| Network | Baseline | PINN Method | Metric | Baseline | PINN | Delta% | Notes |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| 6x6 | DQN | DQN+PINN (MBPO) | Avg Wait (s) | 96.0 | 109.4 | -14.0% | PINN worse |
| 6x6 | DQN | DQN+PINN (MBPO) | Avg Queue | 15.4 | 16.8 | -8.7% | PINN worse |
| 6x6 | DQN | DQN+PINN (MBPO) | Throughput | 625 | 625 | -0.0% | Negligible |
| 6x6 | DQN | DQN+PINN (MBPO) | Avg Travel (s) | 116.1 | 116.8 | -0.6% | PINN marginal worse |
| 6x6 | DDQN | PI-DDQN | Avg Wait (s) | 42.4 | 55.0 | -29.7% | PINN worse |
| 6x6 | DDQN | PI-DDQN | Avg Queue | 10.8 | 12.3 | -14.0% | PINN worse |
| 6x6 | DDQN | PI-DDQN | Throughput | 634 | 625 | -1.4% | PINN marginally worse |
| 6x6 | DDQN | PI-DDQN | Avg Travel (s) | 109.3 | 112.2 | -2.6% | PINN worse |
| 6x6 | PPO | PPO+PINN | Avg Wait (s) | 37.2 | 68.5 | -84.1% | PINN substantially worse |
| 6x6 | PPO | PPO+PINN | Avg Queue | 10.5 | 13.8 | -31.7% | PINN worse |
| 6x6 | PPO | PPO+PINN | Throughput | 615 | 617 | +0.5% | Marginal (not significant) |
| 6x6 | PPO | PPO+PINN | Avg Travel (s) | 110.9 | 113.1 | -2.0% | PINN worse |
| 10x10 | DDQN | PI-DDQN | Avg Wait (s) | 45.5 | 80.2 | -76.3% | PINN substantially worse |
| 10x10 | DDQN | PI-DDQN | Avg Queue | 11.5 | 17.7 | -53.3% | PINN substantially worse |
| 10x10 | DDQN | PI-DDQN | Throughput | 776 | 805 | +3.7% | Marginal improvement |
| 10x10 | DDQN | PI-DDQN | Avg Travel (s) | 119.2 | 131.1 | -10.0% | PINN worse |

> **Scientific interpretation:** In the available data, PINN-integrated methods have not yet demonstrated a performance advantage over their baselines on waiting time, queue length, or travel time. PINN runs were stopped at approximately episode 150 due to GPU VRAM constraints while baselines ran to 200+ episodes. This asymmetry makes direct comparison difficult. The throughput improvement for 10x10 PI-DDQN (+3.7%) is of interest but insufficient to claim superiority without completing the full training run and multi-seed validation.

---

## Table E: MLOps Tool Status

| Tool | Intended Purpose | Verified Usage | Evidence / Status |
| :--- | :--- | :--- | :--- |
| **MLflow** | Experiment and metric tracking | Verified — 14 runs across 3 experiments | `mlflow.db` (5 MB); experiments: `Default`, `Baseline_6x6_Benchmarks`, `Physics_Informed_RL`; 14 FINISHED runs (DQN_6x6, DDQN_6x6, PPO_6x6, IDQN_6x6, PRESSLIGHT_6x6, COLIGHT_6x6, BASE-DQN-6x6, BASE-DDQN-6x6, BASE-PPO-6x6, PINN-DDQN-6x6, PINN-MBPO-6x6, PINN-PPO-6x6, BASE-DDQN-10x10, PINN-DDQN-10x10) |
| **Optuna** | Hyperparameter optimization | Verified — 1 study, 14 complete trials | `optuna.db` (123 KB); study `Traffic_Optimization_Benchmarks`; 14 trials covering DQN/DDQN/PPO/IDQN/PressLight/CoLight/MBPO on 6x6 and 10x10 networks; objective = avg_waiting_time |
| **TensorBoard** | Training and loss monitoring | Verified — 14 event files present | `tensorboard_logs/` contains event files for 6 baseline 6x6 algorithms + 6 PINN-package 6x6 runs + 2 10x10 runs; generated via `make_tensorboard_live.py` from CSV export |
| **DVC** | Dataset/model artifact versioning | Verified — 3 artifacts DVC-tracked | `.dvc/config` uses local remote (`../dvc_remote`); tracked: `pinn_dataset_6x6.pt` (71 MB), `pinn_dataset_10x10.pt` (105 MB), `pinn_ensemble_pretrained_10x10.pth` (273 KB); DVC cache populated locally |
| **Git / GitHub** | Source-code management | Verified | Repository: [Devadn11206/Smart-City-AI-Traffic-Management-System-](https://github.com/Devadn11206/Smart-City-AI-Traffic-Management-System-); active commit history on `main` |

---

## Current Project Status

- [x] Benchmarking six RL algorithms on 2x2 networks (all 6 algorithms, ~200 episodes each)
- [x] Benchmarking six RL algorithms on 4x4 networks (all 6 algorithms, ~200 episodes each)
- [x] Benchmarking six RL algorithms on 6x6 networks (all 6 algorithms, ~200-260 episodes each)
- [x] PINN integration: DQN+PINN (MBPO) on 6x6 network (~154 episodes, interrupted)
- [x] PINN integration: DDQN+PINN (PI-DDQN) on 6x6 network (~153 episodes, interrupted)
- [x] PINN integration: PPO+PINN on 6x6 network (~150 episodes, interrupted)
- [x] PINN integration: DDQN+PINN on 10x10 network (~157 episodes, stopped ep 152 - confirmed by checkpoints)
- [x] MLOps pipeline: MLflow, Optuna, TensorBoard, DVC all verified and operational
- [ ] Complete PINN runs to 200+ episodes (all stopped ~150 due to GPU VRAM limits)
- [ ] Multi-seed validation (minimum 3 seeds) for all PINN vs baseline comparisons
- [ ] Evaluate irregular/heterogeneous networks (infrastructure exists, no training data)
- [ ] Ablation studies on PINN loss weight, prediction horizon, and network architecture
- [ ] Statistical significance testing for any claimed improvements
- [ ] Finalize paper with reproducible, validated results

---

## Results Source Files

| Experiment | Source File |
| :--- | :--- |
| 2x2 benchmark (all algorithms) | `benchmark_research/2x2_benchmark/results/{algorithm}/2x2_episode_metrics.csv` |
| 4x4 benchmark (all algorithms) | `benchmark_research/4x4_benchmark/results/4x4/{algorithm}/4x4_episode_metrics.csv` |
| 6x6 benchmark (all algorithms) | `benchmark_research/6x6_benchmark/results/{algorithm}/6x6_episode_metrics.csv` |
| DQN baseline (PINN pkg) | `dqn_pinn_package/training/dqn_6x6_metrics.csv` |
| DQN+PINN/MBPO | `dqn_pinn_package/training/pi_mbpo_6x6_metrics.csv` |
| DDQN baseline (PINN pkg) | `ddqn_pinn_package/ddqn_baseline_metrics.csv` |
| PI-DDQN (6x6) | `ddqn_pinn_package/pi_ddqn_metrics.csv` |
| PPO baseline (PINN pkg) | `ppo_pinn_package/training/ppo_6x6_metrics.csv` |
| PPO+PINN (6x6) | `ppo_pinn_package/training/ppo_pinn_6x6_metrics.csv` |
| DDQN baseline (10x10) | `ddqn_with_pinn_for_10x10/ddqn_baseline_10x10_metrics.csv` |
| PI-DDQN (10x10) | `ddqn_with_pinn_for_10x10/pi_ddqn_10x10_metrics.csv` |
| Machine-readable summary | `results/experiment_summary.csv` |

---

## Technology Stack

- **Traffic simulation**: SUMO
- **Python-SUMO interface**: TraCI
- **Deep learning / PINN**: PyTorch
- **RL algorithms**: Project-specific implementations (DQN, DDQN, PPO, IDQN, PressLight, CoLight)
- **Data processing**: NumPy, Pandas
- **Visualization**: Matplotlib / TensorBoard
- **Experiment tracking**: MLflow (`mlflow.db`)
- **Hyperparameter optimization**: Optuna (`optuna.db`)
- **Data/model versioning**: DVC (local remote at `../dvc_remote`)
- **Source control**: Git and GitHub

---

## Installation and Setup

1. Install SUMO and verify that its command-line tools work.
2. Install a compatible Python version and create a virtual environment.
3. Install the project Python dependencies.
4. Configure the SUMO executable/tools path and TraCI Python module.
5. Configure paths to network files, route/demand files, trained models, and output directories.
6. Run a small SUMO simulation to verify the environment and TraCI connection.
7. Run a baseline RL experiment before enabling PINN integration.
8. Configure MLflow, TensorBoard, Optuna, and DVC according to the repository scripts.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# MLOps tools
python make_mlflow_live.py
python make_tensorboard_live.py
python make_optuna_live.py
```

---

## Reproducibility and Responsible Reporting

- Store experiment configurations with each run.
- Record random seeds and software versions.
- Keep large model checkpoints and datasets out of ordinary Git history; use DVC.
- Compare baseline and proposed methods under equivalent traffic demand and simulation settings.
- Report multiple independent runs where possible.
- Separate measured results from hypotheses, preliminary observations, and planned work.

---

## Limitations

- All reported results are from a **single random seed** (seed=42). Cross-seed variance is unknown.
- **PINN runs were interrupted at approximately episode 150** (all four variants). Results may not represent converged performance.
- Simulation results may not transfer directly to real-world roads without further validation.
- Larger networks increase simulation and training cost; 10x10 PINN runs stopped due to GPU VRAM exhaustion.
- Reward functions differ between benchmark suites and PINN integration packages — direct cross-table comparisons are invalid.
- The current evidence does not support a claim of PINN improvement over baseline RL.

---

## Future Scope

- Complete PINN runs to 200+ episodes with sufficient GPU VRAM (or mixed-precision training).
- Multi-seed validation for all PINN vs baseline comparisons.
- Test additional heterogeneous and irregular traffic networks.
- Ablation studies on PINN loss weight, prediction horizon, and ensemble size.
- Improve experiment automation and reproducibility through MLOps.
- Investigate real-world applicability only after suitable external validation.

---

## References

1. Wei, H. et al. (2019). *PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network*. ACM SIGKDD. https://doi.org/10.1145/3292500.3330949
2. Wei, H. et al. (2019). *CoLight: Learning Network-level Cooperation for Traffic Signal Control*. ACM CIKM. https://doi.org/10.1145/3357384.3357902
3. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks*. Journal of Computational Physics, 378, 686-707. https://doi.org/10.1016/j.jcp.2018.10.045
4. van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-Learning*. AAAI. https://doi.org/10.1609/aaai.v30i1.10295
5. Krajzewicz, D. et al. (2012). *Recent Development and Applications of SUMO*. International Journal On Advances in Systems and Measurements.
6. SUMO Documentation: https://sumo.dlr.de/docs/
7. MLflow Documentation: https://mlflow.org/docs/latest/
8. Optuna Documentation: https://optuna.readthedocs.io/
9. TensorBoard Documentation: https://www.tensorflow.org/tensorboard
10. DVC Documentation: https://dvc.org/doc

---

## Acknowledgements

This project is being developed as an academic research project at Chennai Institute of Technology. Add the final team members, supervisor details, and institutional acknowledgements as approved by the project team.

*Project status: Research and experimentation in progress. Claims of improved performance should be supported by validated results from the final experimental runs.*
