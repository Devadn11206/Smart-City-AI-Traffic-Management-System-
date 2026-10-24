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
- Begin with grid networks and extend evaluation to larger or irregular networks as experiments become available.

### 2. RL Benchmarking
The project benchmarks six approaches:

| Algorithm | Role in the project |
| :--- | :--- |
| **DQN** | Value-based deep reinforcement learning baseline |
| **DDQN** | Double DQN baseline designed to reduce Q-value overestimation |
| **PPO** | Policy-optimization baseline |
| **IDQN** | Improved/modified DQN implementation used in the project |
| **PressLight** | Traffic-signal control method based on the max-pressure concept |
| **CoLight** | Cooperative, network-level traffic signal control |

The exact implementation details, state/action definitions, reward functions, and hyperparameters should be documented alongside the experiment configuration.

### 3. PINN for Traffic-Flow Prediction
A Physics-Informed Neural Network learns from traffic data while incorporating physical constraints or governing relationships. In this project, the relevant traffic concepts include:
- Vehicle density
- Speed and traffic flow
- Queue evolution
- Conservation of vehicles

The Action-Aware PINN concept takes the current traffic state and a candidate signal action as inputs and predicts aspects of the future traffic state. Physics-based loss terms are used alongside data-driven learning so that predictions are encouraged to respect the selected traffic-flow constraints.

The exact equations, normalization, loss weights, training data, prediction horizon, and validation procedure should be specified in the implementation documentation.

### 4. PINN + RL Integration
PINN integration is being investigated with:
- **DQN + PINN**
- **DDQN + PINN**
- **PPO + PINN**
- **Primary research direction**: PINN + DDQN.

The research goal is to test whether action-conditioned, physics-based traffic predictions can improve RL learning or traffic-control performance compared with the corresponding baseline. Results should be reported only after controlled evaluation; adding a PINN does not by itself guarantee better performance.

### 5. Network Scaling
- Completed benchmark coverage reported for the project includes 2×2, 4×4, and 6×6 networks using the six RL algorithms.
- PINN integrations have been developed/tested on 6×6 networks for DQN, DDQN, and PPO.
- A 10×10 DDQN + PINN experiment has also been pursued.
- Larger and heterogeneous/irregular networks, area-level network composition, and generalization to unseen networks are intended extensions. Mark each experiment as completed, running, or planned in the results table when publishing updates.

---

## Evaluation Metrics

| Metric | Meaning | Preferred direction |
| :--- | :--- | :--- |
| **Waiting time** | Time vehicles spend waiting in the network or at intersections | Lower |
| **Queue length** | Number of vehicles waiting or stopped | Lower |
| **Throughput** | Number of vehicles successfully served or completing trips, according to the chosen definition | Higher |
| **Travel time** | Time taken for vehicles to complete their trips | Lower |
| **Reward** | RL feedback defined by the reward function | Interpret with the reward definition |
| **Learning stability / convergence** | How learning metrics evolve during training | Stable, reproducible learning; not inferred from loss alone |

Metric definitions, units, simulation duration, demand profile, random seeds, and aggregation method must remain consistent when comparing algorithms.

---

## MLOps Pipeline

MLOps tools are used to make experiments easier to track, compare, reproduce, and manage.

| Tool | Purpose | Use in this project |
| :--- | :--- | :--- |
| **MLflow** | Experiment tracking | Record configurations, parameters, metrics, and run comparisons |
| **Optuna** | Hyperparameter optimization | Search selected hyperparameters and optionally prune unpromising trials |
| **TensorBoard** | Training visualization | Inspect reward/loss curves and other logged training statistics |
| **DVC** | Data and artifact versioning | Version datasets and model artifacts/checkpoints alongside code references |
| **Git / GitHub** | Source-code management | Track code, configuration, documentation, and collaboration |

### Workflow Example:

```
                 Code + Configurations (Git)
                              |
                              v
                SUMO / RL / PINN Experiments
                              |
       +----------------------+----------------------+
       |                      |                      |
       v                      v                      v
MLflow: parameters      Optuna: hyperparameter  TensorBoard: training
and metrics             search                  curves
                              |
                              v
                     DVC: data & model
                     artifact versions
                              |
                              v
                  Reproducible evaluation
                      and comparison
```

*Note: Only claim that a tool tracks a particular item if the corresponding logging/versioning is implemented in the repository.*

---

## Current Project Status

Based on the current project notes:
- [x] Benchmarking six RL algorithms on 2×2, 4×4, and 6×6 networks.
- [x] PINN integration work for DQN, DDQN, and PPO on 6×6 networks.
- [x] Development/evaluation work for DDQN + PINN on a 10×10 network.
- [ ] Complete controlled comparisons across multiple random seeds.
- [ ] Finalize and validate the improved physics-guided DDQN design.
- [ ] Evaluate additional heterogeneous/irregular networks.
- [ ] Evaluate generalization to unseen networks.
- [ ] Finalize the paper with reproducible results and statistical analysis.

Update this checklist as experiments are completed. Do not label an experiment as converged or superior without evidence from its logs and repeatable evaluation.

---

## Results and Reporting

Final numerical results are intentionally not hard-coded in this README because they should come from the project's validated experiment logs.

When reporting results, include:
- Baseline and PINN-integrated algorithm names.
- Network size and topology.
- Traffic demand and simulation duration.
- Reward definition and metric definitions.
- Random seeds and number of independent runs.
- Mean and standard deviation (or confidence intervals).
- Training time and compute requirements, where relevant.

### Suggested Results Table:

| Network | Baseline | PINN-integrated method | Waiting time | Queue length | Throughput | Runs / seeds |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2×2** | *Add measured result* | If evaluated | — | — | — | — |
| **4×4** | *Add measured result* | If evaluated | — | — | — | — |
| **6×6** | *Add measured result* | Add measured result | — | — | — | — |
| **10×10** | *Add measured result* | DDQN + PINN, if validated | — | — | — | — |

*Replace the placeholders with values from actual logs before publishing. Do not use illustrative chart values as experimental evidence.*

---

## Technology Stack

- **Traffic simulation**: SUMO
- **Python-SUMO interface**: TraCI
- **Deep learning / PINN**: PyTorch
- **RL algorithms**: Project-specific implementations and/or selected RL libraries
- **Data processing**: NumPy, Pandas
- **Visualization**: Matplotlib / TensorBoard
- **Experiment tracking**: MLflow
- **Hyperparameter optimization**: Optuna
- **Data/model versioning**: DVC
- **Source control**: Git and GitHub

*Confirm exact package versions and libraries from the project environment before publishing a reproducible environment file.*

---

## Installation and Setup

The steps below are a general setup outline. Exact commands depend on the operating system, SUMO installation, and the dependencies used by the repository.

1. Install SUMO and verify that its command-line tools work.
2. Install a compatible Python version and create a virtual environment.
3. Install the project's Python dependencies.
4. Configure the SUMO executable/tools path and the TraCI Python module as required by the local installation.
5. Configure paths to the network files, route/demand files, trained models, and experiment output directories.
6. Run a small SUMO simulation first to verify the environment and TraCI connection.
7. Run a baseline RL experiment before enabling PINN integration.
8. Configure MLflow, TensorBoard, Optuna, and DVC according to the repository's scripts.

### Virtual Environment Setup:

```bash
# Create virtual environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Suggested Repository Structure

```
.
├── README.md
├── requirements.txt
├── configs/
├── sumo/
│   ├── networks/
│   └── routes/
├── src/
│   ├── environments/
│   ├── algorithms/
│   ├── pinn/
│   ├── integration/
│   └── evaluation/
├── experiments/
├── scripts/
├── tests/
├── docs/
└── .gitignore
```

---

## Reproducibility and Responsible Reporting

- Store experiment configurations with each run.
- Record random seeds and software versions.
- Keep large model checkpoints and datasets out of ordinary Git history when appropriate; use DVC or suitable artifact storage.
- Compare baseline and proposed methods under equivalent traffic demand and simulation settings.
- Report multiple independent runs where possible.
- Separate measured results from hypotheses, preliminary observations, and planned work.
- Do not publish sensitive data, credentials, local paths, or private configuration files.

---

## Limitations

- Simulation results may not transfer directly to real-world roads without further validation.
- PINN prediction quality depends on the data, physics constraints, model design, and validation procedure.
- Larger networks increase simulation and training cost.
- Performance can vary with random seeds, traffic demand, network topology, and reward design.
- A physics-guided method must be empirically compared with its baseline; improvement is a research hypothesis until verified.

---

## Future Scope

- Finalize an improved physics-guided DDQN method.
- Test additional heterogeneous and irregular traffic networks.
- Study transfer to unseen network layouts and demand profiles.
- Evaluate multi-intersection or area-level coordination.
- Perform multi-seed statistical comparisons and ablation studies.
- Improve experiment automation, reproducibility, and model artifact management through MLOps.
- Investigate real-world applicability only after suitable external validation.

---

## References

1. Wei, H. et al. (2019). *PressLight: Learning Max Pressure Control to Coordinate Traffic Signals in Arterial Network*. ACM SIGKDD. [https://doi.org/10.1145/3292500.3330949](https://doi.org/10.1145/3292500.3330949)
2. Wei, H. et al. (2019). *CoLight: Learning Network-level Cooperation for Traffic Signal Control*. ACM CIKM. [https://doi.org/10.1145/3357384.3357902](https://doi.org/10.1145/3357384.3357902)
3. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686–707. [https://doi.org/10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
4. van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-Learning*. AAAI. [https://doi.org/10.1609/aaai.v30i1.10295](https://doi.org/10.1609/aaai.v30i1.10295)
5. Krajzewicz, D., Erdmann, J., Behrisch, M., & Bieker, L. (2012). *Recent Development and Applications of SUMO – Simulation of Urban MObility*. International Journal On Advances in Systems and Measurements.
6. SUMO Documentation: [https://sumo.dlr.de/docs/](https://sumo.dlr.de/docs/)
7. MLflow Documentation: [https://mlflow.org/docs/latest/](https://mlflow.org/docs/latest/)
8. Optuna Documentation: [https://optuna.readthedocs.io/](https://optuna.readthedocs.io/)
9. TensorBoard Documentation: [https://www.tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard)
10. DVC Documentation: [https://dvc.org/doc](https://dvc.org/doc)

---

## Acknowledgements

This project is being developed as an academic research project at Chennai Institute of Technology. Add the final team members, supervisor details, and institutional acknowledgements as approved by the project team.

*Project status: Research and experimentation in progress. Claims of improved performance should be supported by validated results from the final experimental runs.*
