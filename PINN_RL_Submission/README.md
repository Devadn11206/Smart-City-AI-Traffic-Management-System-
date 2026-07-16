# PINN-Assisted Reinforcement Learning for Traffic Signal Control

This folder contains the complete codebase, results, trained models, dataset, and generated reports for the Physics-Informed Neural Network (PINN) augmented Proximal Policy Optimization (PPO) experiments.

## ?? Folder Structure & File Descriptions

### 1. code\ (Source Code)
*   **code\pinn\**: Contains the core PyTorch implementation of the Physics-Informed Neural Network. This includes the dataset generation, model architecture, and training scripts.
*   **code\algorithms\ppo_pinn.py**: The augmented Reinforcement Learning algorithm. This file modifies standard PPO to intercept the environment state and feed it through the frozen PINN to extract future macroscopic traffic predictions before taking an action.
*   **code\scripts\run_pinn_experiments.py**: The main execution script used to launch the training for the Baseline PPO and all PINN-ablated PPO agents.
*   **code\scripts\generate_pinn_rl_report.py**: A utility script used to aggregate the metrics from all experiments and perform statistical testing.

### 2. data\ & models\ (Dataset & Trained Weights)
*   **data\traffic_history_6x6.csv**: The 12.5MB macroscopic traffic history dataset used to train the PINN model.
*   **models\pinn_model.pth**: The saved pre-trained PyTorch weights for the Physics-Informed Neural Network used during the RL execution.

### 3. 
eports\ (Analysis)
*   **
eports\PPO_vs_PINN_PPO.md**: The final comprehensive report containing the statistical t-tests, metrics analysis, and conclusions of the ablation study.

### 4. 
esults\ (Raw Data)
*   **
esults\ppo\**: Raw training logs for the Baseline PPO model.
*   **
esults\ppo_pinn_queue\**: Logs for the agent augmented with future queue predictions.
*   **
esults\ppo_pinn_queue_density\**: Logs for the agent augmented with both queue and density predictions.
*   **
esults\ppo_pinn\**: Logs for the agent augmented with all 5 macroscopic PINN predictions.

### 5. plots\ (Visualizations)
*   Contains the generated learning curves comparing the Baseline PPO against the PINN-augmented models across all metrics (Reward, Waiting Time, Queue, Throughput, Speed, Fuel).
