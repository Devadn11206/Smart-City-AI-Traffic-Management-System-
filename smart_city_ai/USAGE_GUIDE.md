# Usage Guide

## Train or evaluate RL agents

```bash
python training/train_dqn.py
python training/train_ddqn.py
python training/train_ppo.py
python evaluation/compare_algorithms.py
```

## Run the hybrid emergency system

```bash
python integration/run_emergency_system.py --duration 20 --pinn pinn_models/trained_pinn.pth --hospital-edges hw2_to_cz1,hw3_to_cz2,hw5_to_rr4
```

## Run full evaluation

```bash
python tools/final_evaluation.py
```

## Open the dashboard

```bash
streamlit run dashboards/app.py
```