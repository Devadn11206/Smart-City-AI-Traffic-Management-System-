@echo off
python benchmark_research\scripts\run_pinn_experiments.py --mode queue_density --algo ppo_pinn_queue_density --episodes 200
pause
