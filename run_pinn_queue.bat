@echo off
python benchmark_research\scripts\run_pinn_experiments.py --mode queue --algo ppo_pinn_queue --episodes 200
pause
