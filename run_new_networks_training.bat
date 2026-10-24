@echo off
echo Starting Baseline DDQN Training on 10x10 Network...
python ddqn_pinn_package\training\train_ddqn_10x10.py
echo 10x10 Baseline Training Finished.

echo Starting PI-DDQN Training on 10x10 Network...
python ddqn_pinn_package\training\train_pi_ddqn_10x10.py
echo 10x10 PI-DDQN Training Finished.

echo Starting PI-DDQN Training on Irregular Network...
python ddqn_pinn_package\training\train_pi_ddqn_irregular.py
echo Irregular PI-DDQN Training Finished.

echo Generating Plots...
python plot_new_networks_results.py
echo All done! Plots are in results\new_networks_comparison.png
pause
