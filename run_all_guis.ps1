# 1. Run 2x2 Network GUI (PPO Agent)
Write-Host "Running 2x2 Network..." -ForegroundColor Green
Start-Process python -ArgumentList "benchmark_research\2x2_benchmark\scripts\evaluate_2x2.py", "--algo", "ppo", "--checkpoint", "200", "--gui"

# 2. Run 4x4 Network GUI (PPO Agent)
Write-Host "Running 4x4 Network..." -ForegroundColor Green
Start-Process python -ArgumentList "benchmark_research\4x4_benchmark\scripts\evaluate_4x4.py", "--algo", "ppo", "--checkpoint", "200", "--gui"

# 3. Run 6x6 Network GUI (PPO Agent)
Write-Host "Running 6x6 Network..." -ForegroundColor Green
Start-Process python -ArgumentList "benchmark_research\6x6_benchmark\scripts\evaluate_6x6.py", "--algo", "ppo", "--ep", "200"

# 4. Run Complete Hybrid Visualization (RL + PINN + ACO on 6x6)
Write-Host "Running Complete Hybrid 6x6 Network..." -ForegroundColor Green
Start-Process python -ArgumentList "run_complete_sumo_visualization.py"

# 5. Run 10x10 Network GUI (Environment Only)
Write-Host "Running 10x10 Network..." -ForegroundColor Green
Start-Process sumo-gui -ArgumentList "-c", "dqn_pinn_package\sumo_network_10x10\config.sumocfg"

Write-Host "All GUI processes have been launched in separate windows!" -ForegroundColor Cyan
