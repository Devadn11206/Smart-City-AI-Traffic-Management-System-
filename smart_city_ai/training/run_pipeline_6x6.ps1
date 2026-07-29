Write-Host "========================================="
Write-Host "1. Generating PINN Transition Dataset"
Write-Host "========================================="
python generate_pinn_dataset_6x6.py

Write-Host "`n========================================="
Write-Host "2. Pre-training Offline PINN Ensemble"
Write-Host "========================================="
python train_pinn_offline_6x6.py

Write-Host "`n========================================="
Write-Host "3. Training Baseline DQN (from checkpoint)"
Write-Host "========================================="
python train_dqn_6x6.py

Write-Host "`n========================================="
Write-Host "4. Training PI-MBPO (from checkpoint)"
Write-Host "========================================="
python train_pi_mbpo_6x6.py

Write-Host "`nPipeline completed successfully!"
