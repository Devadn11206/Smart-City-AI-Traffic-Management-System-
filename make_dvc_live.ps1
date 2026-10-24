# Initialize DVC if not already initialized
if (-Not (Test-Path .dvc)) {
    Write-Host "Initializing DVC..." -ForegroundColor Green
    dvc init
}

Write-Host "Adding heavy datasets to DVC..." -ForegroundColor Cyan
# Add datasets
dvc add pinn_dataset_10x10.pt
dvc add pinn_dataset_6x6.pt

Write-Host "Adding model checkpoints to DVC..." -ForegroundColor Cyan
# Add pretrained models
dvc add pinn_ensemble_pretrained_10x10.pth
dvc add pinn_ensemble_pretrained_6x6.pth

Write-Host "Adding MLOps tracking databases to DVC..." -ForegroundColor Cyan
# Add tracking databases
dvc add mlflow.db
dvc add optuna.db

# Configure a local remote for simulation
if (-Not (Test-Path dvc_remote)) {
    mkdir dvc_remote | Out-Null
    dvc remote add -d local_remote dvc_remote
}

Write-Host "Adding DVC tracking files to Git..." -ForegroundColor Cyan
# Add the DVC pointers to git
git add .dvc/config .gitignore
git add pinn_dataset_10x10.pt.dvc pinn_dataset_6x6.pt.dvc
git add pinn_ensemble_pretrained_10x10.pth.dvc pinn_ensemble_pretrained_6x6.pth.dvc
git add mlflow.db.dvc optuna.db.dvc

Write-Host "DVC setup and data addition complete!" -ForegroundColor Green
