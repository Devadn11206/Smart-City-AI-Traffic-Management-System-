import os
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

from config import *
from dataset import get_dataloaders
from model import PINNModel

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask): return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, _, test_loader, _, y_scaler, _ = get_dataloaders()
    
    model = PINNModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM).to(device)
    model_path = os.path.join(MODEL_DIR, 'pinn_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_preds = []
    all_trues = []
    
    print("Evaluating PINN on Test Set...")
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            
            all_preds.append(preds.cpu().numpy())
            all_trues.append(batch_y.numpy())
            
    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)
    
    # Inverse transform to get actual physical values
    all_preds_real = y_scaler.inverse_transform(all_preds)
    all_trues_real = y_scaler.inverse_transform(all_trues)
    
    targets = ['Density', 'Queue Length', 'Throughput', 'Average Speed', 'Fuel Consumption']
    metrics_res = {}
    
    for i, target in enumerate(targets):
        y_t = all_trues_real[:, i]
        y_p = all_preds_real[:, i]
        
        mae = mean_absolute_error(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        
        # Use WMAPE (Weighted MAPE) to avoid division by zero
        sum_true = np.sum(np.abs(y_t))
        if sum_true > 1e-6:
            wmape = np.sum(np.abs(y_t - y_p)) / sum_true * 100.0
        else:
            wmape = 0.0
            
        r2 = r2_score(y_t, y_p)
        
        metrics_res[target] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'WMAPE': float(wmape),
            'R2': float(r2)
        }
        
        print(f"--- {target} ---")
        print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | WMAPE: {wmape:.2f}% | R2: {r2:.4f}")
        
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics_res, f, indent=4)
        
    # Save predictions for visualization
    np.savez(os.path.join(RESULTS_DIR, 'predictions.npz'), y_true=all_trues_real, y_pred=all_preds_real)

if __name__ == '__main__':
    evaluate()
