import os
import sys
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add pinn folder to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'pinn'))

from config import INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, MODEL_DIR
from dataset import get_dataloaders
from model import PINNModel

def evaluate_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    _, _, test_loader, x_scaler, y_scaler, _ = get_dataloaders(action_aware=True)
    
    # Load model
    model = PINNModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM).to(device)
    model_path = os.path.join(MODEL_DIR, 'action_aware_pinn_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds_ablated = []
    all_trues = []
    
    print("Evaluating Ablated PINN (Zeroing out Phase-Aligned Features)...")
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # Ablation: Zero out action_queue_length (8) and action_vehicle_count (9)
            # Find the scaled value corresponding to 0
            zero_unscaled = np.zeros((1, INPUT_DIM))
            zero_scaled = x_scaler.transform(zero_unscaled)
            
            batch_x[:, 8] = float(zero_scaled[0, 8])
            batch_x[:, 9] = float(zero_scaled[0, 9])
            
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            
            all_preds_ablated.append(preds.cpu().numpy())
            all_trues.append(batch_y.numpy())
            
    all_preds_ablated = np.vstack(all_preds_ablated)
    all_trues = np.vstack(all_trues)
    
    # Inverse transform
    all_preds_real = y_scaler.inverse_transform(all_preds_ablated)
    all_trues_real = y_scaler.inverse_transform(all_trues)
    
    # Reverse square root on throughput (index 2)
    all_preds_real[:, 2] = all_preds_real[:, 2] ** 2
    all_trues_real[:, 2] = all_trues_real[:, 2] ** 2
    
    targets = ['Density', 'Queue Length', 'Throughput', 'Average Speed', 'Fuel Consumption']
    
    for i, target in enumerate(targets):
        y_t = all_trues_real[:, i]
        y_p = all_preds_real[:, i]
        
        r2 = r2_score(y_t, y_p)
        print(f"{target} Ablated R2: {r2:.4f}")

if __name__ == '__main__':
    evaluate_ablation()
