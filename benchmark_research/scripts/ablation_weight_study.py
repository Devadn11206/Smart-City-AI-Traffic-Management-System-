import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'pinn'))

from config import INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, LEARNING_RATE, EPOCHS, PHYSICS_LAMBDA, PATIENCE, MODEL_DIR, RESULTS_DIR
from dataset import get_dataloaders
from model import PINNModel
from physics import compute_physics_loss

def run_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, x_scaler, y_scaler, _ = get_dataloaders(action_aware=True)
    
    weights_to_test = [1.0, 2.0, 3.0, 4.0, 5.0]
    results = {}
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    for w in weights_to_test:
        print(f"\n{'='*50}\nStarting Training with Throughput Weight: {w}\n{'='*50}")
        
        model = PINNModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        def weighted_mse_loss_fn(preds, targets, weight_tp):
            weights = torch.ones(targets.shape[1], device=targets.device)
            weights[2] = weight_tp
            loss = (preds - targets) ** 2
            weighted_loss = (loss * weights).mean()
            throughput_loss = loss[:, 2].mean()
            return weighted_loss, throughput_loss
            
        model_path = os.path.join(MODEL_DIR, f'action_aware_pinn_model_w{w}.pth')
        best_val_loss = float('inf')
        patience_counter = 0
        
        history = {
            'train_total': [], 'train_data': [], 'train_phys': [], 'train_throughput': [], 'val_total': []
        }
        
        for epoch in range(1, EPOCHS + 1):
            model.train()
            ep_total = ep_data = ep_phys = ep_tp = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                
                preds = model(batch_x)
                data_loss, tp_loss = weighted_mse_loss_fn(preds, batch_y, w)
                phys_loss = compute_physics_loss(preds, batch_x)
                
                loss = data_loss + PHYSICS_LAMBDA * phys_loss
                loss.backward()
                optimizer.step()
                
                ep_total += loss.item()
                ep_data += data_loss.item()
                ep_phys += phys_loss.item()
                ep_tp += tp_loss.item()
                
            n = len(train_loader)
            avg_total = ep_total / n
            avg_data = ep_data / n
            avg_phys = ep_phys / n
            avg_tp = ep_tp / n
            
            model.eval()
            ep_val = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    preds = model(batch_x)
                    d_loss, _ = weighted_mse_loss_fn(preds, batch_y, w)
                    p_loss = compute_physics_loss(preds, batch_x)
                    ep_val += (d_loss.item() + PHYSICS_LAMBDA * p_loss.item())
                    
            avg_val = ep_val / len(val_loader)
            scheduler.step(avg_val)
            
            history['train_total'].append(avg_total)
            history['train_data'].append(avg_data)
            history['train_phys'].append(avg_phys)
            history['train_throughput'].append(avg_tp)
            history['val_total'].append(avg_val)
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Total: {avg_total:.4f} | Data: {avg_data:.4f} | Phys: {avg_phys:.4f} | TP: {avg_tp:.4f} | Val: {avg_val:.4f}")
                
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
        print(f"Training completed for weight {w}.")
        
        # Evaluation
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        all_preds, all_trues = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                preds = model(batch_x.to(device))
                all_preds.append(preds.cpu().numpy())
                all_trues.append(batch_y.numpy())
                
        all_preds = np.vstack(all_preds)
        all_trues = np.vstack(all_trues)
        
        all_preds_real = y_scaler.inverse_transform(all_preds)
        all_trues_real = y_scaler.inverse_transform(all_trues)
        
        all_preds_real[:, 2] = all_preds_real[:, 2] ** 2
        all_trues_real[:, 2] = all_trues_real[:, 2] ** 2
        
        targets = ['Density', 'Queue Length', 'Throughput', 'Average Speed', 'Fuel Consumption']
        metrics = {}
        for i, target in enumerate(targets):
            r2 = r2_score(all_trues_real[:, i], all_preds_real[:, i])
            metrics[target] = float(r2)
            
        print(f"Metrics for weight {w}: {metrics}")
        results[f"weight_{w}"] = metrics
        
        # Save history
        np.savez(os.path.join(RESULTS_DIR, f'learning_curves_w{w}.npz'), **history)
        
    with open(os.path.join(RESULTS_DIR, 'ablation_weights_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == '__main__':
    run_ablation()
