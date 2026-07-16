import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, LEARNING_RATE, EPOCHS, PHYSICS_LAMBDA, PATIENCE, MODEL_DIR, RESULTS_DIR
from dataset import get_dataloaders
from model import PINNModel
from physics import compute_physics_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader, x_scaler, y_scaler, _ = get_dataloaders()
    
    model = PINNModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'pinn_model.pth')
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    physics_losses = []
    
    print(f"Starting PINN Training on {device}...")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_train_loss = 0
        epoch_phys_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            preds = model(batch_x)
            
            data_loss = criterion(preds, batch_y)
            phys_loss = compute_physics_loss(preds, batch_x)
            
            loss = data_loss + PHYSICS_LAMBDA * phys_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += data_loss.item()
            epoch_phys_loss += phys_loss.item()
            
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_phys_loss = epoch_phys_loss / len(train_loader)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                data_loss = criterion(preds, batch_y)
                phys_loss = compute_physics_loss(preds, batch_x)
                epoch_val_loss += (data_loss.item() + PHYSICS_LAMBDA * phys_loss.item())
                
        avg_val_loss = epoch_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        train_losses.append(avg_train_loss)
        physics_losses.append(avg_phys_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Phys Loss: {avg_phys_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"--> Saved best model to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
                
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, 'learning_curves.npz'), train=train_losses, val=val_losses, physics=physics_losses)
    print("Training Completed.")

if __name__ == '__main__':
    train()
