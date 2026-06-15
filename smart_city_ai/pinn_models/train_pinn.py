import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse

from pinn_model import TrafficPINN

def load_and_preprocess_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # We need to map string edge_id to numerical spatial coordinate x
    # Sort edge_ids to maintain consistency
    unique_edges = sorted(df['edge_id'].unique())
    edge_to_idx = {edge: i for i, edge in enumerate(unique_edges)}
    df['x'] = df['edge_id'].map(edge_to_idx)
    
    # Normalize inputs for better neural network training
    t_max = df['time_step'].max()
    x_max = df['x'].max()
    
    df['t_norm'] = df['time_step'] / max(t_max, 1)
    df['x_norm'] = df['x'] / max(x_max, 1)
    
    # Extract tensors
    t_tensor = torch.tensor(df['t_norm'].values, dtype=torch.float32).unsqueeze(1)
    x_tensor = torch.tensor(df['x_norm'].values, dtype=torch.float32).unsqueeze(1)
    rho_tensor = torch.tensor(df['density'].values, dtype=torch.float32).unsqueeze(1)
    v_tensor = torch.tensor(df['velocity'].values, dtype=torch.float32).unsqueeze(1)
    
    return TensorDataset(t_tensor, x_tensor, rho_tensor, v_tensor), edge_to_idx, t_max, x_max

def train_pinn(epochs=100, batch_size=2048, lr=1e-3, data_path="../traffic_data/datasets/physics_traffic_data.csv"):
    dataset, edge_map, t_max, x_max = load_and_preprocess_data(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TrafficPINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss weightings
    lambda_data = 1.0
    lambda_phys = 0.1
    
    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_data_loss = 0.0
        total_phys_loss = 0.0
        
        for t, x, rho_true, v_true in dataloader:
            t, x = t.to(device), x.to(device)
            rho_true, v_true = rho_true.to(device), v_true.to(device)
            
            optimizer.zero_grad()
            
            # Predict
            rho_pred, v_pred = model(t, x)
            
            # Data loss
            loss_data = torch.mean((rho_pred - rho_true)**2) + torch.mean((v_pred - v_true)**2)
            
            # Physics loss
            loss_phys = model.physics_loss(t, x)
            
            # Combined loss
            loss = lambda_data * loss_data + lambda_phys * loss_phys
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_data_loss += loss_data.item()
            total_phys_loss += loss_phys.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / len(dataloader)
            avg_data = total_data_loss / len(dataloader)
            avg_phys = total_phys_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} | Total Loss: {avg_loss:.6f} | Data: {avg_data:.6f} | Physics: {avg_phys:.6f}")
            
    # Save the model
    save_dir = "../pinn_models"
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'model_state_dict': model.state_dict(),
        'edge_map': edge_map,
        't_max': t_max,
        'x_max': x_max
    }, f"{save_dir}/trained_pinn.pth")
    print("Model saved to pinn_models/trained_pinn.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    
    train_pinn(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
