import os
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinn_models.pinn_model import TrafficPINN

def generate_congestion_heatmap(data_path, model_path, output_dir):
    print("Loading PINN model for Heatmap generation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_data = torch.load(model_path, map_location=device, weights_only=False)
    
    model = TrafficPINN().to(device)
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    
    edge_map = model_data['edge_map']
    t_max = model_data['t_max']
    x_max = model_data['x_max']
    
    # 2D Grid: Time vs Spatial Edge Index
    num_time_steps = 100
    num_edges = len(edge_map)
    
    t_vals = np.linspace(0, t_max, num_time_steps)
    edge_indices = np.arange(num_edges)
    
    T, X = np.meshgrid(t_vals, edge_indices)
    
    t_norm = torch.tensor(T / max(t_max, 1), dtype=torch.float32).unsqueeze(2).to(device)
    # using index as a proxy for spatial coordinate
    x_norm = torch.tensor(X / max(x_max, 1), dtype=torch.float32).unsqueeze(2).to(device)
    
    # Flatten for prediction
    t_flat = t_norm.view(-1, 1)
    x_flat = x_norm.view(-1, 1)
    
    with torch.no_grad():
        rho_pred_flat, v_pred_flat = model(t_flat, x_flat)
        
    rho_grid = rho_pred_flat.view(num_edges, num_time_steps).cpu().numpy()
    v_grid = v_pred_flat.view(num_edges, num_time_steps).cpu().numpy()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Density Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(rho_grid, cmap='YlOrRd', cbar_kws={'label': 'Density (veh/m)'})
    plt.title('PINN Predicted Traffic Density Spatiotemporal Heatmap', fontsize=16)
    plt.xlabel('Time Step (Normalized)', fontsize=14)
    plt.ylabel('Edge Index (Spatial coordinate)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pinn_density_heatmap.png'))
    plt.close()
    
    # Velocity Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(v_grid, cmap='RdYlGn', cbar_kws={'label': 'Velocity (m/s)'})
    plt.title('PINN Predicted Traffic Velocity Spatiotemporal Heatmap', fontsize=16)
    plt.xlabel('Time Step (Normalized)', fontsize=14)
    plt.ylabel('Edge Index (Spatial coordinate)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pinn_velocity_heatmap.png'))
    plt.close()
    
    print(f"Heatmaps successfully saved to {output_dir}")

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    generate_congestion_heatmap(
        str(repo_root / "traffic_data" / "datasets" / "physics_traffic_data.csv"),
        str(repo_root / "pinn_models" / "trained_pinn.pth"),
        str(repo_root / "visualizations" / "congestion_heatmaps")
    )
