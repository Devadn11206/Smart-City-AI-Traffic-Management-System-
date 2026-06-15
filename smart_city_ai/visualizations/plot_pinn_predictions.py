import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinn_models.pinn_model import TrafficPINN

def plot_predictions(data_path, model_path, output_dir):
    print("Loading model for visualizations...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_data = torch.load(model_path, map_location=device, weights_only=False)
    
    model = TrafficPINN().to(device)
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    
    edge_map = model_data['edge_map']
    t_max = model_data['t_max']
    x_max = model_data['x_max']
    
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    edges = list(edge_map.keys())
    if not edges:
        print("No edges found in edge_map.")
        return
        
    # Sample a few random edges and plot their predictions
    sample_edges = np.random.choice(edges, size=min(3, len(edges)), replace=False)
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    for sample_edge in sample_edges:
        edge_df = df[df['edge_id'] == sample_edge].sort_values('time_step')
        if len(edge_df) == 0:
            continue
            
        t_vals = edge_df['time_step'].values
        x_val = edge_map[sample_edge]
        
        true_rho = edge_df['density'].values
        true_v = edge_df['velocity'].values
        
        pred_rho = []
        pred_v = []
        
        with torch.no_grad():
            for t in t_vals:
                t_norm = t / max(t_max, 1)
                x_norm = x_val / max(x_max, 1)
                
                t_tensor = torch.tensor([[t_norm]], dtype=torch.float32).to(device)
                x_tensor = torch.tensor([[x_norm]], dtype=torch.float32).to(device)
                
                rho_out, v_out = model(t_tensor, x_tensor)
                pred_rho.append(rho_out.item())
                pred_v.append(v_out.item())
                
        # Density plot
        plt.figure(figsize=(12, 6))
        plt.plot(t_vals, true_rho, label='SUMO Target', alpha=0.8, linewidth=2, color='blue')
        plt.plot(t_vals, pred_rho, label='PINN Prediction', alpha=0.8, linewidth=2, linestyle='--', color='red')
        plt.title(f'Density Tracking on Edge: {sample_edge}', fontsize=14)
        plt.xlabel('Simulation Steps', fontsize=12)
        plt.ylabel('Density (veh/m)', fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'density_prediction_{sample_edge}.png'))
        plt.close()
        
        # Velocity plot
        plt.figure(figsize=(12, 6))
        plt.plot(t_vals, true_v, label='SUMO Target', alpha=0.8, linewidth=2, color='green')
        plt.plot(t_vals, pred_v, label='PINN Prediction', alpha=0.8, linewidth=2, linestyle='--', color='orange')
        plt.title(f'Velocity Tracking on Edge: {sample_edge}', fontsize=14)
        plt.xlabel('Simulation Steps', fontsize=12)
        plt.ylabel('Velocity (m/s)', fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'velocity_prediction_{sample_edge}.png'))
        plt.close()
        
    print(f"Prediction plots successfully saved to {output_dir}")

if __name__ == "__main__":
    plot_predictions(
        os.path.join("..", "traffic_data", "datasets", "physics_traffic_data.csv"),
        os.path.join("..", "pinn_models", "trained_pinn.pth"),
        "prediction_graphs"
    )
