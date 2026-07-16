import torch

def compute_physics_loss(y_pred, x_input, delta_t=10.0, delta_x=100.0):
    """
    Computes the physics-informed loss based on the Traffic Flow Continuity Equation:
    ∂ρ/∂t + ∂(ρv)/∂x = 0
    
    x_input features (normalized):
    0: traffic_density (t)
    1: queue_length (t)
    2: throughput (t)
    3: avg_speed (t)
    ...
    
    y_pred features (normalized predictions):
    0: traffic_density (t+1)
    1: queue_length (t+1)
    2: throughput (t+1)
    3: avg_speed (t+1)
    ...
    """
    
    rho_t = x_input[:, 0]
    v_t = x_input[:, 3]
    
    rho_t1 = y_pred[:, 0]
    v_t1 = y_pred[:, 3]
    
    # Time derivative approximation
    d_rho_dt = (rho_t1 - rho_t) / delta_t
    
    # Spatial derivative approximation
    # Assuming spatial change over the intersection segment is proportional to flux difference
    d_flux_dx = (rho_t1 * v_t1 - rho_t * v_t) / delta_x
    
    # Residual of continuity equation
    residual = d_rho_dt + d_flux_dx
    
    # Physics Loss is the MSE of the residual
    physics_loss = torch.mean(residual ** 2)
    
    return physics_loss
