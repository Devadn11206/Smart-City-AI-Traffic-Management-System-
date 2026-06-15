import torch
import torch.nn as nn

class TrafficPINN(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 64, 2]):
        """
        Physics-Informed Neural Network for Traffic Flow.
        Input: (t, x) - time and spatial coordinate
        Output: (rho, v) - density and velocity
        """
        super(TrafficPINN, self).__init__()
        
        self.target_vmax = 20.0  # Max speed m/s
        self.target_rhomax = 0.2 # Max density veh/m
        
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
                
        self.network = nn.Sequential(*modules)
        
    def forward(self, t, x):
        inputs = torch.cat([t, x], dim=1)
        output = self.network(inputs)
        # Outputs: rho is the first column, v is the second
        # Ensure positivity
        rho = torch.sigmoid(output[:, 0:1]) * self.target_rhomax
        v = torch.sigmoid(output[:, 1:2]) * self.target_vmax
        return rho, v
        
    def physics_loss(self, t, x):
        """
        Computes the physics residual loss based on Traffic Continuity Equation.
        del(rho)/del(t) + del(rho*v)/del(x) = 0
        """
        # Enable gradients for inputs
        t.requires_grad_(True)
        x.requires_grad_(True)
        
        rho, v = self.forward(t, x)
        
        # Calculate gradients
        rho_t = torch.autograd.grad(
            rho, t, 
            grad_outputs=torch.ones_like(rho),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Flux q = rho * v
        q = rho * v
        
        q_x = torch.autograd.grad(
            q, x,
            grad_outputs=torch.ones_like(q),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Continuity equation residual: rho_t + q_x = 0
        f_residual = rho_t + q_x
        
        loss_f = torch.mean(f_residual**2)
        return loss_f
