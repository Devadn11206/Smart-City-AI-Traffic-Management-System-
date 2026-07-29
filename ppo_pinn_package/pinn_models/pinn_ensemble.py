import torch
import torch.nn as nn
import numpy as np

class WorldModelPINN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(WorldModelPINN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Predicts next state given (state, action)
        # Action is expected to be one-hot encoded or we can use an embedding.
        # Assuming action is a single integer, we'll embed it.
        self.action_emb = nn.Embedding(action_dim, 16)
        
        self.fc1 = nn.Linear(state_dim + 16, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)
        
        # Free flow speed and jam density for Greenshields Model
        self.v_free = 20.0  # m/s
        self.rho_jam = 0.2  # veh/m
        
    def forward(self, state, action):
        """
        state: [batch_size, state_dim]
        action: [batch_size, 1] or [batch_size]
        """
        if action.dim() == 1:
            action = action.unsqueeze(1)
            
        a_emb = self.action_emb(action.long()).squeeze(1)
        x = torch.cat([state, a_emb], dim=1)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        next_state = self.fc3(x)
        return next_state
    def physics_loss(self, state, next_state):
        """
        Calculate the microscopic physics residual based on queuing theory and intersection dynamics.
        Enforces logical constraints on predicted queue lengths and wait times.
        
        State indices based on SumoEnvironment6x6:
        2: Green vehicles (g_veh) / 50.0
        3: Red vehicles (r_veh) / 50.0
        4: Green wait time (g_wait) / 1000.0
        5: Red wait time (r_wait) / 1000.0
        """
        curr_g_veh, curr_r_veh = state[:, 2], state[:, 3]
        curr_g_wait, curr_r_wait = state[:, 4], state[:, 5]
        
        next_g_veh, next_r_veh = next_state[:, 2], next_state[:, 3]
        next_g_wait, next_r_wait = next_state[:, 4], next_state[:, 5]
        
        # 1. Non-negativity constraint (queues and wait times cannot be negative)
        loss_non_negative = torch.mean(torch.relu(-next_g_veh)**2 + torch.relu(-next_r_veh)**2 + 
                                       torch.relu(-next_g_wait)**2 + torch.relu(-next_r_wait)**2)
                                       
        # 2. Total vehicles conservation (roughly constant +/- small arrivals/departures)
        curr_total = curr_g_veh + curr_r_veh
        next_total = next_g_veh + next_r_veh
        loss_conservation = torch.mean((next_total - curr_total)**2)
        
        # 3. Wait time logic: Total wait time should correlate with total vehicles
        # (A large queue implies wait time must exist and grow)
        loss_wait_logic = torch.mean(torch.relu(next_total - 0.1) * torch.relu(-next_g_wait - next_r_wait + 0.01))
        
        # Total Microscopic Physics Loss
        loss = (10.0 * loss_non_negative) + (0.5 * loss_conservation) + (1.0 * loss_wait_logic)
        return loss

class PINNEnsemble(nn.Module):
    def __init__(self, state_dim, action_dim, num_models=3, device='cpu'):
        super(PINNEnsemble, self).__init__()
        self.num_models = num_models
        self.device = device
        
        self.models = nn.ModuleList([
            WorldModelPINN(state_dim, action_dim).to(device)
            for _ in range(num_models)
        ])
        
    def forward(self, state, action):
        """
        Returns the mean prediction and the variance across the ensemble.
        """
        predictions = []
        for model in self.models:
            pred = model(state, action)
            predictions.append(pred.unsqueeze(0))
            
        # Stack predictions: [num_models, batch_size, state_dim]
        stacked_preds = torch.cat(predictions, dim=0)
        
        mean_pred = torch.mean(stacked_preds, dim=0)
        variance = torch.var(stacked_preds, dim=0).mean(dim=1)  # Mean variance across state dims
        
        return mean_pred, variance
