import torch
import torch.nn as nn
import torch.nn.functional as F

class StateEncoder(nn.Module):
    def __init__(self, state_dim=10):
        super(StateEncoder, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return x

class PredictionEncoder(nn.Module):
    def __init__(self, physics_dim=5):
        super(PredictionEncoder, self).__init__()
        self.fc1 = nn.Linear(physics_dim, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, prediction):
        x = F.relu(self.fc1(prediction))
        x = F.relu(self.fc2(x))
        return x

class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()

    def forward(self, state_emb, physics_emb):
        return torch.cat([state_emb, physics_emb], dim=-1)

class ActionConditionedQNetwork(nn.Module):
    def __init__(self, fused_dim=96):
        super(ActionConditionedQNetwork, self).__init__()
        self.fc1 = nn.Linear(fused_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, fused_features):
        x = F.relu(self.fc1(fused_features))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class GatedFeatureFusion(nn.Module):
    def __init__(self, state_dim=64, physics_dim=32):
        super(GatedFeatureFusion, self).__init__()
        self.physics_proj = nn.Linear(physics_dim, state_dim)
        self.gate_layer = nn.Linear(state_dim * 2, state_dim)
        
    def forward(self, state_emb, physics_emb):
        p_emb_proj = F.relu(self.physics_proj(physics_emb))
        concat_emb = torch.cat([state_emb, p_emb_proj], dim=-1)
        gate = torch.sigmoid(self.gate_layer(concat_emb))
        return gate * state_emb + (1 - gate) * p_emb_proj

class AC_PG_DQN(nn.Module):
    """
    Action-Conditioned Physics-Guided Deep Q-Network.
    Evaluates Q(s, a_i) by fusing the state embedding with the 
    physics-informed prediction for action a_i.
    Supports multiple ablation modes.
    """
    def __init__(self, state_dim=10, physics_dim=5, mode=2):
        super(AC_PG_DQN, self).__init__()
        self.mode = mode
        self.state_encoder = StateEncoder(state_dim)
        
        if self.mode == 1:
            self.q_network = ActionConditionedQNetwork(fused_dim=64)
        elif self.mode == 2:
            self.prediction_encoder = PredictionEncoder(physics_dim)
            self.feature_fusion = FeatureFusion()
            self.q_network = ActionConditionedQNetwork(fused_dim=96)
        elif self.mode == 3:
            self.lightweight_proj = nn.Linear(physics_dim, 32)
            self.feature_fusion = FeatureFusion()
            self.q_network = ActionConditionedQNetwork(fused_dim=96)
        elif self.mode == 4:
            self.prediction_encoder = PredictionEncoder(physics_dim)
            self.gated_fusion = GatedFeatureFusion(state_dim=64, physics_dim=32)
            self.q_network = ActionConditionedQNetwork(fused_dim=64)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def forward(self, state, prediction):
        h_s = self.state_encoder(state)
        
        if self.mode == 1:
            q_val = self.q_network(h_s)
        elif self.mode == 2:
            h_p = self.prediction_encoder(prediction)
            h_f = self.feature_fusion(h_s, h_p)
            q_val = self.q_network(h_f)
        elif self.mode == 3:
            h_p = F.relu(self.lightweight_proj(prediction))
            h_f = self.feature_fusion(h_s, h_p)
            q_val = self.q_network(h_f)
        elif self.mode == 4:
            h_p = self.prediction_encoder(prediction)
            h_f = self.gated_fusion(h_s, h_p)
            q_val = self.q_network(h_f)
            
        return q_val
