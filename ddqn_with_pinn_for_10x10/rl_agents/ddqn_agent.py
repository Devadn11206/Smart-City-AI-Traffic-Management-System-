import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

try:
    from smart_city_ai.utils.reliability import load_torch_checkpoint, extract_state_dict, save_torch_checkpoint
    from smart_city_ai.utils.checkpoint_registry import build_checkpoint_metadata, validate_checkpoint
except ImportError:
    pass # If utils aren't available, we'll implement simple fallback saving

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DDQNAgent:
    """
    Baseline Double Deep Q-Network Agent.
    Does NOT use any physics-informed data or loss.
    """
    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01,
                 replay_size=200000, batch_size=256, target_update=1000, device: str | None = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = QNetwork(state_dim, action_dim).to(self.device)
        self.target_model = QNetwork(state_dim, action_dim).to(self.device)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        
        self.memory = deque(maxlen=replay_size)
        self.batch_size = batch_size
        self.target_update = max(1, int(target_update))
        self.learn_step_counter = 0
        self.last_loss = 0.0
        
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, eval_mode=False):
        if not eval_mode and random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = torch.nan_to_num(self.model(state_tensor), nan=0.0, posinf=0.0, neginf=0.0)
        return torch.argmax(q_values).item()
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(self.device)
        actions = torch.LongTensor([m[1] for m in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([m[2] for m in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch])).to(self.device)
        dones = torch.FloatTensor([float(m[4]) for m in minibatch]).to(self.device)
        
        # DOUBLE DQN LOGIC
        # 1. Action Selection: Online network selects best action for next_state
        with torch.no_grad():
            best_next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            # 2. Action Evaluation: Target network evaluates Q-value of that selected action
            next_q = self.target_model(next_states).gather(1, best_next_actions).squeeze(1)
            next_q = torch.nan_to_num(next_q, nan=0.0, posinf=0.0, neginf=0.0)
            
        target_q = rewards + (1 - dones) * self.gamma * next_q
        target_q = torch.nan_to_num(target_q, nan=0.0, posinf=0.0, neginf=0.0)
        
        current_q = self.model(states).gather(1, actions).squeeze(1)
        
        loss = self.criterion(current_q, target_q.detach())
        self.last_loss = loss.item()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.update_target_network()
        
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            metadata = build_checkpoint_metadata(algorithm='ddqn', state_dim=self.state_dim, action_dim=self.action_dim)
            save_torch_checkpoint(filepath, self.model.state_dict(), metadata=metadata)
        except NameError:
            torch.save(self.model.state_dict(), filepath)
        
    def load(self, filepath):
        try:
            checkpoint_info = validate_checkpoint(filepath, 'ddqn', self.state_dim, self.action_dim)
            if not checkpoint_info.get('compatible'):
                raise ValueError(f"Incompatible DDQN checkpoint: {checkpoint_info.get('reason')}")
            checkpoint = load_torch_checkpoint(filepath, map_location=self.device)
            state = extract_state_dict(checkpoint)
        except NameError:
            state = torch.load(filepath, map_location=self.device)
            
        self.model.load_state_dict(state, strict=True)
        self.update_target_network()
