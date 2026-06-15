import torch
from utils.reliability import load_torch_checkpoint, extract_state_dict, save_torch_checkpoint
from utils.checkpoint_registry import build_checkpoint_metadata, validate_checkpoint
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

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
    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01,
                 replay_size=300000, batch_size=128, target_update=2000, device: str | None = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # allow explicit device selection; fall back to CUDA if available
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
        
        # Double DQN logic
        next_actions = torch.nan_to_num(self.model(next_states), nan=0.0, posinf=0.0, neginf=0.0).max(1)[1].unsqueeze(1)
        next_q = torch.nan_to_num(self.target_model(next_states).gather(1, next_actions).squeeze(1), nan=0.0, posinf=0.0, neginf=0.0)
        
        current_q = self.model(states).gather(1, actions).squeeze(1)
        target_q = rewards + (1 - dones) * self.gamma * next_q
        target_q = torch.nan_to_num(target_q, nan=0.0, posinf=0.0, neginf=0.0)
        
        loss = self.criterion(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        # update target network periodically by step count
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.update_target_network()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        metadata = build_checkpoint_metadata(algorithm='ddqn', state_dim=self.state_dim, action_dim=self.action_dim)
        state = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict()
        }
        save_torch_checkpoint(filepath, state, metadata=metadata)
        
    def load(self, filepath):
        checkpoint_info = validate_checkpoint(filepath, 'ddqn', self.state_dim, self.action_dim)
        if not checkpoint_info.get('compatible'):
            raise ValueError(f"Incompatible DDQN checkpoint: {checkpoint_info.get('reason')}")
        checkpoint = load_torch_checkpoint(filepath, map_location=self.device)
        
        state = checkpoint
        while isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
            
        self.model.load_state_dict(state, strict=True)
        
        target_state = checkpoint
        if 'target_model_state_dict' in target_state:
            while isinstance(target_state, dict) and 'target_model_state_dict' in target_state:
                target_state = target_state['target_model_state_dict']
            self.target_model.load_state_dict(target_state, strict=True)
        else:
            self.update_target_network()
