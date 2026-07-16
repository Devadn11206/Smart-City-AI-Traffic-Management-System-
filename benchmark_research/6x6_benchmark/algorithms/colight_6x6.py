import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import numpy as np
import sys

# Import ReplayBuffer from algorithms directory
# Need to ensure sys.path has benchmark_research
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if base_dir not in sys.path:
    sys.path.append(base_dir)

try:
    from algorithms.replay_buffer import ReplayBuffer
except ImportError:
    # Quick fallback if path issues
    class ReplayBuffer:
        def __init__(self, capacity):
            self.capacity = capacity
            self.buffer = []
            self.position = 0
        def push(self, state, action, reward, next_state, done):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity
        def sample(self, batch_size):
            batch = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done
        def __len__(self):
            return len(self.buffer)

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
    def forward(self, h, adj):
        Wh = self.W(h) # (num_nodes, out_features)
        
        N = Wh.size(0)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        
        e = F.leaky_relu(self.a(all_combinations_matrix).view(N, N))
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

class CoLightNetwork(nn.Module):
    def __init__(self, state_dim=10, action_dim=4, hidden_dim=128):
        super(CoLightNetwork, self).__init__()
        self.embedding = nn.Linear(state_dim, hidden_dim)
        self.gat1 = GATLayer(hidden_dim, hidden_dim)
        self.gat2 = GATLayer(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, states, adj):
        h = F.relu(self.embedding(states))
        h = self.gat1(h, adj)
        h = self.gat2(h, adj)
        return self.out(h)
        
class CoLightAgent:
    def __init__(self, num_nodes=36, grid_size=6, state_dim=10, action_dim=4, lr=1e-3, gamma=0.99, buffer_capacity=50000, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.grid_size = grid_size
        
        self.q_network = CoLightNetwork(state_dim, action_dim).to(self.device)
        self.target_network = CoLightNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.985
        
        self._build_adjacency_matrix()
        
    def _build_adjacency_matrix(self):
        adj = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            adj[i][i] = 1 # Self-connection
            row = i // self.grid_size
            col = i % self.grid_size
            
            # Neighbors
            if col > 0: # Left
                adj[i][i - 1] = 1
            if col < self.grid_size - 1: # Right
                adj[i][i + 1] = 1
            if row > 0: # Top
                adj[i][i - self.grid_size] = 1
            if row < self.grid_size - 1: # Bottom
                adj[i][i + self.grid_size] = 1
                
        self.adj = torch.tensor(adj, dtype=torch.float32).to(self.device)
        
    def act(self, states_list, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return [random.randint(0, 3) for _ in range(self.num_nodes)]
            
        states_tensor = torch.FloatTensor(np.array(states_list)).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(states_tensor, self.adj)
        return torch.argmax(q_values, dim=1).cpu().numpy().tolist()
        
    def step(self, states, actions, rewards, next_states, done):
        self.buffer.push(np.array(states), np.array(actions), np.array(rewards), np.array(next_states), done)
        
    def train(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
            
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(2).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(2).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).unsqueeze(2).to(self.device)
        
        total_loss = 0
        self.optimizer.zero_grad()
        
        for i in range(self.batch_size):
            curr_q = self.q_network(states[i], self.adj).gather(1, actions[i])
            
            with torch.no_grad():
                next_actions = self.q_network(next_states[i], self.adj).argmax(1, keepdim=True)
                next_q = self.target_network(next_states[i], self.adj).gather(1, next_actions)
                target_q = rewards[i] + (1 - dones[i]) * self.gamma * next_q
                
            loss = nn.MSELoss()(curr_q, target_q)
            total_loss += loss
            
        total_loss = total_loss / self.batch_size
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
        
    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
