import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOActorCritic, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        
    def generate_batches(self, batch_size=64):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]
        
        return np.array(self.states),\
               np.array(self.actions),\
               np.array(self.probs),\
               np.array(self.vals),\
               np.array(self.rewards),\
               np.array(self.dones),\
               batches
               
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class PPOAgent:
    def __init__(self, state_dim=10, action_dim=4, lr=3e-4, gamma=0.99, gae_lambda=0.95, policy_clip=0.2, n_epochs=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        
        self.ac = PPOActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        self.memory = PPOMemory()
        
    def act(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, state_value = self.ac(state_tensor)
            
        dist = Categorical(action_probs)
        if evaluate:
            action = torch.argmax(action_probs).item()
        else:
            action = dist.sample().item()
            
        return action, dist.log_prob(torch.tensor(action).to(self.device)).item(), state_value.item()
        
    def step(self, state, action, prob, val, reward, done):
        self.memory.store_memory(state, action, prob, val, reward, done)
        
    def train(self):
        if len(self.memory.states) == 0:
            return 0.0
            
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
        
        advantages = np.zeros(len(reward_arr), dtype=np.float32)
        for t in range(len(reward_arr)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr)-1):
                a_t += discount*(reward_arr[k] + self.gamma*vals_arr[k+1]*(1-int(dones_arr[k])) - vals_arr[k])
                discount *= self.gamma*self.gae_lambda
            advantages[t] = a_t
            
        advantages = torch.tensor(advantages).to(self.device)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        for _ in range(self.n_epochs):
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch], dtype=torch.float).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)
                
                action_probs, state_values = self.ac(states)
                dist = Categorical(action_probs)
                new_probs = dist.log_prob(actions)
                
                prob_ratio = new_probs.exp() / old_probs.exp()
                
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantages[batch]
                
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                returns = advantages[batch] + torch.tensor(vals_arr[batch], dtype=torch.float).to(self.device)
                critic_loss = nn.MSELoss()(state_values.squeeze(), returns)
                
                loss = actor_loss + 0.5*critic_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
        self.memory.clear_memory()
        return total_loss / (self.n_epochs * len(batches))
        
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'ac': self.ac.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)
        
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.ac.load_state_dict(checkpoint['ac'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
