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
        self.memories = {}
        
    def store_memory(self, ts_id, state, action, probs, vals, reward, done):
        if ts_id not in self.memories:
            self.memories[ts_id] = {'states': [], 'actions': [], 'probs': [], 'vals': [], 'rewards': [], 'dones': []}
            
        self.memories[ts_id]['states'].append(state)
        self.memories[ts_id]['actions'].append(action)
        self.memories[ts_id]['probs'].append(probs)
        self.memories[ts_id]['vals'].append(vals)
        self.memories[ts_id]['rewards'].append(reward)
        self.memories[ts_id]['dones'].append(done)
        
    def clear_memory(self):
        self.memories = {}

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
        
    def step(self, ts_id, state, action, prob, val, reward, done):
        self.memory.store_memory(ts_id, state, action, prob, val, reward, done)
        
    def train(self):
        if len(self.memory.memories) == 0:
            return {}
            
        all_states = []
        all_actions = []
        all_probs = []
        all_vals = []
        all_returns = []
        all_advantages = []
        
        # O(N) backward GAE and separation per trajectory
        for ts_id, mem in self.memory.memories.items():
            rewards = mem['rewards']
            vals = mem['vals']
            dones = mem['dones']
            
            n_steps = len(rewards)
            advantages = np.zeros(n_steps, dtype=np.float32)
            gae = 0
            for t in reversed(range(n_steps)):
                # If done, next value is 0, else it's the next step's value
                next_val = vals[t+1] if t + 1 < n_steps else 0
                delta = rewards[t] + self.gamma * next_val * (1 - int(dones[t])) - vals[t]
                gae = delta + self.gamma * self.gae_lambda * gae * (1 - int(dones[t]))
                advantages[t] = gae
                
            returns = advantages + np.array(vals)
            
            all_states.extend(mem['states'])
            all_actions.extend(mem['actions'])
            all_probs.extend(mem['probs'])
            all_vals.extend(mem['vals'])
            all_returns.extend(returns)
            all_advantages.extend(advantages)
            
        if len(all_states) == 0:
            return {}
            
        # Convert to numpy arrays
        state_arr = np.array(all_states)
        action_arr = np.array(all_actions)
        old_prob_arr = np.array(all_probs)
        vals_arr = np.array(all_vals)
        returns_arr = np.array(all_returns)
        adv_arr = np.array(all_advantages)
        
        # Flatten and create batches
        n_states = len(state_arr)
        batch_size = min(64, n_states)
        
        adv_tensor = torch.tensor(adv_arr).to(self.device)
        # Normalize advantages
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_kl = 0
        total_clip_frac = 0
        
        for _ in range(self.n_epochs):
            indices = np.arange(n_states, dtype=np.int64)
            np.random.shuffle(indices)
            batches = [indices[i:i+batch_size] for i in range(0, n_states, batch_size)]
            
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch], dtype=torch.float).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)
                
                action_probs, state_values = self.ac(states)
                dist = Categorical(action_probs)
                new_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                prob_ratio = new_probs.exp() / old_probs.exp()
                
                # Metrics computation
                with torch.no_grad():
                    log_ratio = new_probs - old_probs
                    kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                    clip_frac = torch.mean((torch.abs(prob_ratio - 1) > self.policy_clip).float())
                
                weighted_probs = adv_tensor[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * adv_tensor[batch]
                
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                # Unnormalized returns for critic target
                returns_batch = torch.tensor(returns_arr[batch], dtype=torch.float).to(self.device)
                critic_loss = nn.MSELoss()(state_values.squeeze(), returns_batch)
                
                loss = actor_loss + 0.5*critic_loss - 0.01*entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                total_kl += kl.item()
                total_clip_frac += clip_frac.item()
                
        self.memory.clear_memory()
        
        n_updates = self.n_epochs * len(batches)
        
        # Explained variance
        y_pred = vals_arr
        y_true = returns_arr
        var_y = np.var(y_true)
        explained_var = float('nan') if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)
        
        return {
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'kl_div': total_kl / n_updates,
            'clip_frac': total_clip_frac / n_updates,
            'explained_var': explained_var
        }
        
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
