import torch
from utils.reliability import load_torch_checkpoint, extract_state_dict, save_torch_checkpoint
from utils.checkpoint_registry import build_checkpoint_metadata, validate_checkpoint
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        policy_dist = self.actor(state)
        value = self.critic(state)
        return policy_dist, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=10,
                 gae_lambda=0.95, entropy_coef=0.01, max_grad_norm=0.5, batch_size=256, device: str | None = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        
        # allow explicit device selection; fall back to CUDA if available
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.MseLoss = nn.MSELoss()
        
        self.memory = []
        
    def act(self, state, eval_mode=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.policy_old(state_tensor)
        action_probs = torch.nan_to_num(action_probs, nan=0.0, posinf=0.0, neginf=0.0)
        action_probs = torch.clamp(action_probs, min=1e-8)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        dist = torch.distributions.Categorical(action_probs)
        if eval_mode:
            action = torch.argmax(action_probs).item()
        else:
            action = dist.sample().item()
        action_logprob = dist.log_prob(torch.tensor(action).to(self.device)).item()
        
        return action, action_logprob
        
    def remember(self, state, action, action_logprob, reward, done):
        self.memory.append((state, action, action_logprob, reward, done))
        
    def replay(self):
        if len(self.memory) == 0:
            return
            
        states = torch.FloatTensor(np.array([m[0] for m in self.memory])).to(self.device)
        actions = torch.LongTensor([m[1] for m in self.memory]).to(self.device)
        old_logprobs = torch.FloatTensor([m[2] for m in self.memory]).to(self.device)
        rewards = [m[3] for m in self.memory]
        dones = [m[4] for m in self.memory]
        
        # Compute values for states using current policy
        with torch.no_grad():
            _, state_values_all = self.policy(states)
        state_values_all = state_values_all.squeeze().cpu().numpy()

        # Generalized Advantage Estimation (GAE)
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - float(dones[t])
                next_values = state_values_all[t] if next_non_terminal > 0 else 0.0
            else:
                next_non_terminal = 1.0 - float(dones[t+1])
                next_values = state_values_all[t+1]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - state_values_all[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(state_values_all).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.k_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logprobs = old_logprobs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                action_probs, state_values = self.policy(batch_states)
                action_probs = torch.nan_to_num(action_probs, nan=0.0, posinf=0.0, neginf=0.0)
                action_probs = torch.clamp(action_probs, min=1e-8)
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                dist = torch.distributions.Categorical(action_probs)
                logprobs = dist.log_prob(batch_actions)
                dist_entropy = dist.entropy().mean()

                state_values = state_values.squeeze()
                if state_values.dim() == 0:
                    state_values = state_values.unsqueeze(0)

                ratios = torch.exp(logprobs - batch_old_logprobs.detach())
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * self.MseLoss(state_values, batch_returns)
                entropy_loss = -self.entropy_coef * dist_entropy

                loss = actor_loss + critic_loss + entropy_loss
                self.last_loss = loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear()

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        metadata = build_checkpoint_metadata(algorithm='ppo', state_dim=self.state_dim, action_dim=self.action_dim)
        save_torch_checkpoint(filepath, self.policy.state_dict(), metadata=metadata)
        
    def load(self, filepath):
        checkpoint_info = validate_checkpoint(filepath, 'ppo', self.state_dim, self.action_dim)
        if not checkpoint_info.get('compatible'):
            raise ValueError(f"Incompatible PPO checkpoint: {checkpoint_info.get('reason')}")
        checkpoint = load_torch_checkpoint(filepath, map_location=self.device)
        state = extract_state_dict(checkpoint)
        self.policy.load_state_dict(state, strict=True)
        self.policy_old.load_state_dict(self.policy.state_dict())
