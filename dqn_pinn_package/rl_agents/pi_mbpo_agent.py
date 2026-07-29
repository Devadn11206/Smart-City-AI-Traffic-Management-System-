import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

# Import the base DQNAgent
from rl_agents.dqn_agent import DQNAgent, QNetwork

class PIMBPOAgent(DQNAgent):
    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01,
                 replay_size=300000, batch_size=128, target_update=2000, device: str | None = None,
                 pinn_ensemble=None, variance_threshold=0.1, real_ratio=0.25):
        """
        Physics-Informed Model-Based Policy Optimization Agent.
        Uses a PINN Ensemble for 1-step synthetic rollouts to supplement the replay buffer.
        """
        super().__init__(state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, epsilon_min,
                         replay_size, batch_size, target_update, device)
        
        self.pinn_ensemble = pinn_ensemble
        self.variance_threshold = variance_threshold
        
        # Dual Buffers
        self.real_memory = deque(maxlen=replay_size)
        self.synthetic_memory = deque(maxlen=replay_size)
        
        self.real_ratio = real_ratio
        
    def remember(self, state, action, reward, next_state, done):
        """Override to save only to the real memory buffer."""
        self.real_memory.append((state, action, reward, next_state, done))
        
    def remember_synthetic(self, state, action, reward, next_state, done):
        """Save imagined transitions to the synthetic memory buffer."""
        self.synthetic_memory.append((state, action, reward, next_state, done))
        
    def generate_synthetic_data(self, num_samples=10, env_reward_func=None):
        """
        1-Step Policy-Guided Rollout.
        Samples from real buffer, asks DDQN for an action, and passes it to the PINN Ensemble.
        Filters out transitions with high uncertainty.
        """
        if self.pinn_ensemble is None or len(self.real_memory) < num_samples:
            return
            
        # Sample base states from real memory
        minibatch = random.sample(self.real_memory, num_samples)
        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(self.device)
        
        # Policy-guided actions (with a bit of epsilon-greedy exploration)
        actions = []
        for state in states:
            if random.random() <= self.epsilon:
                actions.append(random.randrange(self.action_dim))
            else:
                with torch.no_grad():
                    q_values = self.model(state.unsqueeze(0))
                actions.append(torch.argmax(q_values).item())
                
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        
        # Predict next states using the PINN ensemble
        with torch.no_grad():
            mean_next_states, variances = self.pinn_ensemble(states, actions_tensor)
            
        # Filter based on variance and add to synthetic buffer
        for i in range(num_samples):
            if variances[i].item() < self.variance_threshold:
                s = states[i].cpu().numpy()
                a = actions[i]
                next_s = mean_next_states[i].cpu().numpy()
                
                # Ground-truth reward computation (requires environment access or a proxy)
                # If a strict reward func is provided, use it. Otherwise, fallback to a heuristic.
                r = env_reward_func(s, a, next_s) if env_reward_func else -1.0 # placeholder
                
                # Assume not done for 1-step rollouts
                self.remember_synthetic(s, a, r, next_s, False)

    def replay(self):
        """
        Override replay to sample from both real and synthetic buffers using the real_ratio.
        """
        real_batch_size = int(self.batch_size * self.real_ratio)
        synthetic_batch_size = self.batch_size - real_batch_size
        
        if len(self.real_memory) < real_batch_size:
            return
            
        # If we don't have enough synthetic data yet, just use real data for the whole batch
        if len(self.synthetic_memory) < synthetic_batch_size:
            minibatch = random.sample(self.real_memory, self.batch_size)
        else:
            real_minibatch = random.sample(self.real_memory, real_batch_size)
            synthetic_minibatch = random.sample(self.synthetic_memory, synthetic_batch_size)
            minibatch = real_minibatch + synthetic_minibatch
            
        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(self.device)
        actions = torch.LongTensor([m[1] for m in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([m[2] for m in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch])).to(self.device)
        dones = torch.FloatTensor([float(m[4]) for m in minibatch]).to(self.device)
        
        # Standard DQN logic
        next_q = torch.nan_to_num(self.target_model(next_states).max(1)[0], nan=0.0, posinf=0.0, neginf=0.0)
        
        current_q = self.model(states).gather(1, actions).squeeze(1)
        target_q = rewards + (1 - dones) * self.gamma * next_q
        target_q = torch.nan_to_num(target_q, nan=0.0, posinf=0.0, neginf=0.0)
        
        loss = self.criterion(current_q, target_q.detach())
        self.last_loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # update target network periodically by step count
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.update_target_network()
