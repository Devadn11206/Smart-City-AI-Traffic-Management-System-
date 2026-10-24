import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import os

from ddqn_pinn_package.rl_agents.ddqn_agent import DDQNAgent

class PIDDQNAgent(DDQNAgent):
    """
    Advanced Physics-Informed Double Deep Q-Network Agent (Dual-Use).
    1. Uses DDQN logic to prevent overestimation.
    2. Uses PINN to generate synthetic physical rollouts.
    3. Adds Physics-Informed loss penalty.
    """
    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01,
                 replay_size=300000, batch_size=128, target_update=2000, device: str | None = None,
                 pinn_ensemble=None, variance_threshold=0.1, real_ratio=0.25, lambda_physics=0.1):
        
        super().__init__(state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, epsilon_min,
                         replay_size, batch_size, target_update, device)
        
        self.pinn_ensemble = pinn_ensemble
        self.variance_threshold = variance_threshold
        self.lambda_physics = lambda_physics
        
        # Dual Buffers for Model-Based Reinforcement Learning
        self.real_memory = deque(maxlen=replay_size)
        self.synthetic_memory = deque(maxlen=replay_size)
        
        self.real_ratio = real_ratio
        
    def remember(self, state, action, reward, next_state, done):
        """Save real experiences to the real buffer."""
        self.real_memory.append((state, action, reward, next_state, done))
        
    def remember_synthetic(self, state, action, reward, next_state, done):
        """Save PINN-imagined experiences to the synthetic buffer."""
        self.synthetic_memory.append((state, action, reward, next_state, done))
        
    def generate_synthetic_data(self, num_samples=10, env_reward_func=None):
        """
        Dual-Use Feature 1: Surrogate Environment Rollouts.
        Uses the PINN to predict the next state for sample augmentation.
        """
        if self.pinn_ensemble is None or len(self.real_memory) < num_samples:
            return
            
        minibatch = random.sample(self.real_memory, num_samples)
        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(self.device)
        
        actions = []
        for state in states:
            if random.random() <= self.epsilon:
                actions.append(random.randrange(self.action_dim))
            else:
                with torch.no_grad():
                    q_values = self.model(state.unsqueeze(0))
                actions.append(torch.argmax(q_values).item())
                
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        
        with torch.no_grad():
            mean_next_states, variances = self.pinn_ensemble(states, actions_tensor)
            
        for i in range(num_samples):
            if variances[i].item() < self.variance_threshold:
                s = states[i].cpu().numpy()
                a = actions[i]
                next_s = mean_next_states[i].cpu().numpy()
                r = env_reward_func(s, a, next_s) if env_reward_func else -1.0 
                self.remember_synthetic(s, a, r, next_s, False)

    def replay(self):
        """
        Dual-Use Feature 2: DDQN Logic with Physics Penalty.
        """
        real_batch_size = int(self.batch_size * self.real_ratio)
        synthetic_batch_size = self.batch_size - real_batch_size
        
        if len(self.real_memory) < real_batch_size:
            return
            
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
        
        # DOUBLE DQN LOGIC 
        # Action selection by online network, evaluation by target network
        with torch.no_grad():
            best_next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            next_q = self.target_model(next_states).gather(1, best_next_actions).squeeze(1)
            next_q = torch.nan_to_num(next_q, nan=0.0, posinf=0.0, neginf=0.0)
            
        target_q = rewards + (1 - dones) * self.gamma * next_q
        target_q = torch.nan_to_num(target_q, nan=0.0, posinf=0.0, neginf=0.0)
        
        current_q = self.model(states).gather(1, actions).squeeze(1)
        
        # Standard TD Loss
        td_loss = self.criterion(current_q, target_q.detach())
        
        # PHYSICS LOSS PENALTY
        # Query PINN to see if the DDQN's greedy action leads to high-variance (physically unsafe) state
        physics_loss = 0.0
        if self.pinn_ensemble is not None:
            greedy_actions = self.model(states).max(1)[1].float()
            # We don't backprop through the PINN, we use its variance as a penalty weight
            with torch.no_grad():
                _, variances = self.pinn_ensemble(states, greedy_actions)
            # Penalize Q-values that cause high variance in the physical surrogate
            physics_penalty = (variances.mean() * current_q.mean()).abs()
            physics_loss = self.lambda_physics * physics_penalty
            
        total_loss = td_loss + physics_loss
        self.last_loss = total_loss.item()
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.update_target_network()
