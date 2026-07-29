import numpy as np
import torch
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, pinn_preds, action, reward, next_state, next_pinn_preds, done):
        """
        Saves a transition, caching the PINN predictions to avoid impossible recomputation
        from the generic RL state during mini-batch sampling.
        
        state: (10,)
        pinn_preds: (4, 5) predictions for all 4 candidate actions
        action: int
        reward: float
        next_state: (10,)
        next_pinn_preds: (4, 5)
        done: bool
        """
        self.buffer.append((state, pinn_preds, action, reward, next_state, next_pinn_preds, done))

    def sample(self, batch_size):
        """
        Returns a random batch of transitions.
        """
        batch = random.sample(self.buffer, batch_size)
        
        state, pinn_preds, action, reward, next_state, next_pinn_preds, done = map(np.stack, zip(*batch))
        
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(pinn_preds),
            torch.LongTensor(action).unsqueeze(1),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(next_pinn_preds),
            torch.FloatTensor(done).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)
