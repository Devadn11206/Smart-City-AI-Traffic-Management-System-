import os
import torch

class CheckpointManager:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self.best_reward = -float('inf')
        self.best_wait = float('inf')

    def save(self, agent, episode, reward, wait):
        checkpoint_data = {
            'model_state_dict': agent.q_network.state_dict(),
            'episode': episode,
            'epsilon': agent.epsilon
        }
        
        # Save latest
        torch.save(checkpoint_data, os.path.join(self.save_dir, 'latest_model.pth'))
        
        # Save every 5 episodes
        if episode % 5 == 0:
            torch.save(checkpoint_data, os.path.join(self.save_dir, f'model_ep{episode}.pth'))
            
        # Save best reward
        if reward > self.best_reward:
            self.best_reward = reward
            torch.save(checkpoint_data, os.path.join(self.save_dir, 'best_reward_model.pth'))
            
        # Save best waiting time
        if wait < self.best_wait:
            self.best_wait = wait
            torch.save(checkpoint_data, os.path.join(self.save_dir, 'best_wait_model.pth'))
