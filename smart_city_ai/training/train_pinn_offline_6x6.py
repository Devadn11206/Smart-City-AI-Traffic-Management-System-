import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pinn_models.pinn_ensemble import PINNEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransitionDataset(Dataset):
    def __init__(self, data_file):
        data = torch.load(data_file, weights_only=False)
        self.states = torch.FloatTensor([d['state'] for d in data])
        self.actions = torch.FloatTensor([d['action'] for d in data])
        self.next_states = torch.FloatTensor([d['next_state'] for d in data])
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx]

def train_pinn_offline_6x6(data_file='pinn_dataset_6x6.pt', epochs=50, batch_size=256):
    if not os.path.exists(data_file):
        logger.error(f"Dataset {data_file} not found. Run generate_pinn_dataset_6x6.py first!")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    dataset = TransitionDataset(data_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    state_dim = dataset.states.shape[1]
    action_dim = 4 
    
    ensemble = PINNEnsemble(state_dim=state_dim, action_dim=action_dim, num_models=3, device=device)
    
    optimizers = [optim.Adam(model.parameters(), lr=1e-3) for model in ensemble.models]
    mse_criterion = nn.MSELoss()
    
    logger.info("Starting Offline PINN Pre-training for 6x6 Network...")
    
    for epoch in range(epochs):
        total_loss = 0
        total_mse = 0
        total_phys = 0
        
        for states, actions, next_states in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            next_states = next_states.to(device)
            
            batch_loss = 0
            
            for i, model in enumerate(ensemble.models):
                optimizers[i].zero_grad()
                pred_next_states = model(states, actions)
                mse_loss = mse_criterion(pred_next_states, next_states)
                phys_loss = model.physics_loss(states, pred_next_states)
                
                loss = mse_loss + 0.1 * phys_loss
                loss.backward()
                optimizers[i].step()
                
                batch_loss += loss.item()
                if i == 0: 
                    total_mse += mse_loss.item()
                    total_phys += phys_loss.item()
                    
            total_loss += batch_loss / 3.0 
            
        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        avg_phys = total_phys / len(dataloader)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | Phys: {avg_phys:.4f}")
            
    save_path = "pinn_ensemble_pretrained_6x6.pth"
    torch.save(ensemble.state_dict(), save_path)
    logger.info(f"Pre-training complete! Saved global physics weights to {save_path}")

if __name__ == "__main__":
    train_pinn_offline_6x6()
