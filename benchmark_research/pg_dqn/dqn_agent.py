import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import sys
import traci

# Setup paths for benchmark and PINN
current_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.dirname(current_dir)
pinn_dir = os.path.join(benchmark_dir, 'pinn')

if benchmark_dir not in sys.path:
    sys.path.append(benchmark_dir)
if pinn_dir not in sys.path:
    sys.path.append(pinn_dir)

from pinn.model import PINNModel
from pinn.config import HIDDEN_DIM, NUM_LAYERS
from pinn.dataset import get_dataloaders
from pg_dqn.pg_dqn_networks import AC_PG_DQN

class AC_PG_DQNAgent:
    def __init__(self, state_dim, action_dim, physics_dim, lr, gamma, epsilon_start, epsilon_end, epsilon_decay, device='cpu', ablation_mode=2, use_ddqn=False):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.physics_dim = physics_dim
        self.gamma = gamma
        self.ablation_mode = ablation_mode
        self.use_ddqn = use_ddqn
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.q_network = AC_PG_DQN(state_dim, physics_dim, mode=ablation_mode).to(self.device)
        self.target_network = AC_PG_DQN(state_dim, physics_dim, mode=ablation_mode).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self._load_frozen_pinn()

    def _load_frozen_pinn(self):
        """Loads the pretrained Action-Aware PINN and scalers."""
        _, _, _, self.x_scaler, self.y_scaler, _ = get_dataloaders(action_aware=True)
        self.pinn = PINNModel(input_dim=10, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=5).to(self.device)
        
        model_path1 = os.path.join(benchmark_dir, 'models', 'action_aware_pinn_model.pth')
        model_path2 = os.path.join(benchmark_dir, 'pinn', 'models', 'action_aware_pinn_model.pth')
        model_path = model_path1 if os.path.exists(model_path1) else model_path2
        
        self.pinn.load_state_dict(torch.load(model_path, map_location=self.device))
        self.pinn.eval()
        for p in self.pinn.parameters():
            p.requires_grad = False

    def extract_pinn_inputs(self, env, ts_ids, last_infos):
        """
        Extracts the required 10-dimensional physics features for the PINN
        across all intersections and all candidate actions.
        Returns a batched numpy array of shape (num_agents * 4, 10).
        """
        all_raw_inputs = []
        for ts in ts_ids:
            lanes = env.tl_in_lanes[ts]
            lanes_N = lanes['N']
            lanes_S = lanes['S']
            lanes_E = lanes['E']
            lanes_W = lanes['W']
            all_lanes = lanes_N + lanes_S + lanes_E + lanes_W
            
            density = queue = speed = fuel = throughput = 0.0
            
            if len(all_lanes) > 0:
                total_length = sum([traci.lane.getLength(l) for l in all_lanes])
                density = sum([traci.lane.getLastStepVehicleNumber(l) for l in all_lanes]) / (total_length + 1e-5)
                queue = sum([traci.lane.getLastStepHaltingNumber(l) for l in all_lanes])
                speed = np.mean([traci.lane.getLastStepMeanSpeed(l) for l in all_lanes])
                fuel = sum([traci.lane.getFuelConsumption(l) for l in all_lanes])
                
                if last_infos is not None and ts in last_infos:
                    throughput = last_infos[ts].get('throughput', 0.0)
                
            phase = traci.trafficlight.getPhase(ts)
            duration = env.phase_duration[ts] # approximation of tracked duration
            
            for action in range(4):
                green_lanes = lanes_N + lanes_S if action in [0, 1] else lanes_E + lanes_W
                if len(green_lanes) > 0:
                    action_queue = sum([traci.lane.getLastStepHaltingNumber(l) for l in green_lanes])
                    action_vehicles = sum([traci.lane.getLastStepVehicleNumber(l) for l in green_lanes])
                else:
                    action_queue = 0
                    action_vehicles = 0
                    
                raw_input = [density, queue, throughput, speed, fuel, phase, duration, action, action_queue, action_vehicles]
                all_raw_inputs.append(raw_input)
                
        return np.array(all_raw_inputs, dtype=np.float32)

    def get_pinn_predictions(self, env, ts_ids, last_infos, episode=None):
        """
        Runs batched inference through the frozen PINN for all candidate actions.
        Returns predictions of shape (num_agents, 4, 5).
        """
        raw_inputs = self.extract_pinn_inputs(env, ts_ids, last_infos)
        scaled_inputs = self.x_scaler.transform(raw_inputs)
        inputs_tensor = torch.FloatTensor(scaled_inputs).to(self.device)
        
        with torch.no_grad():
            preds_scaled = self.pinn(inputs_tensor)
            
        preds_scaled_np = preds_scaled.cpu().numpy()
        
        if episode is not None and episode % 20 == 0 and not hasattr(self, f'_logged_ep_{episode}'):
            setattr(self, f'_logged_ep_{episode}', True)
            if 'TL12' in ts_ids:
                idx = ts_ids.index('TL12')
                # Unscale just for logging
                log_preds = self.y_scaler.inverse_transform(preds_scaled_np[idx*4:(idx+1)*4])
                log_preds[:, 2] = log_preds[:, 2] ** 2 # Reverse square root for throughput
                
                print(f"\n--- PINN Diagnostic (TL12) Ep {episode} ---")
                for a in range(4):
                    print(f"Action {a}: {log_preds[a]}")
                
                std_devs = np.std(log_preds, axis=0)
                print(f"Std Devs across actions: {std_devs}")
                if np.all(std_devs < 1e-3):
                    print("WARNING: PINN predictions have very low variance across actions. The PINN is not providing useful action discrimination.")
                print("---------------------------------------\n")
        
        return preds_scaled_np.reshape(len(ts_ids), 4, 5)

    def act(self, states, pinn_preds):
        """
        Returns epsilon-greedy actions for a batch of intersections.
        states: (num_agents, 10) tensor
        pinn_preds: (num_agents, 4, 5) tensor
        """
        import time
        t0 = time.time()
        num_agents = states.shape[0]
        
        if random.random() < self.epsilon:
            return [random.randint(0, self.action_dim - 1) for _ in range(num_agents)], time.time() - t0
        
        with torch.no_grad():
            states_expanded = states.repeat_interleave(self.action_dim, dim=0)
            preds_flat = pinn_preds.view(-1, self.physics_dim)
            
            q_values = self.q_network(states_expanded, preds_flat)
            q_values = q_values.view(num_agents, self.action_dim)
            
            actions = q_values.argmax(dim=1).cpu().numpy()
            
        return actions.tolist(), time.time() - t0

    def train_step(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        states, pinn_preds, actions, rewards, next_states, next_pinn_preds, dones = replay_buffer.sample(batch_size)
        
        states = states.to(self.device)
        pinn_preds = pinn_preds.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        next_pinn_preds = next_pinn_preds.to(self.device)
        dones = dones.to(self.device)
        
        action_idx = actions.unsqueeze(-1).expand(-1, -1, self.physics_dim)
        taken_pinn_preds = pinn_preds.gather(1, action_idx).squeeze(1)
        
        current_q = self.q_network(states, taken_pinn_preds)
        avg_q = current_q.mean().item()
        
        with torch.no_grad():
            next_states_expanded = next_states.repeat_interleave(self.action_dim, dim=0) # (B*4, 10)
            next_preds_flat = next_pinn_preds.view(-1, self.physics_dim) # (B*4, 5)
            
            target_q_values = self.target_network(next_states_expanded, next_preds_flat) # (B*4, 1)
            target_q_values = target_q_values.view(batch_size, self.action_dim) # (B, 4)
            
            if self.use_ddqn:
                current_q_next = self.q_network(next_states_expanded, next_preds_flat).view(batch_size, self.action_dim)
                best_actions = current_q_next.argmax(dim=1, keepdim=True)
                max_target_q = target_q_values.gather(1, best_actions)
            else:
                max_target_q = target_q_values.max(dim=1, keepdim=True)[0]
                
            expected_q = rewards + (1 - dones) * self.gamma * max_target_q
            avg_target_q = max_target_q.mean().item()
            
        loss = self.loss_fn(current_q, expected_q)
        td_error = torch.abs(current_q - expected_q).mean().item()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        total_norm = 0.0
        for p in self.q_network.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5
        
        self.optimizer.step()
        
        return loss.item(), avg_q, avg_target_q, td_error, grad_norm

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
