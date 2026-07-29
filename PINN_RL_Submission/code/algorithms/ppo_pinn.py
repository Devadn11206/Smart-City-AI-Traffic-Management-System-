import torch
import numpy as np
import time
import os
import sys

# Ensure benchmark_research is in the path to import pinn modules
current_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.dirname(current_dir)
if benchmark_dir not in sys.path:
    sys.path.append(benchmark_dir)
pinn_dir = os.path.join(benchmark_dir, 'pinn')
if pinn_dir not in sys.path:
    sys.path.append(pinn_dir)

from pinn.model import PINNModel
from pinn.config import INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM
from pinn.dataset import get_dataloaders
from algorithms.ppo import PPOAgent

class PINNStateWrapper:
    def __init__(self, mode='baseline', device='cpu'):
        self.mode = mode
        self.device = torch.device(device)
        self.inference_times = []
        
        if self.mode != 'baseline':
            # Load the exact scalers used during training
            # get_dataloaders is lightweight enough to run once on init
            _, _, _, self.x_scaler, self.y_scaler, _ = get_dataloaders(action_aware=False)
            
            # Load the PINN model
            self.model = PINNModel(input_dim=7, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM).to(self.device)
            # Find pinn_model.pth. It might be in pinn/models/ or models/
            model_path1 = os.path.join(benchmark_dir, 'models', 'pinn_model.pth')
            model_path2 = os.path.join(benchmark_dir, 'pinn', 'models', 'pinn_model.pth')
            
            model_path = model_path1 if os.path.exists(model_path1) else model_path2
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            # Track phases for duration calculation
            self.tracked_phase = {}
            self.tracked_duration = {}

    def get_augmented_state_dim(self):
        # Baseline original state is 10-dim
        if self.mode == 'baseline':
            return 10
        elif self.mode == 'queue':
            return 10 + 1
        elif self.mode == 'queue_density':
            return 10 + 2
        elif self.mode == 'all':
            return 10 + 5
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def reset(self):
        self.tracked_phase = {}
        self.tracked_duration = {}

    def extract_and_predict(self, ts, env, infos, original_state):
        if self.mode == 'baseline':
            return original_state

        import traci
        
        # 1. Extract 7-dim PINN features
        if ts not in self.tracked_phase:
            self.tracked_phase[ts] = -1
            self.tracked_duration[ts] = 0.0

        lanes = env.tl_in_lanes[ts]
        all_lanes = [l for d in lanes.values() for l in d]
        
        density = 0.0
        queue = 0.0
        speed = 0.0
        fuel = 0.0
        throughput = 0.0
        
        if len(all_lanes) > 0:
            total_length = sum([traci.lane.getLength(l) for l in all_lanes])
            density = sum([traci.lane.getLastStepVehicleNumber(l) for l in all_lanes]) / (total_length + 1e-5)
            queue = sum([traci.lane.getLastStepHaltingNumber(l) for l in all_lanes])
            speed = np.mean([traci.lane.getLastStepMeanSpeed(l) for l in all_lanes])
            fuel = sum([traci.lane.getFuelConsumption(l) for l in all_lanes])
            
            if infos is not None and ts in infos:
                throughput = infos[ts]['throughput']
            
            phase = traci.trafficlight.getPhase(ts)
            if phase == self.tracked_phase[ts]:
                self.tracked_duration[ts] += env.delta_time
            else:
                self.tracked_phase[ts] = phase
                self.tracked_duration[ts] = 0.0
                
        # Construct 7-dim PINN input
        # ['traffic_density', 'queue_length', 'throughput', 'avg_speed', 'fuel_consumption', 'current_phase', 'phase_duration']
        pinn_input_raw = np.array([[density, queue, throughput, speed, fuel, self.tracked_phase[ts], self.tracked_duration[ts]]])
        
        # 2. Normalize and Infer
        start_time = time.time()
        
        pinn_input_scaled = self.x_scaler.transform(pinn_input_raw)
        pinn_input_tensor = torch.FloatTensor(pinn_input_scaled).to(self.device)
        
        with torch.no_grad():
            pred_scaled = self.model(pinn_input_tensor)
            
        pred_scaled_np = pred_scaled.cpu().numpy()
        pred_raw = self.y_scaler.inverse_transform(pred_scaled_np)[0]
        
        end_time = time.time()
        self.inference_times.append(end_time - start_time)
        
        # PINN outputs (5-dim): ['traffic_density', 'queue_length', 'throughput', 'avg_speed', 'fuel_consumption']
        pred_density = pred_raw[0]
        pred_queue = pred_raw[1]
        pred_throughput = pred_raw[2]
        pred_speed = pred_raw[3]
        pred_fuel = pred_raw[4]
        
        # 3. Augment State
        augmented_state = list(original_state)
        
        if self.mode == 'queue':
            augmented_state.append(pred_queue)
        elif self.mode == 'queue_density':
            augmented_state.extend([pred_queue, pred_density])
        elif self.mode == 'all':
            augmented_state.extend([pred_density, pred_queue, pred_throughput, pred_speed, pred_fuel])
            
        return np.array(augmented_state, dtype=np.float32)

    def get_avg_inference_time(self):
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times)


class ActionAwarePINNWrapper:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.inference_times = []
        
        # Load the exact scalers used during training (action_aware=True)
        _, _, _, self.x_scaler, self.y_scaler, _ = get_dataloaders(action_aware=True)
        
        # Load the Action-Aware PINN model
        # Input dim = 8 (density, queue, throughput, speed, fuel, phase, duration, action)
        # Output dim = 5 (next density, next queue, next throughput, next speed, next fuel)
        self.model = PINNModel(input_dim=8, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=5).to(self.device)
        
        # Find action_aware_pinn_model.pth
        model_path1 = os.path.join(benchmark_dir, 'models', 'action_aware_pinn_model.pth')
        model_path2 = os.path.join(benchmark_dir, 'pinn', 'models', 'action_aware_pinn_model.pth')
        model_path = model_path1 if os.path.exists(model_path1) else model_path2
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Track phases for duration calculation
        self.tracked_phase = {}
        self.tracked_duration = {}

    def reset(self):
        self.tracked_phase = {}
        self.tracked_duration = {}

    def get_predictive_advice(self, ts, env, last_infos, original_state, ppo_probs, alpha=0.5, tau=0.1):
        """
        Predicts future traffic states for all 4 candidate actions, computes cost in normalized space,
        and blends PPO probabilities with the advisor scores.
        """
        import traci
        
        # 1. Extract raw current features
        if ts not in self.tracked_phase:
            self.tracked_phase[ts] = -1
            self.tracked_duration[ts] = 0.0

        lanes = env.tl_in_lanes[ts]
        all_lanes = [l for d in lanes.values() for l in d]
        
        density = 0.0
        queue = 0.0
        speed = 0.0
        fuel = 0.0
        throughput = 0.0
        
        if len(all_lanes) > 0:
            total_length = sum([traci.lane.getLength(l) for l in all_lanes])
            density = sum([traci.lane.getLastStepVehicleNumber(l) for l in all_lanes]) / (total_length + 1e-5)
            queue = sum([traci.lane.getLastStepHaltingNumber(l) for l in all_lanes])
            speed = np.mean([traci.lane.getLastStepMeanSpeed(l) for l in all_lanes])
            fuel = sum([traci.lane.getFuelConsumption(l) for l in all_lanes])
            
            if last_infos is not None and ts in last_infos:
                throughput = last_infos[ts]['throughput']
            
            phase = traci.trafficlight.getPhase(ts)
            if phase == self.tracked_phase[ts]:
                self.tracked_duration[ts] += env.delta_time
            else:
                self.tracked_phase[ts] = phase
                self.tracked_duration[ts] = 0.0
                
        # Get normalized waiting time from original state: mean of indices 4,5,6,7
        # original_state is normalized in [0, 1] for waiting times (W_N, W_S, W_E, W_W)
        wait_scaled = np.mean(original_state[4:8])
        
        # 2. Formulate 4 candidate action inputs
        raw_inputs = []
        for action in range(4):
            raw_input = [density, queue, throughput, speed, fuel, self.tracked_phase[ts], self.tracked_duration[ts], action]
            raw_inputs.append(raw_input)
            
        raw_inputs_np = np.array(raw_inputs)
        
        start_time = time.time()
        
        # Scale and run through model
        scaled_inputs = self.x_scaler.transform(raw_inputs_np)
        inputs_tensor = torch.FloatTensor(scaled_inputs).to(self.device)
        
        with torch.no_grad():
            preds_scaled = self.model(inputs_tensor)
            
        preds_scaled_np = preds_scaled.cpu().numpy()
        
        # 3. Compute cost directly in normalized space
        # Cost = Queue_scaled + Wait_scaled + Fuel_scaled - Throughput_scaled
        # Index mappings of outputs:
        # 0: density, 1: queue_length, 2: throughput, 3: avg_speed, 4: fuel_consumption
        costs = []
        for i in range(4):
            q_pred_scaled = preds_scaled_np[i, 1]
            tp_pred_scaled = preds_scaled_np[i, 2]
            fuel_pred_scaled = preds_scaled_np[i, 4]
            
            cost = q_pred_scaled + wait_scaled + fuel_pred_scaled - tp_pred_scaled
            costs.append(cost)
            
        costs_np = np.array(costs)
        # Softmax over negative costs to get probability scores
        exp_neg_costs = np.exp(-(costs_np - np.min(costs_np)) / tau)
        scores = exp_neg_costs / np.sum(exp_neg_costs)
        
        # 4. Blend PPO probs and Advisor scores
        combined_probs = alpha * ppo_probs + (1 - alpha) * scores
        combined_probs = combined_probs / np.sum(combined_probs)
        
        final_action = np.argmax(combined_probs)
        
        end_time = time.time()
        self.inference_times.append(end_time - start_time)
        
        return final_action, combined_probs[final_action]

    def get_avg_inference_time(self):
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times)

