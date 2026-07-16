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
            _, _, _, self.x_scaler, self.y_scaler, _ = get_dataloaders()
            
            # Load the PINN model
            self.model = PINNModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM).to(self.device)
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
