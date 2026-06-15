import os
import sys
import numpy as np
import torch
import traci
import logging

# Add parent dir to path to import environments and PINN model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.sumo_environment import SumoEnvironment
from pinn_models.pinn_model import TrafficPINN
from utils.reliability import extract_state_dict, load_torch_checkpoint


logger = logging.getLogger(__name__)

class HybridSumoEnv(SumoEnvironment):
    def __init__(self, config_file, pinn_model_path, use_gui=False, num_seconds=3000, 
                 lookahead_steps=10, pinn_weight: float = 0.15, prediction_smoothing: float = 0.9,
                 seed: int | None = 42, pinn_confidence_threshold: float = 0.15):
        super().__init__(config_file, use_gui=use_gui, num_seconds=num_seconds, seed=seed)
        
        self.lookahead_steps = lookahead_steps
        self.pinn_weight = float(pinn_weight)
        self.prediction_smoothing = float(np.clip(prediction_smoothing, 0.0, 0.99))
        self._pinn_ema = np.zeros(2, dtype=np.float32)
        self.last_pinn_prediction = {
            'rho': 0.0,
            'velocity': 0.0,
            'confidence': 0.0,
            'effective_weight': 0.0,
            'loaded': False,
        }
        # PINN confidence gating: if PINN predictions are inconsistent (low confidence), scale down influence
        self.pinn_confidence_threshold = float(np.clip(pinn_confidence_threshold, 0.0, 1.0))
        
        # Load PINN Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Attempt to load PINN if provided; otherwise run without PINN
        base_state_dim = 7
        try:
            if pinn_model_path and os.path.exists(pinn_model_path):
                self.pinn = TrafficPINN().to(self.device)
                self.pinn.eval()
                checkpoint = load_torch_checkpoint(pinn_model_path, map_location=self.device)
                self.pinn.load_state_dict(extract_state_dict(checkpoint))
                self.edge_map = checkpoint.get('edge_map', {})
                self.t_max = checkpoint.get('t_max', 1.0)
                self.x_max = checkpoint.get('x_max', 1.0)
                self.last_pinn_prediction['loaded'] = True
                logger.info('Loaded pretrained PINN checkpoint from %s', pinn_model_path)
                # Original state dim is 7. We append predicted_density and predicted_velocity (2 dims)
                self.state_dim = base_state_dim + 2
            else:
                raise FileNotFoundError(f'PINN checkpoint not found: {pinn_model_path}')
        except Exception:
            # Fallback: run without PINN
            logger.warning('Proceeding without PINN (fallback). Reason: %s', pinn_model_path)
            self.pinn = None
            self.edge_map = {}
            self.t_max = 1.0
            self.x_max = 1.0
            self.last_pinn_prediction['loaded'] = False
            self.state_dim = base_state_dim

        # Reinitialize observation normalizer to match the resolved state dim
        try:
            self._init_obs_normalizer(self.state_dim)
        except Exception:
            logger.exception('Failed to reinitialize observation normalizer for HybridSumoEnv')

    def _get_state(self):
        base_raw_state = super()._get_raw_state()
        # If PINN not loaded, return base normalized observation
        if not self.last_pinn_prediction.get('loaded', False) or self.pinn is None:
            return self._normalize_observation(base_raw_state)
        
        # 2. Get PINN prediction
        if self.controlled_ts is None:
            return self._normalize_observation(np.zeros(self.state_dim, dtype=np.float32))
            
        # Get lanes controlled by TLS
        lanes = traci.trafficlight.getControlledLanes(self.controlled_ts)
        edges = set([traci.lane.getEdgeID(lane) for lane in lanes])
        
        # Prepare inputs for PINN (t + dt, x)
        future_t = self.current_step + self.lookahead_steps
        norm_t = future_t / max(self.t_max, 1)
        
        pred_densities = []
        pred_velocities = []
        
        with torch.no_grad():
            for edge in edges:
                if edge in self.edge_map:
                    norm_x = self.edge_map[edge] / max(self.x_max, 1)
                    
                    t_tensor = torch.tensor([[norm_t]], dtype=torch.float32).to(self.device)
                    x_tensor = torch.tensor([[norm_x]], dtype=torch.float32).to(self.device)
                    
                    rho_pred, v_pred = self.pinn(t_tensor, x_tensor)
                    pred_densities.append(rho_pred.item())
                    pred_velocities.append(v_pred.item())
        
        if len(pred_densities) > 0:
            avg_pred_rho = sum(pred_densities) / len(pred_densities)
            avg_pred_v = sum(pred_velocities) / len(pred_velocities)
        else:
            avg_pred_rho = 0.0
            avg_pred_v = 0.0
            
        # 3. Normalize PINN outputs using model targets to avoid dominating observations
        try:
            rho_norm = float(avg_pred_rho) / float(self.pinn.target_rhomax)
            v_norm = float(avg_pred_v) / float(self.pinn.target_vmax)
        except Exception:
            rho_norm = float(avg_pred_rho)
            v_norm = float(avg_pred_v) / 20.0

        pinn_signal = np.array([
            float(np.clip(rho_norm, 0.0, 1.0)),
            float(np.clip(v_norm, 0.0, 1.0)),
        ], dtype=np.float32)
        # compute simple confidence as inverse of std across predictions
        try:
            rho_std = float(np.std(pred_densities)) if len(pred_densities) > 1 else 0.0
            v_std = float(np.std(pred_velocities)) if len(pred_velocities) > 1 else 0.0
            conf = 1.0 - min(1.0, (rho_std + v_std))
        except Exception:
            conf = 1.0

        self._pinn_ema = self.prediction_smoothing * self._pinn_ema + (1.0 - self.prediction_smoothing) * pinn_signal
        # apply confidence gating: if confidence low, reduce pinn influence
        effective_weight = self.pinn_weight if conf >= self.pinn_confidence_threshold else (self.pinn_weight * conf)
        pinn_signal = effective_weight * pinn_signal + (1.0 - effective_weight) * self._pinn_ema

        self.last_pinn_prediction = {
            'rho': float(avg_pred_rho),
            'velocity': float(avg_pred_v),
            'confidence': float(conf),
            'effective_weight': float(effective_weight),
            'loaded': True,
            'rho_norm': float(rho_norm),
            'v_norm': float(v_norm),
        }

        hybrid_state = np.concatenate((base_raw_state, pinn_signal.astype(np.float32)))
        return self._normalize_observation(hybrid_state)
