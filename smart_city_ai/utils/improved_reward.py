#!/usr/bin/env python3
"""
Improved Reward Function for Traffic Signal Control

This module provides enhanced reward functions that better balance:
- Queue reduction (minimize vehicles waiting)
- Wait time reduction (minimize time vehicles spend waiting)
- Throughput maximization (maximize vehicles passing through)
- Speed maintenance (keep average speed high)
- Signal stability (avoid excessive switching)
"""
import numpy as np


class ImprovedRewardFunction:
    """Multi-objective reward function with configurable weights."""
    
    def __init__(self, 
                 weight_queue=0.30,      # Queue reduction
                 weight_wait=0.25,       # Wait time reduction  
                 weight_throughput=0.25, # Vehicle throughput
                 weight_speed=0.10,      # Average speed
                 weight_stability=0.10,  # Signal switching penalty
                 normalize=True):
        """
        Initialize reward function with configurable weights.
        
        Args:
            weight_queue: Priority for queue reduction (0-1)
            weight_wait: Priority for wait time reduction (0-1)
            weight_throughput: Priority for throughput (0-1)
            weight_speed: Priority for speed maintenance (0-1)
            weight_stability: Priority for signal stability (0-1)
            normalize: Whether to normalize weights to sum to 1.0
        """
        weights = {
            'queue': weight_queue,
            'wait': weight_wait,
            'throughput': weight_throughput,
            'speed': weight_speed,
            'stability': weight_stability,
        }
        
        # Normalize weights
        if normalize:
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}
        
        self.weights = weights
        self.episode_step = 0
        self.reward_history = []
    
    def compute(self, metrics_dict):
        """
        Compute reward from traffic metrics.
        
        Args:
            metrics_dict: Dictionary with keys:
                - queue_delta: Change in queue (positive = improvement)
                - wait_delta: Change in wait time (positive = improvement)
                - throughput: Vehicles passed this step
                - avg_speed: Current average speed (m/s)
                - phase_persistence: How long phase has been active [0,1]
                - num_lanes: Number of controlled lanes
                - last_switch_penalty: Penalty if switched too early
                - max_speed: Maximum speed for normalization (default 20 m/s)
        
        Returns:
            float: Computed reward value
        """
        # Extract metrics with safe defaults
        queue_delta = float(metrics_dict.get('queue_delta', 0.0))
        wait_delta = float(metrics_dict.get('wait_delta', 0.0))
        throughput = float(metrics_dict.get('throughput', 0.0))
        avg_speed = float(metrics_dict.get('avg_speed', 0.0))
        phase_persistence = float(metrics_dict.get('phase_persistence', 0.0))
        num_lanes = float(metrics_dict.get('num_lanes', 1.0))
        last_switch_penalty = float(metrics_dict.get('last_switch_penalty', 0.0))
        max_speed = float(metrics_dict.get('max_speed', 20.0))
        
        # Component 1: Queue reduction reward
        # Strongly reward reducing queue length
        r_queue = np.tanh(queue_delta / max(0.1, num_lanes * 2.0))
        
        # Component 2: Wait time reduction reward
        # Reward reducing cumulative wait time
        r_wait = np.tanh(wait_delta / max(0.1, num_lanes * 30.0))
        
        # Component 3: Throughput reward
        # Reward vehicles passing through
        # Normalize by number of lanes and apply smooth non-linearity
        throughput_norm = np.tanh(throughput / max(0.1, num_lanes * 0.5))
        r_throughput = throughput_norm
        
        # Component 4: Speed maintenance reward
        # Reward keeping average speed high
        speed_ratio = np.clip(avg_speed / max(0.1, max_speed), 0.0, 1.0)
        r_speed = speed_ratio - 0.5  # Centered at 0.5 (neutral)
        
        # Component 5: Signal stability (switching penalty)
        # Penalize frequent switches but reward phase persistence
        r_stability = -1.0 * last_switch_penalty + 0.1 * np.tanh(phase_persistence)
        
        # Combine all components with learned weights
        components = {
            'queue': r_queue,
            'wait': r_wait,
            'throughput': r_throughput,
            'speed': r_speed,
            'stability': r_stability,
        }
        
        reward = sum(self.weights[key] * components[key] for key in components.keys())
        
        # Smooth with exponential moving average to reduce noise
        ema_alpha = 0.1
        if len(self.reward_history) > 0:
            ema = self.reward_history[-1] * (1 - ema_alpha) + reward * ema_alpha
        else:
            ema = reward
        
        self.reward_history.append(ema)
        self.episode_step += 1
        
        # Clip to reasonable range
        return float(np.clip(ema, -2.0, 2.0))
    
    def reset_episode(self):
        """Reset for new episode."""
        self.episode_step = 0
        self.reward_history = []


class AdaptiveRewardFunction:
    """
    Adaptive reward function that adjusts weights based on training progress.
    
    Early training: Focus on throughput and speed
    Mid training: Balance all objectives equally
    Late training: Focus on queue and wait time minimization
    """
    
    def __init__(self):
        self.episode = 0
        self.base_function = ImprovedRewardFunction(
            weight_queue=0.30,
            weight_wait=0.25,
            weight_throughput=0.25,
            weight_speed=0.10,
            weight_stability=0.10,
        )
    
    def update_episode(self, episode_num):
        """Update reward weights based on training progress."""
        self.episode = episode_num
        total_episodes = 300  # Assume 300 episode training
        progress = min(1.0, float(episode_num) / total_episodes)
        
        # Gradually shift from speed/throughput to queue/wait
        if progress < 0.3:
            # Early: Focus on exploration and basic throughput
            weights = {
                'queue': 0.15,
                'wait': 0.15,
                'throughput': 0.35,
                'speed': 0.25,
                'stability': 0.10,
            }
        elif progress < 0.7:
            # Mid: Balance all objectives
            weights = {
                'queue': 0.30,
                'wait': 0.25,
                'throughput': 0.25,
                'speed': 0.12,
                'stability': 0.08,
            }
        else:
            # Late: Optimize for queue and wait time
            weights = {
                'queue': 0.40,
                'wait': 0.35,
                'throughput': 0.10,
                'speed': 0.10,
                'stability': 0.05,
            }
        
        # Apply new weights
        total = sum(weights.values())
        self.base_function.weights = {k: v / total for k, v in weights.items()}
    
    def compute(self, metrics_dict):
        """Compute reward using current weights."""
        return self.base_function.compute(metrics_dict)
    
    def reset_episode(self):
        """Reset for new episode."""
        self.base_function.reset_episode()


# Predefined configurations for different objectives
REWARD_CONFIGS = {
    'balanced': {
        'weight_queue': 0.30,
        'weight_wait': 0.25,
        'weight_throughput': 0.25,
        'weight_speed': 0.10,
        'weight_stability': 0.10,
    },
    'throughput_focused': {
        'weight_queue': 0.20,
        'weight_wait': 0.15,
        'weight_throughput': 0.40,
        'weight_speed': 0.15,
        'weight_stability': 0.10,
    },
    'wait_focused': {
        'weight_queue': 0.25,
        'weight_wait': 0.40,
        'weight_throughput': 0.15,
        'weight_speed': 0.10,
        'weight_stability': 0.10,
    },
    'speed_focused': {
        'weight_queue': 0.20,
        'weight_wait': 0.20,
        'weight_throughput': 0.20,
        'weight_speed': 0.30,
        'weight_stability': 0.10,
    },
    'stability_focused': {
        'weight_queue': 0.25,
        'weight_wait': 0.25,
        'weight_throughput': 0.25,
        'weight_speed': 0.10,
        'weight_stability': 0.15,
    },
}


def create_reward_function(config_name='balanced'):
    """
    Create reward function from predefined configuration.
    
    Args:
        config_name: Name of configuration ('balanced', 'throughput_focused', etc.)
    
    Returns:
        ImprovedRewardFunction instance
    """
    if config_name not in REWARD_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(REWARD_CONFIGS.keys())}")
    
    cfg = REWARD_CONFIGS[config_name]
    return ImprovedRewardFunction(**cfg)
