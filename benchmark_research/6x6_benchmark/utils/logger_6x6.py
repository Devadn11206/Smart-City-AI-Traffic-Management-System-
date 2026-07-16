import csv
import os
import json
import time

def resilient_open_append(file_path, max_retries=15, delay=1.0):
    for i in range(max_retries):
        try:
            return open(file_path, 'a', newline='')
        except PermissionError:
            time.sleep(delay)
    import io
    return io.StringIO()

class MetricLogger:
    def __init__(self, base_dir, algo_name):
        self.algo_dir = os.path.join(base_dir, algo_name)
        os.makedirs(self.algo_dir, exist_ok=True)
        
        # Files
        self.ep_file = os.path.join(self.algo_dir, '6x6_episode_metrics.csv')
        self.ts_file = os.path.join(self.algo_dir, '6x6_intersection_metrics.csv')
        self.sar_file = os.path.join(self.algo_dir, '6x6_state_action_reward.csv')
        self.meta_file = os.path.join(self.algo_dir, 'metadata.json')
        
        self._init_files()
        
    def log_metadata(self, metadata):
        with open(self.meta_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
    def _init_files(self):
        # Episode metrics
        if not os.path.exists(self.ep_file):
            with open(self.ep_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'episode', 'reward', 'avg_waiting_time', 'avg_queue_length',
                    'throughput', 'avg_travel_time', 'fuel_consumption', 'avg_speed',
                    'loss', 'epsilon', 'learning_rate', 'vehicles_completed', 'vehicles_generated'
                ])
                
        # Intersection metrics
        if not os.path.exists(self.ts_file):
            with open(self.ts_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'episode', 'intersection_id', 'reward', 'avg_wait',
                    'avg_queue', 'throughput', 'current_phase', 'selected_action'
                ])
                
        # SAR metrics
        if not os.path.exists(self.sar_file):
            with open(self.sar_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'episode', 'intersection_id', 'step', 'state', 'action', 'reward', 'next_state'
                ])
                
    def log_episode(self, metrics):
        with resilient_open_append(self.ep_file) as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.get('episode', 0),
                metrics.get('reward', 0.0),
                metrics.get('avg_waiting_time', 0.0),
                metrics.get('avg_queue_length', 0.0),
                metrics.get('throughput', 0),
                metrics.get('avg_travel_time', 0.0),
                metrics.get('fuel_consumption', 0.0),
                metrics.get('avg_speed', 0.0),
                metrics.get('loss', 0.0),
                metrics.get('epsilon', 0.0),
                metrics.get('learning_rate', 0.0),
                metrics.get('vehicles_completed', 0),
                metrics.get('vehicles_generated', 0)
            ])
            
    def log_intersection(self, episode, ts_id, reward, avg_wait, avg_queue, throughput, current_phase, selected_action):
        with resilient_open_append(self.ts_file) as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, ts_id, reward, avg_wait, avg_queue, throughput, current_phase, selected_action
            ])
            
    def log_sar(self, episode, ts_id, step, state, action, reward, next_state):
        with resilient_open_append(self.sar_file) as f:
            writer = csv.writer(f)
            # convert states to list strings for csv
            s_str = json.dumps(state.tolist() if hasattr(state, 'tolist') else list(state))
            ns_str = json.dumps(next_state.tolist() if hasattr(next_state, 'tolist') else list(next_state))
            writer.writerow([
                episode, ts_id, step, s_str, action, reward, ns_str
            ])
