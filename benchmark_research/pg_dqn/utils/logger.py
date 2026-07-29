import os
import csv
import time

class PGDQNLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.csv_path = os.path.join(log_dir, 'metrics.csv')
        self.headers = [
            'Episode', 'Reward', 'AvgWait', 'Queue', 'Throughput', 
            'Fuel', 'Speed', 'TravelTime', 'Epsilon', 'Loss', 
            'AvgQ', 'TargetQ', 'TDError', 'BufferSize', 'GradNorm',
            'TrainTime', 'PINN_Inference_Time', 'DQN_Inference_Time'
        ]
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            
    def log_episode(self, episode, reward, wait, queue, tp, fuel, speed, tt, epsilon, loss, avg_q, target_q, td_error, buffer_size, grad_norm, train_time, pinn_time, dqn_time):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, reward, wait, queue, tp, fuel, speed, tt, epsilon, loss,
                avg_q, target_q, td_error, buffer_size, grad_norm,
                train_time, pinn_time, dqn_time
            ])
