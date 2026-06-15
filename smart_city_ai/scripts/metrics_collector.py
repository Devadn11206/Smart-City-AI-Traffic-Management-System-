import os
import sys
import traci

def collect_metrics(step):
    metrics = {
        'step': step,
        'vehicle_count': traci.vehicle.getIDCount(),
        'waiting_time': 0,
        'avg_speed': 0,
        'queue_length': 0
    }
    
    if metrics['vehicle_count'] > 0:
        speeds = []
        waiting_times = []
        for veh_id in traci.vehicle.getIDList():
            speeds.append(traci.vehicle.getSpeed(veh_id))
            waiting_times.append(traci.vehicle.getWaitingTime(veh_id))
            if traci.vehicle.getSpeed(veh_id) < 0.1:
                metrics['queue_length'] += 1
                
        metrics['avg_speed'] = sum(speeds) / len(speeds)
        metrics['waiting_time'] = sum(waiting_times)
        
    return metrics

def save_metrics(metrics_list, output_file):
    import csv
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['step', 'vehicle_count', 'waiting_time', 'avg_speed', 'queue_length']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in metrics_list:
            writer.writerow(data)
    print(f"Metrics saved to {output_file}")
