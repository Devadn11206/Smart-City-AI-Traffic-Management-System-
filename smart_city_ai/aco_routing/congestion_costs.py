import numpy as np


DEFAULT_COST_WEIGHTS = {
    'travel_time': 1.0,
    'queue_length': 1.0,
    'waiting_time': 1.0,
    'traffic_density': 1.0,
    'future_congestion': 2.0,
}

def default_edge_heuristic(edge_id, edge_info=None, pinn_pred=None):
    """Return a heuristic cost for an edge.

    edge_info: optional dict providing static attributes like length, speed
    pinn_pred: predicted congestion metric for the edge (0..1 where 1 is congested)
    """
    length = 1.0
    if edge_info is not None:
        length = edge_info.get('length', 1.0)
    pred = pinn_pred if pinn_pred is not None else 0.0
    # heuristic combines length and predicted congestion
    # longer edges and higher predicted congestion have higher cost
    return length * (1.0 + float(pred) * 5.0)


def live_edge_metrics(edge_id):
    """Return live edge metrics from TraCI if available."""
    try:
        import traci
        metrics = {
            'travel_time': float(traci.edge.getTraveltime(edge_id)) if hasattr(traci.edge, 'getTraveltime') else 0.0,
            'queue_length': float(traci.edge.getLastStepHaltingNumber(edge_id)) if hasattr(traci.edge, 'getLastStepHaltingNumber') else 0.0,
            'waiting_time': float(traci.edge.getWaitingTime(edge_id)) if hasattr(traci.edge, 'getWaitingTime') else 0.0,
            'traffic_density': float(traci.edge.getLastStepVehicleNumber(edge_id)) if hasattr(traci.edge, 'getLastStepVehicleNumber') else 0.0,
            'mean_speed': float(traci.edge.getLastStepMeanSpeed(edge_id)) if hasattr(traci.edge, 'getLastStepMeanSpeed') else 0.0,
            'occupancy': float(traci.edge.getLastStepOccupancy(edge_id)) if hasattr(traci.edge, 'getLastStepOccupancy') else 0.0,
        }
        return metrics
    except BaseException:
        # Catch BaseException to handle KeyboardInterrupt and other non-Exception errors
        # that may occur when TraCI connection becomes unstable or user interrupts
        return {
            'travel_time': 0.0,
            'queue_length': 0.0,
            'waiting_time': 0.0,
            'traffic_density': 0.0,
            'mean_speed': 0.0,
            'occupancy': 0.0,
        }


def weighted_edge_cost(edge_id, edge_info=None, live_metrics=None, pinn_pred=None, weights=None):
    """Combine travel time, queue length, waiting time, density and future congestion into one cost."""
    weights = dict(DEFAULT_COST_WEIGHTS, **(weights or {}))
    live_metrics = live_metrics or live_edge_metrics(edge_id)
    edge_info = edge_info or {}

    length = float(edge_info.get('length', 1.0))
    travel_time = float(live_metrics.get('travel_time', 0.0))
    queue_length = float(live_metrics.get('queue_length', 0.0))
    waiting_time = float(live_metrics.get('waiting_time', 0.0))
    density = float(live_metrics.get('traffic_density', 0.0))
    mean_speed = float(live_metrics.get('mean_speed', 0.0))
    occupancy = float(live_metrics.get('occupancy', 0.0))
    future_congestion = float(pinn_pred if pinn_pred is not None else 0.0)

    travel_component = weights['travel_time'] * (travel_time + length / max(mean_speed, 0.1))
    queue_component = weights['queue_length'] * queue_length
    wait_component = weights['waiting_time'] * waiting_time
    density_component = weights['traffic_density'] * (density + occupancy * 10.0)
    future_component = weights['future_congestion'] * future_congestion

    return float(travel_component + queue_component + wait_component + density_component + future_component)

def normalize_predictions(pred_dict):
    if not pred_dict:
        return {}
    vals = np.array(list(pred_dict.values()), dtype=np.float32)
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-6:
        return {k: 0.0 for k in pred_dict}
    out = {k: (v - vmin) / (vmax - vmin) for k, v in pred_dict.items()}
    return out
