from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional

from aco_routing.congestion_costs import weighted_edge_cost


@dataclass
class EmergencyETAResult:
    eta_seconds: float
    delay_probability: float
    route_risk: float
    congestion_impact: float
    corridor_efficiency: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class EmergencyETAPredictor:
    def __init__(self, edge_info: Optional[Dict[str, Dict[str, float]]] = None):
        self.edge_info = edge_info or {}

    def predict_position_at_time(self, route_edges: Iterable[str], target_time_seconds: float, pinn_predictions: Optional[Dict[str, float]] = None, priority: int = 0) -> int:
        """
        Predict the route index (edge position) where the emergency vehicle will be at target_time_seconds.
        
        Args:
            route_edges: Full route edge list
            target_time_seconds: Lookahead time in seconds (e.g., 30, 60, 90)
            pinn_predictions: Predicted congestion per edge
            priority: Emergency priority (higher = faster travel)
        
        Returns:
            Index into route_edges where vehicle is predicted to be at target_time_seconds.
            Clamped to [0, len(route_edges) - 1].
        """
        route_edges = list(route_edges or [])
        if not route_edges:
            return 0
        
        pinn_predictions = pinn_predictions or {}
        elapsed = 0.0
        edge_index = 0
        
        for idx, edge_id in enumerate(route_edges):
            edge_len = float(self.edge_info.get(edge_id, {}).get('length', 200.0))
            pred_congestion = float(pinn_predictions.get(edge_id, 0.0))
            speed_estimate = max(2.0, 12.0 - 1.5 * pred_congestion - 0.5 * min(priority, 3))
            edge_time = edge_len / speed_estimate
            
            if elapsed + edge_time >= target_time_seconds:
                edge_index = idx
                break
            elapsed += edge_time
            edge_index = idx + 1
        
        return min(edge_index, len(route_edges) - 1)

    def predict(self, route_edges: Iterable[str], live_metrics: Optional[Dict[str, Dict[str, float]]] = None, pinn_predictions: Optional[Dict[str, float]] = None, priority: int = 0) -> EmergencyETAResult:
        route_edges = list(route_edges or [])
        live_metrics = live_metrics or {}
        pinn_predictions = pinn_predictions or {}

        if not route_edges:
            return EmergencyETAResult(0.0, 0.0, 0.0, 0.0, 1.0)

        total_cost = 0.0
        predicted_congestion = []
        route_length = 0.0

        for edge_id in route_edges:
            metrics = live_metrics.get(edge_id, {})
            pred = float(pinn_predictions.get(edge_id, 0.0))
            predicted_congestion.append(pred)
            total_cost += weighted_edge_cost(edge_id, edge_info=self.edge_info.get(edge_id, {}), live_metrics=metrics, pinn_pred=pred)
            route_length += float(self.edge_info.get(edge_id, {}).get('length', 200.0))

        avg_pred = sum(predicted_congestion) / max(1, len(predicted_congestion))
        speed_estimate = max(2.0, 12.0 - 1.5 * avg_pred - 0.5 * min(priority, 3))
        eta_seconds = route_length / speed_estimate
        delay_probability = min(1.0, 0.15 + avg_pred * 0.7 + total_cost / max(1000.0, route_length * 8.0))
        route_risk = min(1.0, avg_pred * 0.8 + delay_probability * 0.5)
        congestion_impact = min(1.0, avg_pred)
        corridor_efficiency = max(0.0, 1.0 - route_risk * 0.8)

        return EmergencyETAResult(
            eta_seconds=float(eta_seconds),
            delay_probability=float(delay_probability),
            route_risk=float(route_risk),
            congestion_impact=float(congestion_impact),
            corridor_efficiency=float(corridor_efficiency),
        )

