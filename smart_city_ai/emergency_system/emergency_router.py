from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import traci
except Exception:  # pragma: no cover
    traci = None

from aco_routing.congestion_costs import DEFAULT_COST_WEIGHTS, live_edge_metrics, weighted_edge_cost

logger = logging.getLogger(__name__)


class EmergencyRouter:
    def __init__(self, acorouter, adjacency: Dict[str, List[str]], edge_info: Optional[Dict[str, Dict[str, float]]] = None, accident_manager=None, traffic_clearing=None, priority_manager=None):
        self.acorouter = acorouter
        self.adjacency = adjacency or {}
        self.edge_info = edge_info or {}
        self.accident_manager = accident_manager
        self.traffic_clearing = traffic_clearing
        self.priority_manager = priority_manager
        self.active_reservations: Dict[str, List[str]] = {}
        self._live_metrics_cache: Dict[str, Dict[str, float]] = {}

    def _fallback_path(self, source: str, target: str, blocked_edges: Optional[Sequence[str]] = None) -> List[str]:
        blocked = set(blocked_edges or [])
        if source == target:
            return [source]
        queue = deque([[source]])
        visited = {source}
        while queue:
            path = queue.popleft()
            current = path[-1]
            for neighbor in self.adjacency.get(current, []):
                if neighbor in blocked and neighbor != target:
                    continue
                if neighbor in visited:
                    continue
                new_path = path + [neighbor]
                if neighbor == target:
                    return new_path
                visited.add(neighbor)
                queue.append(new_path)
        return [source, target] if source != target else [source]

    def _heuristic(self, predictions=None, priority: int = 0, avoid_edges: Optional[Sequence[str]] = None):
        predictions = predictions or {}
        avoid_edges = set(avoid_edges or [])

        def get_live_metrics(edge_id: str) -> Dict[str, float]:
            if edge_id in self._live_metrics_cache:
                return self._live_metrics_cache[edge_id]
            metrics = live_edge_metrics(edge_id)
            self._live_metrics_cache[edge_id] = metrics
            return metrics

        def score(edge_id: str) -> float:
            live = get_live_metrics(edge_id)
            pred = float(predictions.get(edge_id, 0.0))
            blocked_penalty = 0.0
            
            # AGGRESSIVE CONGESTION AVOIDANCE
            # Penalize edges with high vehicle density
            if live.get('vehicle_count', 0) > 5:
                blocked_penalty += 200.0 * (live.get('vehicle_count', 0) / 10.0)
            
            if edge_id in avoid_edges:
                blocked_penalty += 500.0
            if self.accident_manager is not None:
                blocked_penalty += self.accident_manager.avoidance_penalty(edge_id)
            
            weighted = weighted_edge_cost(
                edge_id,
                edge_info=self.edge_info.get(edge_id, {}),
                live_metrics=live,
                pinn_pred=pred,
                weights=DEFAULT_COST_WEIGHTS,
            )
            priority_bonus = max(0.1, 1.0 - 0.12 * priority)
            return max(0.1, weighted * priority_bonus + blocked_penalty)

        return score

    def plan_route(self, vehicle_id: str, destination_edge: str, predictions: Optional[Dict[str, float]] = None, priority: int = 0, avoid_edges: Optional[Sequence[str]] = None, ants: Optional[int] = None, max_steps: Optional[int] = None) -> Tuple[List[str], float]:
        if traci is None:
            return [], float('inf')

        try:
            source_edge = traci.vehicle.getRoadID(vehicle_id)
        except Exception:
            source_edge = ''

        self._live_metrics_cache.clear()
        heuristic_fn = self._heuristic(predictions=predictions, priority=priority, avoid_edges=avoid_edges)
        route, cost = self.acorouter.find_route(source_edge, destination_edge, self.adjacency, heuristic_fn=heuristic_fn, ants=ants, max_steps=max_steps)
        if not route or len(route) < 2:
            route = self._fallback_path(source_edge, destination_edge, blocked_edges=avoid_edges)
            cost = sum(heuristic_fn(edge_id) for edge_id in route[1:]) if len(route) > 1 else 0.0
        return route, float(cost)

    def apply_route(self, vehicle_id: str, route: Sequence[str]) -> bool:
        """Apply route to vehicle with smooth transition and retry logic."""
        if traci is None or not route or len(route) < 2:
            return False
        
        route_list = list(route)
        
        try:
            # Get current edge and ensure it's in the route
            current_edge = traci.vehicle.getRoadID(vehicle_id)
            if current_edge:
                if current_edge.startswith(':'):
                    return False
                if current_edge not in route_list:
                    route_list = [current_edge] + route_list
                elif route_list[0] != current_edge:
                    route_list = route_list[route_list.index(current_edge):]
            
            # First attempt: standard setRoute
            try:
                traci.vehicle.setRoute(vehicle_id, route_list)
                logger.debug(f"✅ Applied route to {vehicle_id}: {route_list[:3]}...→{route_list[-1]}")
                return True
            except Exception as ex1:
                logger.debug(f"Route attempt 1 failed for {vehicle_id}: {ex1}")
        
        except Exception:
            pass
        
        # Second attempt: Alternative method using changeTarget
        try:
            if len(route_list) > 0:
                # Try to set just the destination as fallback
                destination = route_list[-1]
                traci.vehicle.changeTarget(vehicle_id, destination)
                logger.info(f"✅ Fallback: Changed target for {vehicle_id} to {destination}")
                return True
        except Exception as ex2:
            logger.warning(f"Both route methods failed for {vehicle_id}: setRoute failed, changeTarget failed: {ex2}")
        
        return False

    def reroute_vehicle(self, vehicle_id: str, destination_edge: str, predictions: Optional[Dict[str, float]] = None, priority: int = 0, avoid_edges: Optional[Sequence[str]] = None, ants: Optional[int] = None, max_steps: Optional[int] = None) -> List[str]:
        route, _ = self.plan_route(vehicle_id, destination_edge, predictions=predictions, priority=priority, avoid_edges=avoid_edges, ants=ants, max_steps=max_steps)
        if route:
            self.apply_route(vehicle_id, route)
            if self.traffic_clearing is not None:
                self.traffic_clearing.reserve_route(vehicle_id, route)
            self.active_reservations[vehicle_id] = list(route)
        return list(route)

    def reserve_route(self, vehicle_id: str, route_edges: Sequence[str]) -> None:
        self.active_reservations[vehicle_id] = list(route_edges)
        if self.traffic_clearing is not None:
            self.traffic_clearing.reserve_route(vehicle_id, route_edges)

