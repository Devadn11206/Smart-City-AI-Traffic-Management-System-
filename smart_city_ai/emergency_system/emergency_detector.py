from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from .emergency_priority import EmergencyPriorityManager

try:
    import traci
except Exception:  # pragma: no cover - SUMO may be unavailable during import
    traci = None


@dataclass
class EmergencyVehicleObservation:
    vehicle_id: str
    vehicle_type: str
    priority: int
    road_id: str
    lane_id: str
    route: List[str]
    route_index: int
    speed: float
    position: tuple
    destination: str
    mission_active: bool
    last_seen_step: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class EmergencyVehicleDetector:
    def __init__(self, vehicle_types: Optional[List[str]] = None, priority_manager: Optional[EmergencyPriorityManager] = None):
        self.vehicle_types = [EmergencyPriorityManager.normalize_vehicle_type(v) for v in (vehicle_types or [
            'ambulance', 'fire_truck', 'firetruck', 'police', 'police_car', 'emergency'
        ])]
        self.priority_manager = priority_manager or EmergencyPriorityManager()
        self.active_vehicles: Dict[str, EmergencyVehicleObservation] = {}

    def is_emergency_type(self, vehicle_type: str) -> bool:
        normalized = EmergencyPriorityManager.normalize_vehicle_type(vehicle_type)
        return any(token in normalized for token in self.vehicle_types) or self.priority_manager.is_emergency_vehicle(normalized)

    def scan(self, sim_step: Optional[int] = None) -> Dict[str, EmergencyVehicleObservation]:
        if traci is None:
            return {}

        sim_step = int(sim_step or 0)
        current_ids = set()

        try:
            vehicle_ids = list(traci.vehicle.getIDList())
        except Exception:
            return dict(self.active_vehicles)

        for vehicle_id in vehicle_ids:
            try:
                vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            except Exception:
                vehicle_type = vehicle_id

            if not self.is_emergency_type(vehicle_type) and not self.is_emergency_type(vehicle_id):
                continue

            try:
                route = list(traci.vehicle.getRoute(vehicle_id))
            except Exception:
                route = []

            try:
                road_id = traci.vehicle.getRoadID(vehicle_id)
            except Exception:
                road_id = route[0] if route else ''

            try:
                lane_id = traci.vehicle.getLaneID(vehicle_id)
            except Exception:
                lane_id = ''

            try:
                route_index = int(traci.vehicle.getRouteIndex(vehicle_id))
            except Exception:
                route_index = 0

            try:
                speed = float(traci.vehicle.getSpeed(vehicle_id))
            except Exception:
                speed = 0.0

            try:
                position = traci.vehicle.getPosition(vehicle_id)
            except Exception:
                position = (0.0, 0.0)

            destination = route[-1] if route else road_id
            priority = self.priority_manager.priority_for_vehicle(vehicle_id, vehicle_type)
            obs = EmergencyVehicleObservation(
                vehicle_id=vehicle_id,
                vehicle_type=vehicle_type,
                priority=priority,
                road_id=road_id,
                lane_id=lane_id,
                route=route,
                route_index=route_index,
                speed=speed,
                position=position,
                destination=destination,
                mission_active=True,
                last_seen_step=sim_step,
            )
            self.active_vehicles[vehicle_id] = obs
            current_ids.add(vehicle_id)

        # keep stale observations only for a brief period in case of TraCI gaps
        for vehicle_id in list(self.active_vehicles.keys()):
            if vehicle_id not in current_ids and sim_step - self.active_vehicles[vehicle_id].last_seen_step > 20:
                self.active_vehicles.pop(vehicle_id, None)

        return dict(self.active_vehicles)

    def active_vehicle_ids(self) -> List[str]:
        return list(self.active_vehicles.keys())

    def get(self, vehicle_id: str) -> Optional[EmergencyVehicleObservation]:
        return self.active_vehicles.get(vehicle_id)

    def upcoming_route_edges(self, vehicle_id: str, lookahead: int = 8) -> List[str]:
        observation = self.active_vehicles.get(vehicle_id)
        if not observation or not observation.route:
            return []
        idx = max(0, observation.route_index)
        return observation.route[idx: idx + max(1, int(lookahead))]

