from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .emergency_detector import EmergencyVehicleObservation
from .emergency_priority import EmergencyPriorityManager


@dataclass
class EmergencyScheduleItem:
    vehicle_id: str
    priority: int
    route: List[str]
    corridor_tls: List[str]
    eta_seconds: float
    conflict_score: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class MultiEmergencyManager:
    def __init__(self, priority_manager: Optional[EmergencyPriorityManager] = None):
        self.priority_manager = priority_manager or EmergencyPriorityManager()
        self.active_schedule: Dict[str, EmergencyScheduleItem] = {}

    def update(self, detections: Dict[str, EmergencyVehicleObservation], routes: Dict[str, List[str]], corridor_tls: Dict[str, List[str]], eta_by_vehicle: Optional[Dict[str, float]] = None) -> List[EmergencyScheduleItem]:
        eta_by_vehicle = eta_by_vehicle or {}
        items: List[EmergencyScheduleItem] = []
        for vehicle_id, obs in detections.items():
            item = EmergencyScheduleItem(
                vehicle_id=vehicle_id,
                priority=int(obs.priority),
                route=list(routes.get(vehicle_id, obs.route)),
                corridor_tls=list(corridor_tls.get(vehicle_id, [])),
                eta_seconds=float(eta_by_vehicle.get(vehicle_id, 0.0)),
            )
            items.append(item)

        items.sort(key=lambda entry: (-entry.priority, entry.eta_seconds, entry.vehicle_id))
        self.active_schedule = {item.vehicle_id: item for item in items}
        self._resolve_conflicts()
        return list(self.active_schedule.values())

    def _resolve_conflicts(self) -> None:
        occupied_tls: Dict[str, str] = {}
        for vehicle_id, item in self.active_schedule.items():
            conflict = 0.0
            for tls_id in item.corridor_tls:
                if tls_id in occupied_tls and occupied_tls[tls_id] != vehicle_id:
                    conflict += 1.0
            item.conflict_score = conflict
            if conflict > 0:
                # keep the highest priority corridor; lower priorities retain route planning but do not pre-empt.
                continue
            for tls_id in item.corridor_tls:
                occupied_tls[tls_id] = vehicle_id

    def best_vehicle(self) -> Optional[str]:
        if not self.active_schedule:
            return None
        return next(iter(self.active_schedule.keys()))

    def active_vehicles(self) -> List[str]:
        return list(self.active_schedule.keys())

    def corridor_tls(self) -> List[str]:
        tls = []
        for item in self.active_schedule.values():
            tls.extend(item.corridor_tls)
        return list(dict.fromkeys(tls))

