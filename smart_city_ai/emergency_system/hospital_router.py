from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .emergency_eta_predictor import EmergencyETAPredictor


@dataclass
class HospitalRoutePlan:
    hospital_id: str
    route: List[str]
    eta_seconds: float
    route_risk: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class HospitalRouter:
    def __init__(self, hospitals: Optional[Sequence[str]] = None, edge_info: Optional[Dict[str, Dict[str, float]]] = None):
        self.hospitals = list(hospitals or [])
        self.eta_predictor = EmergencyETAPredictor(edge_info=edge_info)

    def set_hospitals(self, hospitals: Sequence[str]) -> None:
        self.hospitals = list(hospitals)

    def choose_best_hospital(self, vehicle_id: str, candidate_hospitals: Optional[Sequence[str]], emergency_router, pinn_predictions: Optional[Dict[str, float]] = None, priority: int = 0) -> HospitalRoutePlan:
        candidates = list(candidate_hospitals or self.hospitals)
        best_plan = HospitalRoutePlan(hospital_id='', route=[], eta_seconds=float('inf'), route_risk=1.0)
        for hospital_edge in candidates:
            route, _ = emergency_router.plan_route(vehicle_id, hospital_edge, predictions=pinn_predictions, priority=priority)
            if not route:
                continue
            eta = self.eta_predictor.predict(route, pinn_predictions=pinn_predictions, priority=priority)
            plan = HospitalRoutePlan(hospital_id=hospital_edge, route=list(route), eta_seconds=eta.eta_seconds, route_risk=eta.route_risk)
            if plan.eta_seconds < best_plan.eta_seconds:
                best_plan = plan
        return best_plan

