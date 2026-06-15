from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class AccidentEvent:
    edge_id: str
    start_step: int
    end_step: int
    severity: float = 1.0
    reason: str = 'accident'
    lane_id: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class AccidentManager:
    def __init__(self):
        self.events: Dict[str, AccidentEvent] = {}

    def register_accident(self, edge_id: str, start_step: int, duration_steps: int = 120, severity: float = 1.0, reason: str = 'accident') -> AccidentEvent:
        event = AccidentEvent(
            edge_id=edge_id,
            start_step=int(start_step),
            end_step=int(start_step) + max(1, int(duration_steps)),
            severity=float(severity),
            reason=reason,
        )
        self.events[edge_id] = event
        return event

    def register_lane_closure(self, lane_id: str, start_step: int, duration_steps: int = 120, severity: float = 1.0, reason: str = 'lane_closure') -> AccidentEvent:
        edge_id = lane_id.split('_')[0] if '_' in lane_id else lane_id
        event = self.register_accident(edge_id, start_step, duration_steps, severity, reason)
        event.lane_id = lane_id
        self.events[edge_id] = event
        return event

    def step(self, sim_step: int) -> None:
        for edge_id in list(self.events.keys()):
            if self.events[edge_id].end_step <= sim_step:
                self.events.pop(edge_id, None)

    def is_blocked(self, edge_id: str, sim_step: Optional[int] = None) -> bool:
        event = self.events.get(edge_id)
        if event is None:
            return False
        if sim_step is None:
            return True
        return event.start_step <= sim_step < event.end_step

    def blocked_edges(self, sim_step: Optional[int] = None) -> List[str]:
        if sim_step is None:
            return list(self.events.keys())
        return [edge_id for edge_id, event in self.events.items() if event.start_step <= sim_step < event.end_step]

    def avoidance_penalty(self, edge_id: str, sim_step: Optional[int] = None) -> float:
        event = self.events.get(edge_id)
        if event is None:
            return 0.0
        if sim_step is not None and not (event.start_step <= sim_step < event.end_step):
            return 0.0
        return 50.0 * float(event.severity)

