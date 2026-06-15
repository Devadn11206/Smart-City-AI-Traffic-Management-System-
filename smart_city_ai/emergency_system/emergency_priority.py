from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmergencyPriorityProfile:
    vehicle_type: str
    priority: int
    label: str


class EmergencyPriorityManager:
    PRIORITY_MAP = {
        'ambulance': 3,
        'ambulance_vehicle': 3,
        'emergency': 3,
        'firetruck': 2,
        'fire_truck': 2,
        'fire': 2,
        'police': 1,
        'police_car': 1,
    }

    LABELS = {
        3: 'critical',
        2: 'high',
        1: 'medium',
        0: 'normal',
    }

    @classmethod
    def normalize_vehicle_type(cls, vehicle_type: str) -> str:
        return (vehicle_type or '').strip().lower().replace('-', '_').replace(' ', '_')

    @classmethod
    def priority_for_vehicle_type(cls, vehicle_type: str) -> int:
        normalized = cls.normalize_vehicle_type(vehicle_type)
        for key, value in cls.PRIORITY_MAP.items():
            if key in normalized:
                return value
        return 0

    @classmethod
    def profile_for_vehicle_type(cls, vehicle_type: str) -> EmergencyPriorityProfile:
        priority = cls.priority_for_vehicle_type(vehicle_type)
        return EmergencyPriorityProfile(
            vehicle_type=vehicle_type,
            priority=priority,
            label=cls.LABELS.get(priority, 'normal'),
        )

    @classmethod
    def is_emergency_vehicle(cls, vehicle_type: str) -> bool:
        return cls.priority_for_vehicle_type(vehicle_type) > 0

    @classmethod
    def priority_for_vehicle(cls, vehicle_id: str, vehicle_type: str) -> int:
        # Vehicle id may carry a semantic label even when SUMO type ids are generic.
        priority = cls.priority_for_vehicle_type(vehicle_type)
        if priority > 0:
            return priority
        return cls.priority_for_vehicle_type(vehicle_id)

    @classmethod
    def priority_multiplier(cls, priority: int) -> float:
        return {3: 2.2, 2: 1.6, 1: 1.2}.get(int(priority), 1.0)

