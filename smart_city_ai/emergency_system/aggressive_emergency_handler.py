"""
Aggressive Emergency Vehicle Handler - Ensures emergency vehicles never stop, change lanes aggressively,
and forces other vehicles to cooperate and yield immediately.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Set, Tuple
import time

try:
    import traci
except Exception:
    traci = None

from .traffic_signal_prioritizer import TrafficSignalPrioritizer

logger = logging.getLogger(__name__)

# Lane change modes for SUMO:
# 0 = no lane changing
# 511 = maximum permissive (change left/right/cooperate)
# 256 = cooperate with others
# 512 = strategic lane changing

class AggressiveEmergencyHandler:
    """Aggressive emergency vehicle handler that ensures zero stops and optimal routing."""
    
    def __init__(self):
        self.emergency_vehicles: Set[str] = set()
        self.min_emergency_speed = 25.0  # m/s - maintain minimum speed for emergency vehicles
        self.max_normal_speed = 13.89  # m/s - standard speed limit
        self.clearing_distance = 100.0  # meters ahead to clear
        self.lane_change_frequency = 1  # change lanes EVERY step (most aggressive)
        self.last_lane_change: Dict[str, int] = {}
        self.vehicle_cooperation_cache: Dict[str, bool] = {}
        self.step_counter = 0
        self.signal_prioritizer = TrafficSignalPrioritizer(signal_detection_distance=200.0)
        
    def step(self) -> None:
        """Called every simulation step to manage emergency vehicles."""
        if traci is None:
            return
        
        self.step_counter += 1
        self._update_emergency_vehicles()
        self.signal_prioritizer.step()  # NEW: Force green signals for emergency vehicles
        self._enforce_emergency_speeds()
        self._enforce_lane_changes()
        self._clear_ahead_of_emergency()
        self._force_vehicle_cooperation()
        
    def _update_emergency_vehicles(self) -> None:
        """Identify and register emergency vehicles."""
        try:
            all_vehicles = traci.vehicle.getIDList()
        except Exception:
            return
            
        for vehicle_id in all_vehicles:
            if self._is_emergency_vehicle(vehicle_id):
                self.emergency_vehicles.add(vehicle_id)
            elif vehicle_id in self.emergency_vehicles:
                # Vehicle is no longer emergency (reached destination)
                self.emergency_vehicles.discard(vehicle_id)
                self.last_lane_change.pop(vehicle_id, None)
    
    def _is_emergency_vehicle(self, vehicle_id: str) -> bool:
        """Check if vehicle is an emergency vehicle."""
        try:
            vehicle_type = traci.vehicle.getTypeID(vehicle_id).lower()
        except Exception:
            vehicle_type = vehicle_id.lower()
        return any(token in vehicle_type for token in ('ambulance', 'fire', 'police', 'rescue', 'emergency'))
    
    def _enforce_emergency_speeds(self) -> None:
        """Ensure emergency vehicles maintain minimum speed - NEVER stop."""
        for vehicle_id in self.emergency_vehicles:
            try:
                current_speed = traci.vehicle.getSpeed(vehicle_id)
                max_speed = traci.vehicle.getMaxSpeed(vehicle_id)
                
                # Determine target speed
                target_speed = max(self.min_emergency_speed, max_speed)
                
                # If emergency is below minimum, enforce it
                if current_speed < self.min_emergency_speed:
                    # Instead of setting absolute speed, reduce deceleration
                    try:
                        # Set a high desired speed to overcome congestion
                        traci.vehicle.setSpeed(vehicle_id, target_speed)
                        logger.info(f"🚨 EMERGENCY SPEED BOOST: {vehicle_id} speed set to {target_speed:.1f} m/s (was {current_speed:.1f})")
                    except Exception:
                        pass
                
                # Disable speed reduction (decel) for emergency vehicles
                try:
                    traci.vehicle.setAccel(vehicle_id, 2.6)  # Maximum acceleration
                    traci.vehicle.setDecel(vehicle_id, 2.0)  # Maintain deceleration capability
                except Exception:
                    pass
                    
            except Exception as e:
                logger.debug(f"Failed to enforce speed for {vehicle_id}: {e}")
    
    def _enforce_lane_changes(self) -> None:
        """Force emergency vehicles to perform aggressive lane changes EVERY STEP."""
        for vehicle_id in self.emergency_vehicles:
            try:
                # Set maximum permissive lane changing mode
                try:
                    # Mode 511 = all permissions (left/right/cooperative/strategic)
                    traci.vehicle.setLaneChangeMode(vehicle_id, 511)
                except Exception:
                    pass
                
                # Get current position
                road_id = traci.vehicle.getRoadID(vehicle_id)
                lane_id = traci.vehicle.getLaneID(vehicle_id)
                current_lane = int(lane_id.split('_')[-1]) if '_' in lane_id else 0
                
                # Get available lanes on this road
                try:
                    num_lanes = traci.edge.getLaneNumber(road_id)
                    
                    # Determine best lane for passing (rightmost/leftmost based on preference)
                    if num_lanes > 1:
                        # Try to alternate lanes to maximize passing
                        target_lane = (current_lane + 1) % num_lanes
                        
                        # Force change to target lane with short duration
                        try:
                            traci.vehicle.changeLane(vehicle_id, target_lane, duration=1)
                            logger.debug(f"🚨 AGGRESSIVE LANE CHANGE: {vehicle_id} changing to lane {target_lane}")
                        except Exception:
                            # If can't change, try the other direction
                            try:
                                target_lane = (current_lane - 1) % num_lanes
                                traci.vehicle.changeLane(vehicle_id, target_lane, duration=1)
                            except Exception:
                                pass
                except Exception:
                    pass
                    
            except Exception as e:
                logger.debug(f"Failed to enforce lane change for {vehicle_id}: {e}")
    
    def _clear_ahead_of_emergency(self) -> None:
        """Clear all vehicles ahead of emergency vehicles."""
        for vehicle_id in self.emergency_vehicles:
            try:
                current_edge = traci.vehicle.getRoadID(vehicle_id)
                current_lane = traci.vehicle.getLaneID(vehicle_id)
                vehicle_pos = traci.vehicle.getPosition(vehicle_id)
                
                # Get vehicles on same edge and lane
                try:
                    vehicles_on_lane = traci.lane.getLastStepVehicleIDs(current_lane)
                except Exception:
                    continue
                
                # For each vehicle ahead, apply clearing strategy
                for other_vehicle in vehicles_on_lane:
                    if other_vehicle == vehicle_id or self._is_emergency_vehicle(other_vehicle):
                        continue
                    
                    try:
                        other_pos = traci.vehicle.getPosition(other_vehicle)
                        other_speed = traci.vehicle.getSpeed(other_vehicle)
                        
                        # Calculate distance (simple 2D distance)
                        distance = ((vehicle_pos[0] - other_pos[0])**2 + (vehicle_pos[1] - other_pos[1])**2)**0.5
                        
                        # If vehicle is ahead within clearing distance, make it yield
                        if 0 < distance < self.clearing_distance:
                            self._make_vehicle_yield(other_vehicle, urgency="IMMEDIATE")
                    except Exception:
                        pass
                        
            except Exception as e:
                logger.debug(f"Failed to clear ahead of {vehicle_id}: {e}")
    
    def _force_vehicle_cooperation(self) -> None:
        """Force normal vehicles to cooperate with emergency vehicles."""
        try:
            all_vehicles = traci.vehicle.getIDList()
        except Exception:
            return
        
        for vehicle_id in all_vehicles:
            if self._is_emergency_vehicle(vehicle_id):
                continue
            
            try:
                # Set cooperative lane changing mode for all non-emergency vehicles
                try:
                    # Mode 256 = cooperative (help others pass)
                    traci.vehicle.setLaneChangeMode(vehicle_id, 256)
                except Exception:
                    pass
                
                # Check if this vehicle is blocking any emergency vehicle
                vehicle_edge = traci.vehicle.getRoadID(vehicle_id)
                
                # If on same edge as any emergency, prioritize yielding
                for emergency_id in self.emergency_vehicles:
                    try:
                        emergency_edge = traci.vehicle.getRoadID(emergency_id)
                        if emergency_edge == vehicle_edge:
                            self._make_vehicle_yield(vehicle_id, urgency="HIGH")
                            break
                    except Exception:
                        pass
                        
            except Exception as e:
                logger.debug(f"Failed to enforce cooperation for {vehicle_id}: {e}")
    
    def _make_vehicle_yield(self, vehicle_id: str, urgency: str = "NORMAL") -> None:
        """Make a normal vehicle yield to emergency vehicles."""
        if traci is None or self._is_emergency_vehicle(vehicle_id):
            return
        
        try:
            current_speed = traci.vehicle.getSpeed(vehicle_id)
            
            if urgency == "IMMEDIATE":
                # Hard brake but don't stop - reduce to 0.5 m/s minimum
                target_speed = max(0.5, current_speed * 0.2)
                action = "HARD YIELD"
            elif urgency == "HIGH":
                # Significant slowdown - 50% reduction
                target_speed = max(1.0, current_speed * 0.5)
                action = "URGENT YIELD"
            else:
                # Gradual slowdown - 20% reduction
                target_speed = max(1.5, current_speed * 0.8)
                action = "NORMAL YIELD"
            
            traci.vehicle.setSpeed(vehicle_id, target_speed)
            logger.info(f"🚗 {action}: {vehicle_id} reduced speed from {current_speed:.1f} → {target_speed:.1f} m/s")
            
            # Set to cooperative mode to allow safe lane changes
            try:
                traci.vehicle.setLaneChangeMode(vehicle_id, 256)
            except Exception:
                pass
                
        except Exception as e:
            logger.debug(f"Failed to make {vehicle_id} yield: {e}")
    
    def get_emergency_status(self) -> Dict[str, object]:
        """Return status of all emergency vehicles."""
        status = {
            'total_emergency_vehicles': len(self.emergency_vehicles),
            'vehicle_ids': list(self.emergency_vehicles),
            'statistics': {}
        }
        
        for vehicle_id in self.emergency_vehicles:
            try:
                status['statistics'][vehicle_id] = {
                    'speed': float(traci.vehicle.getSpeed(vehicle_id)),
                    'edge': traci.vehicle.getRoadID(vehicle_id),
                    'lane': traci.vehicle.getLaneID(vehicle_id),
                }
            except Exception:
                pass
        
        return status
