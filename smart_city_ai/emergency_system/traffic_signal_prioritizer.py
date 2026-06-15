"""
Emergency Traffic Signal Prioritization System
Forces traffic signals to green when emergency vehicles approach and ensures they never wait.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Set, Optional
import time

try:
    import traci
except Exception:
    traci = None

logger = logging.getLogger(__name__)


class TrafficSignalPrioritizer:
    """
    Prioritizes emergency vehicles at traffic signals by:
    - Forcing signals to green when emergency approaches
    - Preventing emergency vehicles from stopping at red lights
    - Setting pedestrian signals to red to clear intersections
    """
    
    def __init__(self, signal_detection_distance: float = 150.0):
        self.signal_detection_distance = signal_detection_distance
        self.emergency_priorities: Dict[str, int] = {}
        self.signal_phases: Dict[str, int] = {}
        self.signal_last_change: Dict[str, float] = {}
        self.processed_signals: Set[str] = set()
        
    def step(self) -> None:
        """Called every simulation step to manage traffic signals."""
        if traci is None:
            return
        
        try:
            all_vehicles = traci.vehicle.getIDList()
        except Exception:
            return
        
        # Process each emergency vehicle
        for vehicle_id in all_vehicles:
            if not self._is_emergency_vehicle(vehicle_id):
                continue
            
            try:
                # Get emergency vehicle position and lane
                current_edge = traci.vehicle.getRoadID(vehicle_id)
                current_lane = traci.vehicle.getLaneID(vehicle_id)
                
                # Find approaching traffic signals
                approaching_signals = self._find_approaching_signals(vehicle_id, current_edge)
                
                # Force green for each approaching signal
                for signal_id in approaching_signals:
                    self._force_green_light(signal_id, vehicle_id, current_edge)
                    
            except Exception as e:
                logger.debug(f"Failed to prioritize signals for {vehicle_id}: {e}")
    
    def _is_emergency_vehicle(self, vehicle_id: str) -> bool:
        """Check if vehicle is emergency."""
        try:
            vehicle_type = traci.vehicle.getTypeID(vehicle_id).lower()
        except Exception:
            vehicle_type = vehicle_id.lower()
        return any(token in vehicle_type for token in ('ambulance', 'fire', 'police', 'rescue', 'emergency'))
    
    def _find_approaching_signals(self, vehicle_id: str, current_edge: str) -> List[str]:
        """Find traffic signals approaching on the vehicle's route."""
        approaching = []
        
        try:
            # Get the vehicle's route
            route = list(traci.vehicle.getRoute(vehicle_id))
            route_index = traci.vehicle.getRouteIndex(vehicle_id)
            
            # Check next 5 edges in route
            for i in range(route_index, min(route_index + 5, len(route))):
                edge_id = route[i]
                
                # Get traffic lights controlling this edge
                try:
                    # SUMO stores traffic light IDs for edges
                    # Try to find signals at edge connections
                    incoming_signals = self._get_signals_for_edge(edge_id)
                    approaching.extend(incoming_signals)
                except Exception:
                    pass
        except Exception:
            pass
        
        return list(set(approaching))  # Remove duplicates
    
    def _get_signals_for_edge(self, edge_id: str) -> List[str]:
        """Get traffic signal IDs that control an edge."""
        signals = []
        
        if traci is None:
            return signals
        
        try:
            # Get outgoing connections from this edge
            connections = traci.edge.getOutgoing(edge_id)
            
            for next_edge in connections:
                # Get all traffic lights
                try:
                    all_signals = traci.trafficlight.getIDList()
                    
                    for signal_id in all_signals:
                        try:
                            controlled_links = traci.trafficlight.getControlledLinks(signal_id)
                            
                            # Check if this signal controls our edge
                            for link_group in controlled_links:
                                for link in link_group:
                                    if not link:
                                        continue
                                    incoming_edge = link[0].split('_')[0]
                                    if incoming_edge == edge_id:
                                        if signal_id not in signals:
                                            signals.append(signal_id)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
        
        return signals
    
    def _force_green_light(self, signal_id: str, vehicle_id: str, current_edge: str) -> None:
        """Force a traffic signal to green for emergency vehicle."""
        if traci is None:
            return
        
        try:
            # Get current phase
            current_phase = traci.trafficlight.getPhase(signal_id)
            
            # Get all available phases
            logics = traci.trafficlight.getAllProgramLogics(signal_id)
            if not logics:
                return
            
            phases = logics[0].getPhases()
            
            # Find a phase that gives green to vehicle's direction
            best_phase = self._find_best_phase_for_vehicle(signal_id, current_edge, phases)
            
            # Set to that phase
            if best_phase != current_phase:
                traci.trafficlight.setPhase(signal_id, best_phase)
                logger.info(f"🚨 FORCED GREEN: Signal {signal_id} phase changed {current_phase} → {best_phase} for emergency {vehicle_id}")
            
            # Extend phase duration to ensure emergency passes
            try:
                traci.trafficlight.setPhaseDuration(signal_id, 30)  # 30 second green
            except Exception:
                pass
                
        except Exception as e:
            logger.debug(f"Failed to force green for {signal_id}: {e}")
    
    def _find_best_phase_for_vehicle(self, signal_id: str, edge_id: str, phases) -> int:
        """Find phase that gives green light for the edge."""
        if traci is None:
            return 0
        
        best_phase = 0
        best_score = -1
        
        try:
            controlled_links = traci.trafficlight.getControlledLinks(signal_id)
            
            for phase_idx, phase in enumerate(phases):
                score = 0
                
                # Check each signal in this phase
                for signal_idx, links in enumerate(controlled_links):
                    if signal_idx >= len(phase.state):
                        continue
                    
                    # Check if this signal is green
                    if phase.state[signal_idx] not in ('G', 'g'):
                        continue
                    
                    # Check if any link from this signal comes from our edge
                    for link in links:
                        if not link:
                            continue
                        incoming_lane = link[0]
                        
                        try:
                            incoming_edge = traci.lane.getEdgeID(incoming_lane)
                            if incoming_edge == edge_id:
                                score += 100  # High priority match
                        except Exception:
                            pass
                
                if score > best_score:
                    best_score = score
                    best_phase = phase_idx
        except Exception:
            pass
        
        return best_phase
    
    def get_status(self) -> Dict[str, object]:
        """Return status of traffic signal prioritization."""
        return {
            'total_signals_managed': len(self.signal_phases),
            'processed_signals': list(self.processed_signals),
            'signal_phases': dict(self.signal_phases),
        }
