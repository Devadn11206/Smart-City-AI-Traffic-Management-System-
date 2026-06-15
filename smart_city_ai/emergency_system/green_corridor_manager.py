from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import traci
except Exception:  # pragma: no cover
    traci = None
import logging
import os

logger = logging.getLogger(__name__)


def _is_traci_alive() -> bool:
    """Check if TraCI connection is still active."""
    if traci is None:
        return False
    try:
        # Try a lightweight query to check connection health
        return traci.isLoaded()
    except Exception:
        return False


@dataclass
class CorridorAllocation:
    vehicle_id: str
    tls_ids: List[str]
    route_edges: List[str]
    priority: int
    start_step: int
    release_step: int
    phase_plan: Dict[str, int]
    # Predictive corridor fields:
    future_tls_sequence: Optional[List[str]] = None  # Ordered TLS ahead of vehicle
    current_tls_index: int = 0  # Current position in future_tls_sequence
    prediction_window_seconds: int = 30  # Lookahead for predictive activation

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def select_best_phase_for_tls(ts_id: str, route_edges: Iterable[str]) -> int:
    if traci is None:
        return 0
    route_edges = set(route_edges or [])
    try:
        logics = traci.trafficlight.getAllProgramLogics(ts_id)
        if not logics:
            return 0
        phases = logics[0].getPhases()
        controlled_links = traci.trafficlight.getControlledLinks(ts_id)
        best_phase = 0
        best_score = float('-inf')
        for phase_idx, phase in enumerate(phases):
            score = 0.0
            for signal_idx, links in enumerate(controlled_links):
                if signal_idx >= len(phase.state):
                    continue
                if phase.state[signal_idx] not in ('G', 'g'):
                    continue
                for link in links:
                    if not link:
                        continue
                    incoming_lane = link[0]
                    try:
                        incoming_edge = traci.lane.getEdgeID(incoming_lane)
                    except Exception:
                        incoming_edge = None
                    if incoming_edge in route_edges:
                        score += 2.0
                    else:
                        score += 0.2
            if score > best_score:
                best_score = score
                best_phase = phase_idx
        return best_phase
    except Exception:
        return 0


def _safe_phase_index(ts_id: str, desired_phase: int) -> int:
    if traci is None:
        return max(0, int(desired_phase))
    try:
        logics = traci.trafficlight.getAllProgramLogics(ts_id)
        if logics and len(logics) > 0:
            phase_count = max(1, len(logics[0].getPhases()))
            return max(0, min(int(desired_phase), phase_count - 1))
    except Exception:
        logger.exception('Failed to compute safe phase index for %s', ts_id)
    return max(0, int(desired_phase))


def _has_traci_gui() -> bool:
    return traci is not None and hasattr(traci, 'gui') and callable(getattr(traci.gui, 'getIDList', None))


class GreenCorridorManager:
    def __init__(self, use_gui: bool = False):
        self.active_corridors: Dict[str, CorridorAllocation] = {}
        self.phase_cache: Dict[str, int] = {}
        self._tls_controlled_edges: Dict[str, set] = {}
        self.use_gui = bool(use_gui)
        self._screenshot_warning_emitted = False

        # Metrics tracking
        self.metrics_activation_count = 0
        self.metrics_reserved_edge_count = 0
        self.metrics_success_count = 0
        self.metrics_completed_corridors = 0

    def _build_tls_edge_cache(self) -> None:
        if traci is None:
            return
        try:
            for ts_id in traci.trafficlight.getIDList():
                lanes = traci.trafficlight.getControlledLanes(ts_id)
                edges = set()
                for lane_id in lanes:
                    try:
                        edges.add(traci.lane.getEdgeID(lane_id))
                    except Exception:
                        continue
                self._tls_controlled_edges[ts_id] = edges
        except Exception:
            logger.exception('Failed to cache TLS controlled edges')

    def _route_tls_candidates(self, route_edges: Iterable[str]) -> List[str]:
        if traci is None or not _is_traci_alive():
            return []
        if not self._tls_controlled_edges:
            self._build_tls_edge_cache()
        route_edges = set(route_edges or [])
        tls_ids: List[str] = []
        try:
            for ts_id in traci.trafficlight.getIDList():
                lanes = traci.trafficlight.getControlledLanes(ts_id)
                edges = set()
                for lane_id in lanes:
                    try:
                        edges.add(traci.lane.getEdgeID(lane_id))
                    except Exception:
                        logger.exception('Failed to get edge for lane %s on TLS %s', lane_id, ts_id)
                        continue
                if edges.intersection(route_edges):
                    tls_ids.append(ts_id)
        except Exception:
            logger.exception('Failed while enumerating traffic lights for route TLS candidates')
        return tls_ids

    def activate(self, vehicle_id: str, route_edges: Iterable[str], sim_step: int, hold_steps: int = 20, priority: int = 0) -> CorridorAllocation:
        route_edges = list(dict.fromkeys(route_edges))
        tls_ids = self._route_tls_candidates(route_edges)
        phase_plan: Dict[str, int] = {}
        if traci is not None:
            for ts_id in tls_ids:
                try:
                    if ts_id not in self.phase_cache:
                        self.phase_cache[ts_id] = int(traci.trafficlight.getPhase(ts_id))
                    phase_plan[ts_id] = _safe_phase_index(ts_id, select_best_phase_for_tls(ts_id, route_edges))
                    traci.trafficlight.setPhase(ts_id, phase_plan[ts_id])
                except Exception:
                    logger.exception('Failed during activation phase set for TLS %s', ts_id)
                    continue

        allocation = CorridorAllocation(
            vehicle_id=vehicle_id,
            tls_ids=tls_ids,
            route_edges=route_edges,
            priority=int(priority),
            start_step=int(sim_step),
            release_step=int(sim_step) + max(1, int(hold_steps)),
            phase_plan=phase_plan,
            future_tls_sequence=None,
            current_tls_index=0,
        )
        if vehicle_id not in self.active_corridors:
            self.metrics_activation_count += 1
        self.active_corridors[vehicle_id] = allocation
        self.metrics_reserved_edge_count += len(tls_ids)
        logger.info("Activated green corridor for %s tls=%s hold_steps=%s priority=%s", vehicle_id, tls_ids, hold_steps, priority)
        # capture screenshot for visual confirmation
        try:
            if self.use_gui and _has_traci_gui():
                views = traci.gui.getIDList()
                if views:
                    out_dir = os.path.join('results', 'emergency_metrics', 'screenshots')
                    os.makedirs(out_dir, exist_ok=True)
                    filename = os.path.join(out_dir, f"corridor_{int(sim_step)}_{vehicle_id}.png")
                    try:
                        traci.gui.screenshot(views[0], filename)
                    except Exception:
                        pass
            elif not self.use_gui and not self._screenshot_warning_emitted:
                logger.warning('GUI disabled: skipping green corridor screenshot capture')
                self._screenshot_warning_emitted = True
        except Exception:
            pass
        return allocation

    def activate_predictive(self, vehicle_id: str, route_edges: Iterable[str], sim_step: int, hold_steps: int = 20, priority: int = 0, future_tls_sequence: Optional[List[str]] = None) -> CorridorAllocation:
        """
        Activate a predictive green corridor with a sequence of future TLS to be pre-activated.
        
        Args:
            vehicle_id: Emergency vehicle ID
            route_edges: Full route edges
            sim_step: Current simulation step
            hold_steps: Duration to hold corridor
            priority: Emergency priority level
            future_tls_sequence: Ordered list of TLS IDs from current position to destination.
                               If None, uses full route TLS.
        
        Returns:
            CorridorAllocation with predictive corridor setup.
        """
        route_edges = list(dict.fromkeys(route_edges))
        
        # Use provided sequence or derive from route
        if future_tls_sequence is None:
            future_tls_sequence = self._route_tls_candidates(route_edges)
        else:
            future_tls_sequence = list(dict.fromkeys(future_tls_sequence))
        
        phase_plan: Dict[str, int] = {}
        if traci is not None:
            for ts_id in future_tls_sequence:
                try:
                    if ts_id not in self.phase_cache:
                        self.phase_cache[ts_id] = int(traci.trafficlight.getPhase(ts_id))
                    # Select phase that aligns with route edges
                    phase_plan[ts_id] = _safe_phase_index(ts_id, select_best_phase_for_tls(ts_id, route_edges))
                    traci.trafficlight.setPhase(ts_id, phase_plan[ts_id])
                except Exception:
                    logger.exception('Failed during predictive phase set for TLS %s', ts_id)
                    continue
        
        allocation = CorridorAllocation(
            vehicle_id=vehicle_id,
            tls_ids=future_tls_sequence,
            route_edges=route_edges,
            priority=int(priority),
            start_step=int(sim_step),
            release_step=int(sim_step) + max(1, int(hold_steps)),
            phase_plan=phase_plan,
            future_tls_sequence=future_tls_sequence,
            current_tls_index=0,
            prediction_window_seconds=30,
        )
        if vehicle_id not in self.active_corridors:
            self.metrics_activation_count += 1
        self.active_corridors[vehicle_id] = allocation
        self.metrics_reserved_edge_count += len(future_tls_sequence)
        logger.info("Activated predictive green corridor for %s future_tls=%s hold_steps=%s priority=%s", vehicle_id, future_tls_sequence, hold_steps, priority)
        # capture screenshot for visual confirmation
        try:
            if self.use_gui and _has_traci_gui():
                views = traci.gui.getIDList()
                if views:
                    out_dir = os.path.join('results', 'emergency_metrics', 'screenshots')
                    os.makedirs(out_dir, exist_ok=True)
                    filename = os.path.join(out_dir, f"predictive_corridor_{int(sim_step)}_{vehicle_id}.png")
                    try:
                        traci.gui.screenshot(views[0], filename)
                    except Exception:
                        pass
        except Exception:
            pass
        return allocation

    def sync_citywide(self, vehicle_id: str, route_edges: Iterable[str], sim_step: int, hold_steps: int = 20, priority: int = 0) -> CorridorAllocation:
        return self.activate(vehicle_id, route_edges, sim_step=sim_step, hold_steps=hold_steps, priority=priority)

    def update(self, sim_step: int) -> None:
        if traci is None:
            return
        
        # Maintain active corridors: keep green phases for all active TLS
        for vehicle_id in list(self.active_corridors.keys()):
            allocation = self.active_corridors[vehicle_id]
            
            # If corridor still active, maintain the green phase
            if allocation.release_step > sim_step:
                for ts_id, phase in allocation.phase_plan.items():
                    try:
                        # Continuously maintain the phase during hold period
                        current_phase = traci.trafficlight.getPhase(ts_id)
                        if current_phase != phase:
                            traci.trafficlight.setPhase(ts_id, _safe_phase_index(ts_id, phase))
                            logger.debug(f"Holding phase={phase} for TLS {ts_id} (vehicle={vehicle_id}, step={sim_step}, release_at={allocation.release_step})")
                    except Exception:
                        logger.exception('Failed to maintain phase for TLS %s during corridor hold', ts_id)
            else:
                # Release: restore original phases
                for ts_id, phase in allocation.phase_plan.items():
                    try:
                        prev_phase = self.phase_cache.pop(ts_id, None)
                        if prev_phase is not None:
                            traci.trafficlight.setPhase(ts_id, _safe_phase_index(ts_id, prev_phase))
                        else:
                            traci.trafficlight.setPhase(ts_id, _safe_phase_index(ts_id, phase))
                    except Exception:
                        logger.exception('Failed to restore phase for TLS %s during release', ts_id)
                self.active_corridors.pop(vehicle_id, None)
                logger.info("Released green corridor for %s at step %s", vehicle_id, sim_step)

    def active_tls(self) -> List[str]:
        tls_ids = []
        for allocation in self.active_corridors.values():
            tls_ids.extend(allocation.tls_ids)
        return list(dict.fromkeys(tls_ids))

    def active_route_edges(self) -> List[str]:
        edges = []
        for allocation in self.active_corridors.values():
            edges.extend(allocation.route_edges)
        return list(dict.fromkeys(edges))

