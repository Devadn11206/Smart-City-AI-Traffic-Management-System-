from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque
import statistics
from pathlib import Path
from typing import Dict, List, Any

import traci

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMO_CFG = REPO_ROOT / 'sumo_network' / 'config.sumocfg'

logger = logging.getLogger('dash.traci')
logger.setLevel(logging.INFO)


class SumoController:
    """Simple SUMO controller that starts SUMO and streams state via TraCI.

    It runs SUMO in a background thread using TraCI and keeps an in-memory
    snapshot of vehicles, traffic-lights and basic metrics for dashboards.
    """

    def __init__(self, port: int = 8813, gui: bool = False):
        self.port = port
        self.gui = gui
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._conn_lock = threading.Lock()
        self._running = threading.Event()
        self._stopping = threading.Event()
        self._paused = threading.Event()
        self._snapshot_lock = threading.Lock()
        self._latency_ewma_ms = 0.0
        self._snapshot_latency_ewma_ms = 0.0
        # rolling buffers for profiling (keeps recent N samples)
        self._step_durations_ms: deque[float] = deque(maxlen=200)
        self._snapshot_durations_ms: deque[float] = deque(maxlen=200)
        self._loop_count = 0
        self._errors = 0
        self._snapshot: Dict[str, Any] = {
            'vehicles': {},
            'tls': {},
            'time': 0.0,
            'step': 0,
            'vehicle_count': 0,
            'avg_speed': 0.0,
            'traci_step_latency_ms': 0.0,
            'snapshot_latency_ms': 0.0,
            'traci_step_min_ms': 0.0,
            'traci_step_max_ms': 0.0,
            'traci_step_p95_ms': 0.0,
            'snapshot_min_ms': 0.0,
            'snapshot_max_ms': 0.0,
            'snapshot_p95_ms': 0.0,
            'loop_count': 0,
            'traci_errors': 0,
        }

    def _sumo_command(self) -> List[str]:
        exe = 'sumo-gui' if self.gui else 'sumo'
        # Note: do NOT include '--remote-port' here because traci.start will set it
        # when passed via the `port=` argument. Including it twice causes a
        # "remote-port already set" parse error in TraCI.
        return [exe, '-c', str(SUMO_CFG), '--start']

    def start(self) -> None:
        with self._conn_lock:
            if self._running.is_set() and self._thread and self._thread.is_alive():
                logger.info('SUMO already running')
                return
            cmd = self._sumo_command()
            logger.info('Starting SUMO: %s', cmd)
            # Start SUMO via traci.start for a single process lifecycle
            # Pass the port explicitly to traci.start to avoid duplicate CLI args
            traci.start(cmd, port=self.port)
            self._running.set()
            self._stopping.clear()
            self._paused.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def _run_loop(self) -> None:
        try:
            while self._running.is_set():
                if self._paused.is_set():
                    time.sleep(0.1)
                    continue
                t0 = time.perf_counter()
                traci.simulationStep()
                step_ms = (time.perf_counter() - t0) * 1000.0
                # record recent step durations
                try:
                    self._step_durations_ms.append(step_ms)
                except Exception:
                    pass
                self._latency_ewma_ms = 0.9 * self._latency_ewma_ms + 0.1 * step_ms
                s0 = time.perf_counter()
                self._capture_snapshot()
                snap_ms = (time.perf_counter() - s0) * 1000.0
                try:
                    self._snapshot_durations_ms.append(snap_ms)
                except Exception:
                    pass
                self._snapshot_latency_ewma_ms = 0.9 * self._snapshot_latency_ewma_ms + 0.1 * snap_ms
                self._loop_count += 1
        except Exception as e:
            self._errors += 1
            logger.exception('Error in SUMO loop: %s', e)
        finally:
            self._stopping.set()
            try:
                traci.close()
            except Exception:
                pass
            self._running.clear()
            self._paused.clear()

    def _capture_snapshot(self) -> None:
        try:
            veh_ids = traci.vehicle.getIDList()
            vehicles: Dict[str, Dict[str, Any]] = {}
            speed_total = 0.0
            for vid in veh_ids:
                pos = traci.vehicle.getPosition(vid)
                speed = traci.vehicle.getSpeed(vid)
                lane = traci.vehicle.getLaneID(vid)
                veh_type = traci.vehicle.getTypeID(vid)
                vehicles[vid] = {
                    'pos': pos,
                    'speed': speed,
                    'lane': lane,
                    'type': veh_type,
                }
                speed_total += speed

            tls_ids = traci.trafficlight.getIDList()
            tls_states: Dict[str, Dict[str, Any]] = {}
            for tl in tls_ids:
                tls_states[tl] = {
                    'state': traci.trafficlight.getRedYellowGreenState(tl),
                    'phase': traci.trafficlight.getPhase(tl),
                }

            step = traci.simulation.getTime()
            avg_speed = (speed_total / len(veh_ids)) if veh_ids else 0.0
            # compute simple stats from recent buffers
            def _stats(buf: deque[float]):
                if not buf:
                    return 0.0, 0.0, 0.0
                data = list(buf)
                mn = min(data)
                mx = max(data)
                p95 = statistics.quantiles(data, n=100)[94] if len(data) >= 100 else sorted(data)[int(len(data)*0.95)-1 if len(data)>1 else 0]
                return mn, mx, p95

            traci_min, traci_max, traci_p95 = _stats(self._step_durations_ms)
            snap_min, snap_max, snap_p95 = _stats(self._snapshot_durations_ms)

            with self._snapshot_lock:
                step_ms_value = int(step * 1000)
                self._snapshot.update({
                    'vehicles': vehicles,
                    'tls': tls_states,
                    'time': step,
                    'step': step_ms_value,
                    'vehicle_count': len(veh_ids),
                    'avg_speed': avg_speed,
                    'traci_step_latency_ms': round(self._latency_ewma_ms, 3),
                    'snapshot_latency_ms': round(self._snapshot_latency_ewma_ms, 3),
                    'traci_step_min_ms': round(traci_min, 3),
                    'traci_step_max_ms': round(traci_max, 3),
                    'traci_step_p95_ms': round(traci_p95, 3),
                    'snapshot_min_ms': round(snap_min, 3),
                    'snapshot_max_ms': round(snap_max, 3),
                    'snapshot_p95_ms': round(snap_p95, 3),
                    'loop_count': self._loop_count,
                    'traci_errors': self._errors,
                })
        except OSError:
            # TraCI can close while a snapshot is in progress during shutdown.
            # Treat this as a clean stop if the controller is no longer running.
            if self._stopping.is_set() or not self._running.is_set():
                return
            self._errors += 1
            logger.exception('Failed to capture snapshot due to TraCI socket error')
        except Exception:
            if self._stopping.is_set() or not self._running.is_set():
                return
            self._errors += 1
            logger.exception('Failed to capture snapshot')

    def stop(self) -> None:
        with self._conn_lock:
            self._stopping.set()
            self._running.clear()
            self._paused.clear()
            try:
                traci.close()
            except Exception:
                pass
            thread = self._thread
            if thread and thread.is_alive():
                thread.join(timeout=5.0)

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def is_running(self) -> bool:
        return self._running.is_set()

    def get_snapshot(self) -> Dict[str, Any]:
        with self._snapshot_lock:
            # shallow copy
            return dict(self._snapshot)

    # --- Convenience control methods ---
    def spawn_vehicle(self, veh_id: str, route_id: str | None = None, type_id: str = 'car', depart: str = 'now') -> bool:
        try:
            if route_id is None:
                routes = traci.route.getIDList()
                if not routes:
                    return False
                route_id = routes[0]
            traci.vehicle.add(veh_id, route_id, typeID=type_id, depart=depart)
            return True
        except Exception:
            logger.exception('Failed to spawn vehicle')
            return False

    def spawn_emergency_vehicle(self, base_id: str = 'ambulance') -> str | None:
        try:
            vid = f"{base_id}_{int(time.time()*1000)}"
            routes = traci.route.getIDList()
            route = routes[0] if routes else ''
            traci.vehicle.add(vid, route, typeID='emergency', depart='now')
            return vid
        except Exception:
            logger.exception('Failed to spawn emergency vehicle')
            return None

    def set_edge_max_speed(self, edge_id: str, speed: float) -> bool:
        try:
            traci.edge.setMaxSpeed(edge_id, float(speed))
            return True
        except Exception:
            logger.exception('Failed to set edge max speed')
            return False

    def close_edge(self, edge_id: str) -> bool:
        try:
            # set very low speed to emulate closure
            traci.edge.setMaxSpeed(edge_id, 0.1)
            return True
        except Exception:
            logger.exception('Failed to close edge')
            return False

    def open_edge(self, edge_id: str, speed: float = 13.9) -> bool:
        try:
            traci.edge.setMaxSpeed(edge_id, float(speed))
            return True
        except Exception:
            logger.exception('Failed to open edge')
            return False

    def get_tls_ids(self) -> List[str]:
        try:
            return traci.trafficlight.getIDList()
        except Exception:
            return []

    def set_tls_phase(self, tls_id: str, phase: int) -> bool:
        try:
            traci.trafficlight.setPhase(tls_id, int(phase))
            return True
        except Exception:
            logger.exception('Failed to set TLS phase')
            return False
