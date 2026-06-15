import time
import traci
import logging
import csv
import os
import json
from collections import deque, defaultdict
from aco_routing.congestion_costs import weighted_edge_cost, live_edge_metrics

logger = logging.getLogger(__name__)

class RouteOptimizer:
    def __init__(
        self,
        acorouter,
        adjacency,
        edge_info=None,
        reroute_interval=30,
        cost_weights=None,
        reroute_log_path=None,
        debug_log_path=None,
        protected_edge_multiplier: float = 1000.0,
        reserved_edges_by_emergency=None,
        aco_reserved_avoid_non_emergency: bool = True,
    ):
        self.acorouter = acorouter
        self.adjacency = adjacency
        self.edge_info = edge_info or {}
        self.reroute_interval = reroute_interval
        self.cost_weights = cost_weights or {}
        self.reroute_log_path = reroute_log_path
        self.protected_edge_multiplier = float(protected_edge_multiplier)
        # emergency_id -> set(edge_id)
        self.reserved_edges_by_emergency = reserved_edges_by_emergency if reserved_edges_by_emergency is not None else {}
        self.aco_reserved_avoid_non_emergency = bool(aco_reserved_avoid_non_emergency)

        # detailed per-vehicle debug CSV path
        self.debug_log_path = debug_log_path
        self._last_reroute = 0
        # track per-vehicle last reroute step to avoid excessive reroutes
        self._last_reroute_per_vehicle = {}
        # minimal time between reroutes for the same vehicle (in simulation steps)
        self.per_vehicle_cooldown = max(1, int(self.reroute_interval // 2))
        # debug counters
        self.vehicles_evaluated = 0
        self.reroute_attempts = 0
        self.successful_reroutes = 0
        self.failed_reroutes = 0
        self.skip_counts_by_reason = defaultdict(int)

    def _log_reroute(self, vehicle_id, original_route, new_route, sim_step, reason):
        if not self.reroute_log_path:
            return
        os.makedirs(os.path.dirname(self.reroute_log_path), exist_ok=True)
        row = {
            'vehicle_id': vehicle_id,
            'original_route': '>'.join(original_route or []),
            'new_route': '>'.join(new_route or []),
            'time_rerouted': int(sim_step),
            'reason': reason,
        }
        write_header = not os.path.exists(self.reroute_log_path)
        with open(self.reroute_log_path, 'a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _log_debug(self, row: dict):
        if not self.debug_log_path:
            return
        os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True)
        write_header = not os.path.exists(self.debug_log_path)
        with open(self.debug_log_path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def dump_debug_summary(self, out_path: str | None = None):
        data = {
            'vehicles_evaluated': int(self.vehicles_evaluated),
            'reroute_attempts': int(self.reroute_attempts),
            'successful_reroutes': int(self.successful_reroutes),
            'failed_reroutes': int(self.failed_reroutes),
            'skip_counts_by_reason': dict(self.skip_counts_by_reason),
        }
        if out_path:
            try:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, 'w', encoding='utf-8') as fh:
                    json.dump(data, fh, indent=2)
            except Exception:
                logger.exception('Failed to write debug summary to %s', out_path)
        return data

    def _append_integration_probe(self, entry: dict):
        try:
            out_dir = os.path.join('results')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, 'aco_integration_probe_report.json')
            # load existing list if present
            existing = []
            if os.path.exists(out_path):
                try:
                    with open(out_path, 'r', encoding='utf-8') as fh:
                        existing = json.load(fh)
                except Exception:
                    existing = []
            existing.append(entry)
            with open(out_path, 'w', encoding='utf-8') as fh:
                json.dump(existing, fh, indent=2)
        except Exception:
            logger.exception('Failed to append integration probe entry')

    def _append_runtime_connectivity_report(self, entry: dict):
        try:
            out_dir = os.path.join('results')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, 'aco_runtime_connectivity_report.json')
            existing = []
            if os.path.exists(out_path):
                try:
                    with open(out_path, 'r', encoding='utf-8') as fh:
                        existing = json.load(fh)
                except Exception:
                    existing = []
            existing.append(entry)
            with open(out_path, 'w', encoding='utf-8') as fh:
                json.dump(existing, fh, indent=2)
        except Exception:
            logger.exception('Failed to append runtime connectivity report entry')

    def _find_connectivity_path(self, source, target):
        import heapq
        dist = {source: 0}
        prev = {}
        pq = [(0, source)]
        visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == target:
                path = []
                cur = u
                while cur in prev:
                    path.append(cur)
                    cur = prev[cur]
                path.append(source)
                return list(reversed(path))
            for v in self.adjacency.get(u, []):
                if v in visited:
                    continue
                nd = d + 1
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return None

    def _make_runtime_connectivity_entry(self, sim_step, debug_row: dict, cur=None, dest=None, path=None, aco_debug=None, aco_exception=None):
        try:
            source_neighbors = list(self.adjacency.get(cur, [])) if cur is not None else []
            neighbor_pheromones = {}
            try:
                for nbr in source_neighbors[:50]:
                    neighbor_pheromones[nbr] = float(self.acorouter.ph.get(cur, nbr))
            except Exception:
                neighbor_pheromones = {}
            dijkstra_path = self._find_connectivity_path(cur, dest) if cur is not None and dest is not None else None
            dijkstra_exists = bool(dijkstra_path)
            dijkstra_length = len(dijkstra_path) if dijkstra_path else 0
            adjacency_id = id(self.adjacency) if self.adjacency is not None else None
            adjacency_size = len(self.adjacency) if hasattr(self.adjacency, '__len__') else None
            source_in_adjacency = bool(cur in self.adjacency) if cur is not None else False
            target_in_adjacency = bool(dest in self.adjacency) if dest is not None else False
            pheromone_count = None
            try:
                pheromone_count = len(self.acorouter.ph.pheromones)
            except Exception:
                pheromone_count = None

            if not source_in_adjacency or not target_in_adjacency:
                classification = 'Runtime adjacency mismatch'
            elif not source_neighbors:
                classification = 'Source edge has no neighbors'
            elif not dijkstra_exists:
                classification = 'Target unreachable'
            elif pheromone_count == 0:
                classification = 'Pheromone initialization failure'
            elif not path:
                classification = 'ACORouter search bug'
            else:
                classification = 'Other'

            entry = {
                'sim_step': int(sim_step),
                'vehicle_id': debug_row.get('vehicle_id'),
                'source_edge': cur,
                'target_edge': dest,
                'source_neighbors': source_neighbors[:50],
                'source_neighbor_pheromones': neighbor_pheromones,
                'source_has_neighbors': bool(source_neighbors),
                'target_in_adjacency': target_in_adjacency,
                'source_in_adjacency': source_in_adjacency,
                'adjacency_size': adjacency_size,
                'adjacency_id': adjacency_id,
                'pheromone_count': pheromone_count,
                'dijkstra_path_exists': dijkstra_exists,
                'dijkstra_path_length': dijkstra_length,
                'dijkstra_path': dijkstra_path,
                'aco_debug': aco_debug or {},
                'path': path or [],
                'classification': classification,
                'exception_message': debug_row.get('exception_message', '') or (str(aco_exception) if aco_exception else ''),
            }
            return entry
        except Exception:
            logger.exception('Failed to build runtime connectivity entry')
            return {
                'vehicle_id': debug_row.get('vehicle_id'),
                'error': 'runtime_connectivity_entry_failed',
            }

    def _make_probe_entry(self, debug_row: dict, cur=None, dest=None, path=None, aco_called=False, aco_debug=None, aco_exception=None):
        """Build a structured probe entry from per-vehicle debug row and runtime objects."""
        try:
            # Extract source_neighbors from actual cur value, not debug_row
            source_neighbors = list(self.adjacency.get(cur, [])) if cur is not None and self.adjacency else []
            neighbor_pheromones = {}
            try:
                for nbr in source_neighbors[:50]:
                    neighbor_pheromones[nbr] = float(self.acorouter.ph.get(cur, nbr))
            except Exception:
                neighbor_pheromones = {}
            
            adjacency_id = None
            adjacency_size = None
            adjacency_keys_sample = []
            try:
                adjacency_id = id(self.adjacency)
                adjacency_size = len(self.adjacency) if hasattr(self.adjacency, '__len__') else None
                adjacency_keys_sample = list(self.adjacency.keys())[:50]
            except Exception:
                adjacency_id = None
            acorouter_has_adjacency = hasattr(self.acorouter, 'adjacency')
            adjacency_id_equal = False
            try:
                adjacency_id_equal = (adjacency_id == id(getattr(self.acorouter, 'adjacency', None)))
            except Exception:
                adjacency_id_equal = False
            pheromone_present = False
            pheromone_count = None
            pheromone_keys_sample = []
            try:
                if hasattr(self.acorouter, 'ph') and hasattr(self.acorouter.ph, 'pheromones'):
                    pheromone_present = True
                    pheromone_count = len(self.acorouter.ph.pheromones)
                    pheromone_keys_sample = [f"{u}->{v}" for (u, v) in list(self.acorouter.ph.pheromones.keys())[:50]]
            except Exception:
                pheromone_present = False

            source_in_adjacency = bool(cur in self.adjacency) if cur is not None and self.adjacency else False
            target_in_adjacency = bool(dest in self.adjacency) if dest is not None and self.adjacency else False

            p = path or []
            path_length = len(p)
            path_continuous = debug_row.get('path_continuity_valid', False)
            cooldown_active = bool(debug_row.get('cooldown_active', False))
            interval_active = bool(debug_row.get('interval_active', True))
            attempted = bool(debug_row.get('attempted_reroute', False))
            setRoute_attempted = bool(debug_row.get('setRoute_attempted', False))
            setRoute_success = bool(debug_row.get('setRoute_success', False))
            skipped_reason = debug_row.get('skipped_reason', '') or ''
            exception_message = debug_row.get('exception_message', '') or (str(aco_exception) if aco_exception else '')

            # classification - first failure point
            classification = None
            if not aco_called:
                classification = 'ACORouter not called'
            elif aco_called and (not attempted) and skipped_reason == 'no_candidate_path':
                classification = 'ACORouter returns no path'
            elif aco_called and attempted and not path_continuous:
                classification = 'Path rejected by continuity check'
            elif not attempted and cooldown_active:
                classification = 'Path rejected by cooldown'
            elif not attempted and not interval_active:
                classification = 'Path rejected by interval logic'
            elif attempted and not setRoute_attempted:
                classification = 'setRoute not attempted'
            elif setRoute_attempted and not setRoute_success:
                classification = 'setRoute failed'
            else:
                classification = 'Other'

            entry = {
                'vehicle_id': debug_row.get('vehicle_id'),
                'source_edge': cur,
                'target_edge': dest,
                'source_neighbors': source_neighbors[:50],  # Added: actual neighbors from cur
                'source_neighbor_pheromones': neighbor_pheromones,  # Added: actual pheromone values
                'source_has_neighbors': bool(source_neighbors),  # Added: whether source has neighbors
                'path_generated': p,
                'path_length': path_length,
                'path_continuity_valid': path_continuous,
                'cooldown_active': cooldown_active,
                'interval_active': interval_active,
                'attempted_reroute': attempted,
                'setRoute_attempted': setRoute_attempted,
                'setRoute_success': setRoute_success,
                'rejection_reason': skipped_reason,
                'exception_message': exception_message,
                # adjacency / pheromone checks
                'adjacency_id': adjacency_id,
                'acorouter_has_adjacency': acorouter_has_adjacency,
                'adjacency_id_equal': adjacency_id_equal,
                'adjacency_size': adjacency_size,
                'adjacency_keys_sample': adjacency_keys_sample,
                'pheromone_present': pheromone_present,
                'pheromone_count': pheromone_count,
                'pheromone_keys_sample': pheromone_keys_sample,
                'source_in_adjacency': source_in_adjacency,
                'target_in_adjacency': target_in_adjacency,
                'aco_called': aco_called,
                'aco_debug': aco_debug or {},
                'classification': classification,
            }
            return entry
        except Exception:
            logger.exception('Failed to build probe entry')
            return {'vehicle_id': debug_row.get('vehicle_id'), 'error': 'probe_build_failed'}

    def step(self, sim_step, pinn_predictions=None):
        # Reroute vehicles at set intervals
        # Emergency corridor edge protection (Phase 2):
        # If reserved_edges_by_emergency is populated and a given edge is reserved,
        # inflate the cost for *non-emergency* vehicles only.
        # Emergency vehicles ignore the penalty.
        #
        # Note: We keep reserved edges per emergency_id in self.reserved_edges_by_emergency;
        # for cost inflation we use the union of all reserved edges.
        #
        # This must be best-effort and must never break routing.

        if sim_step - self._last_reroute < self.reroute_interval:
            return
        self._last_reroute = sim_step

        vehs = traci.vehicle.getIDList()
        for vid in vehs:
            try:

                # per-vehicle diagnostics
                self.vehicles_evaluated += 1
                # keep reserved-edge inflation heuristic consistent for this vehicle id
                # (closure in heuristic references vid)
                _ = vid
                # skip vehicles that are arrived

                if traci.vehicle.isStopped(vid):

                    self.skip_counts_by_reason['stopped_vehicle'] += 1
                    row = {
                        'vehicle_id': vid,
                        'current_route': '>'.join(traci.vehicle.getRoute(vid) or []),
                        'attempted_reroute': False,
                        'skipped_reason': 'stopped_vehicle',
                        'pheromone_score': 0.0,
                        'congestion_score': 0.0,
                        'cooldown_active': False,
                        'interval_active': True,
                        'candidate_path_found': False,
                        'path_continuity_valid': False,
                        'setRoute_attempted': False,
                        'setRoute_success': False,
                        'exception_message': '',
                    }
                    try:
                        probe = self._make_probe_entry(row, cur=None, dest=None, path=None, aco_called=False, aco_debug=None, aco_exception=None)
                        self._append_integration_probe(probe)
                    except Exception:
                        logger.exception('Failed to write probe for stopped vehicle %s', vid)
                    self._log_debug(row)
                    continue
                # enforce per-vehicle reroute cooldown
                last_v = self._last_reroute_per_vehicle.get(vid, -10**9)
                if sim_step - last_v < self.per_vehicle_cooldown:
                    self.skip_counts_by_reason['cooldown'] += 1
                    row = {
                        'vehicle_id': vid,
                        'current_route': '>'.join(traci.vehicle.getRoute(vid) or []),
                        'attempted_reroute': False,
                        'skipped_reason': 'cooldown',
                        'pheromone_score': 0.0,
                        'congestion_score': 0.0,
                        'cooldown_active': True,
                        'interval_active': True,
                        'candidate_path_found': False,
                        'path_continuity_valid': False,
                        'setRoute_attempted': False,
                        'setRoute_success': False,
                        'exception_message': '',
                    }
                    try:
                        probe = self._make_probe_entry(row, cur=None, dest=None, path=None, aco_called=False, aco_debug=None, aco_exception=None)
                        self._append_integration_probe(probe)
                    except Exception:
                        logger.exception('Failed to write probe for cooldown vehicle %s', vid)
                    self._log_debug(row)
                    continue
                route = traci.vehicle.getRoute(vid)
                if not route or len(route) < 2:
                    self.skip_counts_by_reason['invalid_route'] += 1
                    row = {
                        'vehicle_id': vid,
                        'current_route': '>'.join(route or []),
                        'attempted_reroute': False,
                        'skipped_reason': 'invalid_route',
                        'pheromone_score': 0.0,
                        'congestion_score': 0.0,
                        'cooldown_active': False,
                        'interval_active': True,
                        'candidate_path_found': False,
                        'path_continuity_valid': False,
                        'setRoute_attempted': False,
                        'setRoute_success': False,
                        'exception_message': '',
                    }
                    try:
                        probe = self._make_probe_entry(row, cur=None, dest=None, path=None, aco_called=False, aco_debug=None, aco_exception=None)
                        self._append_integration_probe(probe)
                    except Exception:
                        logger.exception('Failed to write probe for invalid route vehicle %s', vid)
                    self._log_debug(row)
                    continue
                original_route = list(route)
                # get current edge and destination
                cur = traci.vehicle.getRoadID(vid)
                dest = traci.vehicle.getRoute(vid)[-1]
                if cur.startswith(':') or dest.startswith(':'):
                    self.skip_counts_by_reason['invalid_route'] += 1
                    row = {
                        'vehicle_id': vid,
                        'current_route': '>'.join(original_route or []),
                        'attempted_reroute': False,
                        'skipped_reason': 'invalid_route',
                        'pheromone_score': 0.0,
                        'congestion_score': 0.0,
                        'cooldown_active': False,
                        'interval_active': True,
                        'candidate_path_found': False,
                        'path_continuity_valid': False,
                        'setRoute_attempted': False,
                        'setRoute_success': False,
                        'exception_message': '',
                    }
                    try:
                        probe = self._make_probe_entry(row, cur=cur, dest=dest, path=None, aco_called=False, aco_debug=None, aco_exception=None)
                        self._append_integration_probe(probe)
                    except Exception:
                        logger.exception('Failed to write probe for internal-edge vehicle %s', vid)
                    self._log_debug(row)
                    continue
                # compute new route via ACO
                # prepare heuristic function binding current predicted congestion
                # IMPORTANT: Phase-2 protected emergency corridor cost inflation is applied
                # only for NON-emergency vehicles.
                def heuristic(e):
                    pred = None
                    if pinn_predictions is not None:
                        pred = pinn_predictions.get(e, 0.0)
                    info = self.edge_info.get(e, {})
                    live_metrics = live_edge_metrics(e)
                    base_cost = weighted_edge_cost(
                        e,
                        edge_info=info,
                        live_metrics=live_metrics,
                        pinn_pred=pred,
                        weights=self.cost_weights,
                    )

                    # Apply reserved-edge inflation for non-emergency vehicles.
                    if self.aco_reserved_avoid_non_emergency and self.reserved_edges_by_emergency:
                        # Determine if current vehicle is an emergency type.
                        is_emergency = False
                        try:
                            vtype = traci.vehicle.getTypeID(vid).lower()
                            is_emergency = any(
                                tok in vtype
                                for tok in (
                                    'emergency', 'ambulance', 'police', 'firetruck', 'fire_truck', 'rescue'
                                )
                            )
                        except Exception:
                            is_emergency = False

                        if not is_emergency:
                            reserved_union = set()
                            for _eid, edges in self.reserved_edges_by_emergency.items():
                                reserved_union.update(edges or [])
                            if e in reserved_union:
                                return float(base_cost) * float(self.protected_edge_multiplier)

                    return base_cost


                aco_called = False
                aco_exception = None
                path = None
                cost = None
                # DEBUG: capture adjacency state before ACO call
                neighbor_count = len(self.adjacency.get(cur, [])) if self.adjacency else 0
                adj_size_at_call = len(self.adjacency) if self.adjacency else 0
                # Log first few vehicles to diagnose adjacency issue
                if self.vehicles_evaluated <= 15:  # Only log first 15 vehicles
                    print(f'[ACO CALL #{self.vehicles_evaluated}] cur={cur}, neighbors_in_adj={neighbor_count}, adj_total_size={adj_size_at_call}')
                try:
                    path, cost = self.acorouter.find_route(cur, dest, self.adjacency, heuristic_fn=heuristic)
                    aco_called = True
                except Exception as e:
                    aco_called = True
                    aco_exception = e
                    logger.exception('ACORouter raised exception for %s->%s', cur, dest)
                # attach ACO debug info if available
                aco_debug = getattr(self.acorouter, 'last_debug_info', {}) or {}
                candidate_found = bool(path and len(path) > 1)
                path_continuous = self._is_continuous_path(path) if candidate_found else False

                # compute pheromone and congestion scores for debugging
                pheromone_score = 0.0
                congestion_score = 0.0
                if candidate_found:
                    try:
                        for u, v in zip(path[:-1], path[1:]):
                            pheromone_score += float(getattr(self.acorouter, 'ph').get(u, v))
                            if pinn_predictions is not None:
                                congestion_score += float(pinn_predictions.get(v, 0.0))
                            else:
                                # fallback to live edge queue length
                                congestion_score += float(live_edge_metrics(v).get('queue_length', 0.0))
                    except Exception:
                        logger.exception('Failed to compute debug scores for path %s', path)

                # prepare debug row base
                debug_row = {
                    'vehicle_id': vid,
                    'current_route': '>'.join(original_route or []),
                    'attempted_reroute': False,
                    'skipped_reason': '',
                    'pheromone_score': float(pheromone_score),
                    'congestion_score': float(congestion_score),
                    'cooldown_active': False,
                    'interval_active': True,
                    'candidate_path_found': bool(candidate_found),
                    'path_continuity_valid': bool(path_continuous),
                    'setRoute_attempted': False,
                    'setRoute_success': False,
                    'exception_message': '',
                    # ACO diagnostics
                    'aco_ants_launched': int(aco_debug.get('ants_launched', 0)),
                    'aco_iterations': int(aco_debug.get('iterations', 0)),
                    'aco_successful_paths': int(aco_debug.get('successful_paths', 0)),
                    'aco_failed_paths': int(aco_debug.get('failed_paths', 0)),
                    'aco_explored_nodes_count': int(aco_debug.get('explored_nodes_count', 0)),
                    'aco_top_pheromones': json.dumps(aco_debug.get('top_pheromones', [])),
                }

                if not candidate_found:
                    # runtime connectivity validation for failed ACO routes
                    try:
                        connectivity_entry = self._make_runtime_connectivity_entry(sim_step, debug_row, cur=cur, dest=dest, path=path, aco_debug=aco_debug, aco_exception=aco_exception)
                        self._append_runtime_connectivity_report(connectivity_entry)
                    except Exception:
                        logger.exception('Failed to write runtime connectivity report for failed ACO route %s', vid)
                    # try Dijkstra fallback before giving up
                    self.skip_counts_by_reason['no_candidate_path'] += 1
                    fallback = self._dijkstra_shortest_path(cur, dest, heuristic)
                    if fallback and len(fallback) > 1:
                        path = fallback
                        candidate_found = True
                        path_continuous = self._is_continuous_path(path)
                        debug_row['skipped_reason'] = 'dijkstra_fallback'
                    else:
                        debug_row['attempted_reroute'] = False
                        debug_row['skipped_reason'] = 'no_candidate_path'
                        try:
                            probe = self._make_probe_entry(debug_row, cur=cur, dest=dest, path=path, aco_called=aco_called, aco_debug=aco_debug, aco_exception=aco_exception)
                            self._append_integration_probe(probe)
                        except Exception:
                            logger.exception('Failed to write integration probe for no_candidate_path %s', vid)
                        self._log_debug(debug_row)
                        continue

                if not path_continuous:
                    self.skip_counts_by_reason['discontinuous_path'] += 1
                    debug_row['attempted_reroute'] = False
                    debug_row['skipped_reason'] = 'discontinuous_path'
                    try:
                        probe = self._make_probe_entry(debug_row, cur=cur, dest=dest, path=path, aco_called=aco_called, aco_debug=aco_debug, aco_exception=aco_exception)
                        self._append_integration_probe(probe)
                    except Exception:
                        logger.exception('Failed to write integration probe for discontinuous_path %s', vid)
                    self._log_debug(debug_row)
                    continue

                # At this point we will attempt to setRoute
                if path and path[0] != cur:
                    debug_row['attempted_reroute'] = False
                    debug_row['skipped_reason'] = 'current_edge_mismatch'
                    self.skip_counts_by_reason['current_edge_mismatch'] += 1
                    continue
                
                debug_row['attempted_reroute'] = True
                self.reroute_attempts += 1
                try:
                    traci.vehicle.setRoute(vid, path)
                    self._last_reroute_per_vehicle[vid] = sim_step
                    self._log_reroute(vid, original_route, path, sim_step, 'aco_setRoute')
                    debug_row['setRoute_attempted'] = True
                    debug_row['setRoute_success'] = True
                    self.successful_reroutes += 1
                    try:
                        probe = self._make_probe_entry(debug_row, cur=cur, dest=dest, path=path, aco_called=aco_called, aco_debug=aco_debug, aco_exception=aco_exception)
                        self._append_integration_probe(probe)
                    except Exception:
                        logger.exception('Failed to write integration probe for successful setRoute %s', vid)
                    self._log_debug(debug_row)
                except Exception as e:
                    debug_row['setRoute_attempted'] = True
                    debug_row['setRoute_success'] = False
                    debug_row['exception_message'] = str(e)
                    # try fallback
                    try:
                        traci.vehicle.changeTarget(vid, dest)
                        self._last_reroute_per_vehicle[vid] = sim_step
                        self._log_reroute(vid, original_route, path, sim_step, 'aco_changeTarget')
                        self.failed_reroutes += 1
                        try:
                            probe = self._make_probe_entry(debug_row, cur=cur, dest=dest, path=path, aco_called=aco_called, aco_debug=aco_debug, aco_exception=aco_exception)
                            self._append_integration_probe(probe)
                        except Exception:
                            logger.exception('Failed to write integration probe for changeTarget fallback %s', vid)
                        self._log_debug(debug_row)
                    except Exception as e2:
                        self.failed_reroutes += 1
                        debug_row['exception_message'] = debug_row['exception_message'] + ' | changeTarget: ' + str(e2)
                        logger.exception('Failed to change route for vehicle %s', vid)
                        try:
                            probe = self._make_probe_entry(debug_row, cur=cur, dest=dest, path=path, aco_called=aco_called, aco_debug=aco_debug, aco_exception=aco_exception)
                            self._append_integration_probe(probe)
                        except Exception:
                            logger.exception('Failed to write integration probe for failed changeTarget %s', vid)
                        self._log_debug(debug_row)

            except Exception:
                # log per-vehicle errors so hidden failures become visible
                logger.exception('Per-vehicle routing error for %s at step %s', vid, sim_step)

    def _is_continuous_path(self, path):
        if not path or len(path) < 2:
            return False
        for u, v in zip(path[:-1], path[1:]):
            outs = self.adjacency.get(u, [])
            if v not in outs:
                return False
        return True

    def _dijkstra_shortest_path(self, source, target, cost_fn):
        # cost_fn(edge) -> numeric cost for entering edge
        import heapq
        dist = {source: 0.0}
        prev = {}
        pq = [(0.0, source)]
        visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == target:
                # reconstruct path
                path = []
                cur = u
                while cur in prev:
                    path.append(cur)
                    cur = prev[cur]
                path.append(source)
                return list(reversed(path))
            for v in self.adjacency.get(u, []):
                if v in visited:
                    continue
                try:
                    w = float(cost_fn(v))
                except Exception:
                    w = 1.0
                nd = d + max(1e-6, w)
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return None
