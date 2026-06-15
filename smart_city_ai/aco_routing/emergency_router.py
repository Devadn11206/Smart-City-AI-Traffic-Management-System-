import logging
import traci

logger = logging.getLogger(__name__)


class EmergencyRouter:
    def __init__(self, acorouter, adjacency, edge_info=None):
        self.acorouter = acorouter
        self.adjacency = adjacency
        self.edge_info = edge_info or {}

    def route_emergency(self, vehicle_id, destination, pinn_predictions=None):
        # Check connection before making traci calls
        try:
            if not traci.isLoaded():
                logger.warning('TraCI connection not active; cannot route emergency vehicle')
                return []
        except Exception:
            logger.warning('Failed to check TraCI connection; skipping emergency routing')
            return []
        
        # Use ACORouter with more emphasis on predicted congestion avoidance
        def heuristic(e):
            pred = None
            if pinn_predictions is not None:
                pred = pinn_predictions.get(e, 0.0)
            from aco_routing.congestion_costs import default_edge_heuristic
            return default_edge_heuristic(e, self.edge_info.get(e, {}), pred)

        try:
            source = traci.vehicle.getRoadID(vehicle_id)
        except Exception as ex:
            logger.exception('Failed to get road ID for vehicle %s', vehicle_id)
            return []
        path, cost = self.acorouter.find_route(source, destination, self.adjacency)
        if path and len(path) > 1:
            try:
                # Only call setRoute if the vehicle's current edge is present in the new path.
                # If not, use changeTarget which avoids the "current edge not found in new route" error.
                current_edge = source
            except Exception:
                current_edge = None

            try:
                if current_edge and current_edge in path:
                    traci.vehicle.setRoute(vehicle_id, path)
                else:
                    logger.debug('Current edge %s not in new path for %s; using changeTarget instead.', current_edge, vehicle_id)
                    traci.vehicle.changeTarget(vehicle_id, destination)
            except Exception as exc:
                logger.exception('Failed to apply route/change target for %s: %s', vehicle_id, exc)
        return path
