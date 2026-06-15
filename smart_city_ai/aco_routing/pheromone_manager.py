import math
import random

class PheromoneManager:
    """Manage pheromone values between directed edges in the road network.

    Stores pheromone as a dict keyed by (from_edge, to_edge) -> float.
    Provides evaporation and reinforcement operations.
    """
    def __init__(self, adjacency, initial_pheromone=1.0, evaporation_rate=0.05):
        # adjacency: dict ed.ge_id -> list(outgoing_edge_ids)
        self.adjacency = adjacency
        self.evaporation_rate = evaporation_rate
        self.pheromones = {}
        for u, outs in adjacency.items():
            for v in outs:
                self.pheromones[(u, v)] = float(initial_pheromone)

    def get(self, u, v):
        return self.pheromones.get((u, v), 0.0)

    def set(self, u, v, value):
        self.pheromones[(u, v)] = float(value)

    def evaporate(self):
        for k in list(self.pheromones.keys()):
            self.pheromones[k] = max(1e-6, self.pheromones[k] * (1.0 - self.evaporation_rate))

    def reinforce_path(self, path, reward):
        """Increase pheromone for edges on the path proportional to reward."""
        if reward <= 0:
            return
        delta = reward / max(1.0, len(path) - 1)
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            self.pheromones[(u, v)] = self.pheromones.get((u, v), 0.0) + delta

    def dump_csv(self, path):
        """Dump current pheromone values to CSV: from,to,pheromone"""
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['from', 'to', 'pheromone'])
            for (u, v), val in sorted(self.pheromones.items()):
                writer.writerow([u, v, val])

    def best_edge_from(self, u):
        outs = self.adjacency.get(u, [])
        if not outs:
            return None
        best = max(outs, key=lambda v: self.get(u, v))
        return best
