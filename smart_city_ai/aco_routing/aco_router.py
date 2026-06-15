import random
import math
from collections import deque

class ACORouter:
    """Simple Ant Colony Optimization router for SUMO networks.

    The router performs multiple ant walks between a source and a destination edge and
    returns the best found path according to a cost function. It relies on a PheromoneManager
    and a heuristic function that scores edges (lower is better).
    """
    def __init__(self, pheromone_manager, heuristic_fn, alpha=1.0, beta=2.0, ants=20, max_steps=200):
        self.ph = pheromone_manager
        self.heuristic = heuristic_fn
        self.alpha = alpha
        self.beta = beta
        self.ants = ants
        self.max_steps = max_steps

    def _transition_probabilities(self, u, neighbors, heuristic_fn=None):
        weights = []
        for v in neighbors:
            tau = max(1e-9, self.ph.get(u, v))
            h = heuristic_fn(v) if heuristic_fn is not None else self.heuristic(v)
            eta = 1.0 / max(1e-6, h)
            weights.append((tau ** self.alpha) * (eta ** self.beta))
        s = sum(weights)
        if s == 0:
            return [1.0 / len(weights)] * len(weights)
        return [w / s for w in weights]

    def _probabilistic_choice(self, neighbors, probs):
        r = random.random()
        cum = 0.0
        for n, p in zip(neighbors, probs):
            cum += p
            if r <= cum:
                return n
        return neighbors[-1]

    def find_route(self, source, target, adjacency, max_iterations=None, heuristic_fn=None, ants=None, max_steps=None):
        """Run ants from source to target and return the best path and its cost."""
        best_path = None
        best_cost = float('inf')
        
        num_ants = ants if ants is not None else self.ants
        steps_limit = max_steps if max_steps is not None else self.max_steps
        
        ants_launched = 0
        successful_paths = 0
        failed_paths = 0
        explored_nodes = set()

        iterations = num_ants if max_iterations is None else max_iterations
        for i in range(iterations):
            ants_launched += 1
            path = [source]
            cur = source
            steps = 0
            visited = set([cur])
            rejection_reason = None
            while cur != target and steps < steps_limit:
                neighbors = adjacency.get(cur, [])
                if not neighbors:
                    rejection_reason = 'no_neighbors'
                    break
                probs = self._transition_probabilities(cur, neighbors, heuristic_fn=heuristic_fn)
                nxt = self._probabilistic_choice(neighbors, probs)
                path.append(nxt)
                cur = nxt
                explored_nodes.add(cur)
                steps += 1
                # prevent infinite loops by limiting revisits
                if path.count(cur) > 5:
                    rejection_reason = 'too_many_revisits'
                    break

            if path[-1] == target:
                successful_paths += 1
                # compute cost as sum of heuristic over edges
                cost = sum((heuristic_fn(e) if heuristic_fn is not None else self.heuristic(e)) for e in path[1:])
                # reinforce pheromones proportional to inverse cost
                reward = 1.0 / max(1e-6, cost)
                self.ph.reinforce_path(path, reward)
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
            else:
                failed_paths += 1

        # evaporate after each invocation
        self.ph.evaporate()

        # expose debugging information for callers
        try:
            top_ph = None
            # sample top pheromone edges
            items = sorted(self.ph.pheromones.items(), key=lambda kv: -kv[1])[:10]
            top_ph = [(u + '->' + v, float(val)) for ((u, v), val) in items]
        except Exception:
            top_ph = []

        self.last_debug_info = {
            'ants_launched': int(ants_launched),
            'iterations': int(iterations),
            'successful_paths': int(successful_paths),
            'failed_paths': int(failed_paths),
            'explored_nodes_count': int(len(explored_nodes)),
            'explored_nodes_sample': list(list(explored_nodes)[:20]),
            'top_pheromones': top_ph,
        }

        return best_path, best_cost
