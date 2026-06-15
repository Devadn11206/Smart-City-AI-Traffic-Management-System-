import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from aco_routing.pheromone_manager import PheromoneManager
from aco_routing.aco_router import ACORouter

adj = {'A':['B','C'], 'B':['D'], 'C':['D'], 'D':[]}
ph = PheromoneManager(adj, initial_pheromone=1.0, evaporation_rate=0.1)
router = ACORouter(ph, lambda e: 1.0, ants=10, max_steps=10)
path, cost = router.find_route('A','D', adj)
print('ACO path:', path, 'cost:', cost)
ph.dump_csv(os.path.join(ROOT, 'results', 'pheromone_test.csv'))
print('Pheromone CSV written')
