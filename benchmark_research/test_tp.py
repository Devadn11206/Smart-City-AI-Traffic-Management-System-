import sys
sys.path.append('c:/Users/ddnan/intenrship_IIIT/benchmark_research')
from importlib import import_module
env_module = import_module('6x6_benchmark.env.sumo_env_6x6')
net_file = 'c:/Users/ddnan/intenrship_IIIT/benchmark_research/6x6_benchmark/data/grid6x6.net.xml'
route_file = 'c:/Users/ddnan/intenrship_IIIT/benchmark_research/6x6_benchmark/data/routes.rou.xml'
env = env_module.SumoMultiAgentEnv6x6(net_file, route_file, use_gui=False)
env.reset()
total_tp = 0
for _ in range(300):
    _,_,_,_,infos=env.step({ts:0 for ts in env.ts_ids})
    total_tp += sum([infos['step'][ts]['throughput'] for ts in env.ts_ids])
print('Total throughput over 300 steps:', total_tp)
env.close()
