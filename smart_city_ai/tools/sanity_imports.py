import importlib
import sys
# ensure project root is on PYTHONPATH
sys.path.append('.')
mods=['rl_agents.ddqn_agent','rl_agents.ppo_agent','rl_agents.dqn_agent']
for m in mods:
    try:
        importlib.import_module(m)
        print('OK',m)
    except Exception as e:
        print('ERROR',m,e)
        sys.exit(1)
print('All imports OK')
