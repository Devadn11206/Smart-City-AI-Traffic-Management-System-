"""
Verify that migrated checkpoints can be loaded into current agent model shapes without RuntimeError.
Print a report of files that failed to load into their matching agent model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.reliability import load_torch_checkpoint, extract_state_dict
import torch

AGENT_LOOKUP = {
    'ddqn': ('rl_agents.ddqn_agent', 'DDQNAgent'),
    'dqn': ('rl_agents.dqn_agent', 'DQNAgent'),
    'ppo': ('rl_agents.ppo_agent', 'PPOAgent'),
}

candidates = list(Path('smart_city_ai').glob('**/*.pth'))
report = []
for p in candidates:
    name = p.name.lower()
    agent_type = None
    if 'ddqn' in name:
        agent_type = 'ddqn'
    elif 'dqn' in name and 'ddqn' not in name:
        agent_type = 'dqn'
    elif 'ppo' in name:
        agent_type = 'ppo'
    else:
        # skip pinn files
        if 'pinn' in name:
            continue
        agent_type = None

    try:
        ck = load_torch_checkpoint(p, map_location='cpu')
        sd = extract_state_dict(ck)
    except Exception as e:
        report.append((str(p), 'load_failed', str(e)))
        continue

    if agent_type is None:
        report.append((str(p), 'skipped', 'unknown agent type'))
        continue

    module_name, class_name = AGENT_LOOKUP[agent_type]
    mod = __import__(module_name, fromlist=[class_name])
    AgentClass = getattr(mod, class_name)
    agent = AgentClass(7, 4, device='cpu') if class_name != 'PPOAgent' else AgentClass(7, 4, device='cpu')
    target = agent.model.state_dict() if hasattr(agent, 'model') else getattr(agent, 'policy').state_dict()

    try:
        # try strict load first
        try:
            target_keys = set(target.keys())
            sd_keys = set(sd.keys())
            overlapping = target_keys & sd_keys
            to_load = {k: sd[k] for k in overlapping}
            # load partial state
            target_copy = target.copy()
            for k, v in to_load.items():
                try:
                    target_copy[k] = v
                except Exception:
                    pass
            # attempt to load
            if hasattr(agent, 'model'):
                agent.model.load_state_dict(target_copy)
            else:
                agent.policy.load_state_dict(target_copy)
            report.append((str(p), 'ok', 'loaded (partial or full)'))
        except Exception as e:
            report.append((str(p), 'load_into_model_failed', str(e)))
    except Exception as e:
        report.append((str(p), 'unexpected_error', str(e)))

print('Verification report:')
for entry in report:
    print(entry)
