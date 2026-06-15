"""
Migrate existing model checkpoints to the new metadata-wrapped format and
attempt to adapt mismatched linear layer shapes by copying overlaps and
preserving as much of the original parameters as possible.

Backups of original files are stored under results/checkpoint_backups/<timestamp>/
"""
import os
import shutil
import time
from pathlib import Path
import torch
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.reliability import load_torch_checkpoint, extract_state_dict, save_torch_checkpoint

# list of candidate checkpoint files to migrate (search common folders)
CANDIDATES = list(Path('.').glob('**/*.pth'))

BACKUP_ROOT = Path('results') / 'checkpoint_backups' / time.strftime('%Y%m%d_%H%M%S')
BACKUP_ROOT.mkdir(parents=True, exist_ok=True)

print(f'Found {len(CANDIDATES)} .pth files. Backing up originals to {BACKUP_ROOT}')

# helper to safely copy overlapping tensor shapes

def merge_param(target_tensor: torch.Tensor, src_tensor: torch.Tensor) -> torch.Tensor:
    """Copy overlapping entries from src_tensor into target_tensor and return new tensor."""
    out = target_tensor.clone()
    # determine common slice indices
    if out.dim() != src_tensor.dim():
        return out
    if out.dim() == 2:
        r = min(out.size(0), src_tensor.size(0))
        c = min(out.size(1), src_tensor.size(1))
        out[:r, :c] = src_tensor[:r, :c]
    elif out.dim() == 1:
        l = min(out.size(0), src_tensor.size(0))
        out[:l] = src_tensor[:l]
    else:
        # for higher dims, copy the minimal subtensor on each axis
        sizes = [min(a,b) for a,b in zip(out.size(), src_tensor.size())]
        slices = tuple(slice(0,s) for s in sizes)
        out[slices] = src_tensor[slices]
    return out

# load agent model types to create compatible model instances
AGENT_FACTORIES = {
    'dqn': ('rl_agents.dqn_agent', 'DQNAgent'),
    'ddqn': ('rl_agents.ddqn_agent', 'DDQNAgent'),
    'ppo': ('rl_agents.ppo_agent', 'PPOAgent'),
}

migrated = []
errors = []
for p in CANDIDATES:
    p = Path(p)
    try:
        # skip policy files that look like PINN or non-pt format (we still back them up)
        if 'pinn' in p.name.lower() or p.stat().st_size == 0:
            shutil.copy2(p, BACKUP_ROOT / p.name)
            continue

        # backup original
        shutil.copy2(p, BACKUP_ROOT / p.name)

        # try load via wrapper
        try:
            ck = load_torch_checkpoint(p, map_location='cpu')
        except Exception:
            ck = torch.load(p, map_location='cpu')

        # try to extract state dict
        try:
            sd = extract_state_dict(ck)
        except Exception:
            # If object is plain state dict, use it
            if isinstance(ck, dict) and all(isinstance(k, str) for k in ck.keys()):
                sd = ck
            else:
                raise

        # guess agent type from metadata or filename
        meta_agent = None
        if isinstance(ck, dict):
            meta_agent = ck.get('agent') or ck.get('agent_type')
        if not meta_agent:
            name = p.name.lower()
            if 'ddqn' in name:
                meta_agent = 'ddqn'
            elif 'dqn' in name and 'ddqn' not in name:
                meta_agent = 'dqn'
            elif 'ppo' in name:
                meta_agent = 'ppo'

        # fallback: just wrap and save using existing state dict
        if not meta_agent:
            save_torch_checkpoint(p, sd, metadata={'version': 'migrated-1.0', 'timestamp': time.time()})
            migrated.append(p)
            continue

        # instantiate a fresh agent model to get target param shapes
        module_name, class_name = AGENT_FACTORIES.get(meta_agent, (None, None))
        if module_name is None:
            # unknown agent type, wrap and continue
            save_torch_checkpoint(p, sd, metadata={'agent': meta_agent, 'version': 'migrated-1.0', 'timestamp': time.time()})
            migrated.append(p)
            continue

        mod = __import__(module_name, fromlist=[class_name])
        AgentClass = getattr(mod, class_name)
        # use default dims used across code: state_dim=7, action_dim=4
        agent = AgentClass(7, 4, device='cpu') if class_name != 'PPOAgent' else AgentClass(7, 4, device='cpu')
        target_sd = agent.model.state_dict() if hasattr(agent, 'model') else getattr(agent, 'policy').state_dict()

        # produce merged state dict
        merged = {}
        for k, v in target_sd.items():
            if k in sd:
                src = sd[k]
                if isinstance(src, (list, tuple)):
                    src = torch.tensor(src)
                if isinstance(src, torch.Tensor):
                    if src.shape == v.shape:
                        merged[k] = src.clone()
                    else:
                        merged[k] = merge_param(v, src)
                else:
                    # if not tensor, try to coerce
                    try:
                        ssrc = torch.tensor(src)
                        merged[k] = merge_param(v, ssrc)
                    except Exception:
                        merged[k] = v
            else:
                merged[k] = v

        # save wrapped checkpoint metadata
        metadata = {
            'agent': meta_agent,
            'state_dim': 7,
            'action_dim': 4,
            'version': 'migrated-1.0',
            'timestamp': time.time(),
        }
        save_torch_checkpoint(p, merged, metadata=metadata)
        migrated.append(p)
        print(f'Migrated {p}')
    except Exception as e:
        errors.append((p, str(e)))
        print(f'Error migrating {p}: {e}')

print('\nMigration complete')
print('Migrated:', len(migrated))
print('Errors:', len(errors))
if errors:
    for p, msg in errors:
        print(p, msg)
