from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def ensure_parent_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def load_torch_checkpoint(path: str | Path, map_location: str | torch.device = 'cpu') -> Any:
    """Load a trusted local checkpoint across PyTorch versions.

    PyTorch 2.6 changed the default `weights_only` behavior. Our checkpoints are
    created locally inside this repository, so we explicitly allow full loading.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def save_torch_checkpoint(path: str | Path, state_dict: dict, *, metadata: dict | None = None) -> None:
    """Save a checkpoint with a consistent metadata wrapper.

    The saved format is a dict with keys: 'model_state_dict' and optional metadata
    such as 'state_dim', 'action_dim', 'agent', 'version'. This helps with
    compatibility when loading checkpoints produced at different code versions.
    """
    wrapper = {'model_state_dict': state_dict}
    if metadata:
        wrapper.update(metadata)
    output = ensure_parent_dir(path)
    torch.save(wrapper, str(output))


def extract_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ('model_state_dict', 'state_dict', 'policy_state_dict', 'ddqn_state_dict'):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint
    raise TypeError(f'Checkpoint does not contain a valid state dict: {type(checkpoint).__name__}')


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    output = ensure_parent_dir(path)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
