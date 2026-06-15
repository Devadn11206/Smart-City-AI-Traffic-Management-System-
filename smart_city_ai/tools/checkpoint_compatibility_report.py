import os
import json
from dataclasses import dataclass
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))
BACKUP_DIR = os.path.join(WORKSPACE_ROOT, 'results', 'checkpoint_backups', '20260530_203534')
CURRENT_DIR = os.path.join(PROJECT_ROOT, 'results', 'hybrid_results')
REPORT_DIR = os.path.join(WORKSPACE_ROOT, 'results', 'aco_results')
CURRENT_STATE_DIM = 9
CURRENT_ACTION_DIM = 4


@dataclass
class CheckpointInfo:
    path: str
    algorithm: str
    saved_state_dim: int | None
    saved_action_dim: int | None
    saved_architecture: str
    current_state_dim: int
    current_action_dim: int
    current_architecture: str
    compatibility_status: str
    strict_load_ok: bool
    inference_ok: bool
    inference_summary: str
    load_error: str | None = None


def _read_state_dict(path):
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict'], checkpoint
    if isinstance(checkpoint, dict) and any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state_dict = {k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}
        return state_dict, checkpoint
    return checkpoint, checkpoint


def _infer_architecture(algorithm, state_dict):
    saved_state_dim = None
    saved_action_dim = None
    if algorithm in ('dqn', 'ddqn'):
        saved_state_dim = int(state_dict['fc1.weight'].shape[1])
        saved_action_dim = int(state_dict['fc3.weight'].shape[0])
        saved_arch = f"QNetwork({saved_state_dim} -> 64 -> 64 -> {saved_action_dim})"
    else:
        saved_state_dim = int(state_dict['actor.0.weight'].shape[1])
        saved_action_dim = int(state_dict['actor.4.weight'].shape[0])
        saved_arch = (
            f"ActorCritic(actor: {saved_state_dim} -> 64 -> 64 -> {saved_action_dim}, "
            f"critic: {saved_state_dim} -> 64 -> 64 -> 1)"
        )
    return saved_state_dim, saved_action_dim, saved_arch


def _current_architecture(algorithm):
    if algorithm in ('dqn', 'ddqn'):
        return f"QNetwork({CURRENT_STATE_DIM} -> 64 -> 64 -> {CURRENT_ACTION_DIM})"
    return (
        f"ActorCritic(actor: {CURRENT_STATE_DIM} -> 64 -> 64 -> {CURRENT_ACTION_DIM}, "
        f"critic: {CURRENT_STATE_DIM} -> 64 -> 64 -> 1)"
    )


def _build_agent(algorithm):
    if algorithm == 'dqn':
        return DQNAgent(CURRENT_STATE_DIM, CURRENT_ACTION_DIM, device='cpu')
    if algorithm == 'ddqn':
        return DDQNAgent(CURRENT_STATE_DIM, CURRENT_ACTION_DIM, device='cpu')
    return PPOAgent(CURRENT_STATE_DIM, CURRENT_ACTION_DIM, device='cpu')


def _strict_load_and_infer(path, algorithm):
    agent = _build_agent(algorithm)
    state_dict, raw = _read_state_dict(path)
    try:
        if algorithm == 'dqn':
            agent.model.load_state_dict(state_dict, strict=True)
        elif algorithm == 'ddqn':
            agent.model.load_state_dict(state_dict, strict=True)
        else:
            agent.policy.load_state_dict(state_dict, strict=True)
            agent.policy_old.load_state_dict(agent.policy.state_dict())
        strict_load_ok = True
        load_error = None
    except Exception as exc:
        strict_load_ok = False
        load_error = str(exc)

    inference_ok = False
    inference_summary = ''
    try:
        sample = torch.zeros((1, CURRENT_STATE_DIM), dtype=torch.float32)
        if algorithm in ('dqn', 'ddqn'):
            with torch.no_grad():
                out = agent.model(sample)
            inference_ok = out.shape == (1, CURRENT_ACTION_DIM) and torch.isfinite(out).all().item()
            inference_summary = f'q_values_shape={tuple(out.shape)}'
        else:
            with torch.no_grad():
                probs, value = agent.policy(sample)
            finite = torch.isfinite(probs).all().item() and torch.isfinite(value).all().item()
            inference_ok = probs.shape == (1, CURRENT_ACTION_DIM) and value.shape == (1, 1) and finite
            inference_summary = f'policy_shape={tuple(probs.shape)}, value_shape={tuple(value.shape)}'
    except Exception as exc:
        inference_ok = False
        inference_summary = f'error={exc}'

    return strict_load_ok, inference_ok, inference_summary, load_error


def main():
    checkpoints = []
    for directory in [CURRENT_DIR, BACKUP_DIR]:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if not name.endswith('.pth'):
                continue
            if 'hybrid_' not in name:
                continue
            checkpoints.append(os.path.join(directory, name))

    # deduplicate while preserving order
    seen = set()
    unique = []
    for path in checkpoints:
        if path not in seen:
            unique.append(path)
            seen.add(path)

    entries = []
    for path in unique:
        name = os.path.basename(path)
        if 'hybrid_dqn' in name:
            algorithm = 'dqn'
        elif 'hybrid_ddqn' in name:
            algorithm = 'ddqn'
        elif 'hybrid_ppo' in name:
            algorithm = 'ppo'
        else:
            continue

        try:
            state_dict, _ = _read_state_dict(path)
            saved_state_dim, saved_action_dim, saved_arch = _infer_architecture(algorithm, state_dict)
            current_arch = _current_architecture(algorithm)
            strict_load_ok, inference_ok, inference_summary, load_error = _strict_load_and_infer(path, algorithm)
            compatibility_status = 'compatible' if strict_load_ok and inference_ok else 'incompatible'
        except Exception as exc:
            saved_state_dim = None
            saved_action_dim = None
            saved_arch = 'unknown'
            current_arch = _current_architecture(algorithm)
            strict_load_ok = False
            inference_ok = False
            inference_summary = ''
            load_error = str(exc)
            compatibility_status = 'error'

        entries.append(CheckpointInfo(
            path=path,
            algorithm=algorithm,
            saved_state_dim=saved_state_dim,
            saved_action_dim=saved_action_dim,
            saved_architecture=saved_arch,
            current_state_dim=CURRENT_STATE_DIM,
            current_action_dim=CURRENT_ACTION_DIM,
            current_architecture=current_arch,
            compatibility_status=compatibility_status,
            strict_load_ok=strict_load_ok,
            inference_ok=inference_ok,
            inference_summary=inference_summary,
            load_error=load_error,
        ))

    out = [entry.__dict__ for entry in entries]
    os.makedirs(REPORT_DIR, exist_ok=True)
    compat_path = os.path.join(REPORT_DIR, 'checkpoint_compatibility_report.json')
    verify_path = os.path.join(REPORT_DIR, 'checkpoint_loading_verification_report.json')
    with open(compat_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    with open(verify_path, 'w', encoding='utf-8') as f:
        json.dump({
            'current_state_dim': CURRENT_STATE_DIM,
            'current_action_dim': CURRENT_ACTION_DIM,
            'results': out,
        }, f, indent=2)
    print(json.dumps({'compatibility_report': compat_path, 'verification_report': verify_path, 'count': len(out)}, indent=2))


if __name__ == '__main__':
    main()
