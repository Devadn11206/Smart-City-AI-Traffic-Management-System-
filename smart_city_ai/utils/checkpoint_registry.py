from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch

from .reliability import extract_state_dict, load_torch_checkpoint, save_torch_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

CANONICAL_ROOT = WORKSPACE_ROOT / 'results' / 'checkpoints'
ACTIVE_DIR = CANONICAL_ROOT / 'active'
ARCHIVE_DIR = CANONICAL_ROOT / 'archive'
REGISTRY_PATH = CANONICAL_ROOT / 'registry.json'
HEALTH_REPORT_PATH = CANONICAL_ROOT / 'checkpoint_health_report.json'

ALGORITHMS = ('dqn', 'ddqn', 'ppo')
EXPECTED_STATE_DIMS = {'dqn': 9, 'ddqn': 9, 'ppo': 9}
EXPECTED_ACTION_DIMS = {'dqn': 4, 'ddqn': 4, 'ppo': 4}


@dataclass
class CheckpointRecord:
    algorithm: str
    path: str
    active_path: str
    archived: bool
    saved_state_dim: Optional[int]
    saved_action_dim: Optional[int]
    model_version: str
    training_date: str
    git_hash: str
    benchmark_score: Optional[float]
    status: str
    reason: str = ''


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or 'unknown'
    except Exception:
        pass
    return 'unknown'


def build_checkpoint_metadata(
    *,
    algorithm: str,
    state_dim: int,
    action_dim: int,
    model_version: str = 'hybrid-v2',
    benchmark_score: float | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = {
        'agent': algorithm,
        'algorithm': algorithm,
        'state_dim': int(state_dim),
        'action_dim': int(action_dim),
        'model_version': model_version,
        'training_date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'git_hash': _git_hash(),
        'benchmark_score': benchmark_score,
        'version': model_version,
    }
    if extra:
        payload.update(extra)
    return payload


def _ensure_dirs() -> None:
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    CANONICAL_ROOT.mkdir(parents=True, exist_ok=True)


def _load_metadata(path: Path) -> Dict[str, Any]:
    checkpoint = load_torch_checkpoint(path, map_location='cpu')
    if isinstance(checkpoint, dict):
        meta = {k: checkpoint.get(k) for k in ('agent', 'algorithm', 'state_dim', 'action_dim', 'model_version', 'training_date', 'git_hash', 'benchmark_score', 'version') if k in checkpoint}
        return meta
    return {}


def _infer_dims(path: Path) -> tuple[Optional[int], Optional[int], str, Dict[str, Any]]:
    checkpoint = load_torch_checkpoint(path, map_location='cpu')
    state_dict = extract_state_dict(checkpoint)
    saved_state_dim = None
    saved_action_dim = None
    arch = 'unknown'
    if 'fc1.weight' in state_dict and 'fc3.weight' in state_dict:
        saved_state_dim = int(state_dict['fc1.weight'].shape[1])
        saved_action_dim = int(state_dict['fc3.weight'].shape[0])
        arch = f'QNetwork({saved_state_dim} -> 64 -> 64 -> {saved_action_dim})'
    elif 'actor.0.weight' in state_dict and 'actor.4.weight' in state_dict:
        saved_state_dim = int(state_dict['actor.0.weight'].shape[1])
        saved_action_dim = int(state_dict['actor.4.weight'].shape[0])
        arch = f'ActorCritic(actor: {saved_state_dim} -> 64 -> 64 -> {saved_action_dim}, critic: {saved_state_dim} -> 64 -> 64 -> 1)'
    return saved_state_dim, saved_action_dim, arch, checkpoint if isinstance(checkpoint, dict) else {}


def _copy_to_active(src: Path, algorithm: str) -> Path:
    target = ACTIVE_DIR / f'{algorithm}.pth'
    shutil.copy2(src, target)
    return target


def _candidate_paths(algorithm: str) -> list[Path]:
    roots = [
        WORKSPACE_ROOT / 'results' / 'checkpoint_backups' / '20260530_203534',
        PROJECT_ROOT / 'results' / 'hybrid_results',
        WORKSPACE_ROOT / 'results' / 'hybrid_results',
    ]
    names = [f'hybrid_{algorithm}_model_best.pth', f'hybrid_{algorithm}_model_final.pth', f'hybrid_{algorithm}_model.pth']
    candidates: list[Path] = []
    for root in roots:
        for name in names:
            candidates.append(root / name)
    return candidates


def validate_checkpoint(path: str | Path, algorithm: str, expected_state_dim: int, expected_action_dim: int) -> Dict[str, Any]:
    path = Path(path)
    result: Dict[str, Any] = {
        'path': str(path),
        'algorithm': algorithm,
        'expected_state_dim': int(expected_state_dim),
        'expected_action_dim': int(expected_action_dim),
        'strict_load_ok': False,
        'compatible': False,
        'reason': '',
    }
    try:
        saved_state_dim, saved_action_dim, arch, checkpoint = _infer_dims(path)
        meta = _load_metadata(path)
        result.update({
            'saved_state_dim': saved_state_dim,
            'saved_action_dim': saved_action_dim,
            'saved_architecture': arch,
            'metadata': meta,
        })
        if saved_state_dim not in (None, expected_state_dim) or saved_action_dim not in (None, expected_action_dim):
            result['reason'] = 'dimension mismatch'
            return result
        if meta.get('algorithm') not in (None, algorithm):
            result['reason'] = 'algorithm mismatch'
            return result
        if meta.get('state_dim') not in (None, expected_state_dim):
            result['reason'] = 'metadata state_dim mismatch'
            return result
        if meta.get('action_dim') not in (None, expected_action_dim):
            result['reason'] = 'metadata action_dim mismatch'
            return result
        result['strict_load_ok'] = True
        result['compatible'] = True
        result['reason'] = 'ok'
        return result
    except Exception as exc:
        result['reason'] = str(exc)
        return result


def ensure_canonical_checkpoints(force: bool = False) -> Dict[str, Dict[str, Any]]:
    _ensure_dirs()
    registry: Dict[str, Dict[str, Any]] = {}
    for algorithm in ALGORITHMS:
        active_path = ACTIVE_DIR / f'{algorithm}.pth'
        if active_path.exists() and not force:
            registry[algorithm] = validate_checkpoint(active_path, algorithm, EXPECTED_STATE_DIMS[algorithm], EXPECTED_ACTION_DIMS[algorithm])
            registry[algorithm]['active_path'] = str(active_path)
            continue

        selected = None
        for candidate in _candidate_paths(algorithm):
            if not candidate.exists():
                continue
            validation = validate_checkpoint(candidate, algorithm, EXPECTED_STATE_DIMS[algorithm], EXPECTED_ACTION_DIMS[algorithm])
            if validation.get('compatible'):
                selected = candidate
                registry[algorithm] = validation
                registry[algorithm]['source_path'] = str(candidate)
                registry[algorithm]['active_path'] = str(active_path)
                _copy_to_active(candidate, algorithm)
                break
        if selected is None:
            registry[algorithm] = {
                'algorithm': algorithm,
                'compatible': False,
                'strict_load_ok': False,
                'reason': 'no compatible checkpoint found',
                'active_path': str(active_path),
            }
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding='utf-8')
    return registry


def resolve_active_checkpoint(algorithm: str) -> Path:
    _ensure_dirs()
    active_path = ACTIVE_DIR / f'{algorithm}.pth'
    if active_path.exists():
        validation = validate_checkpoint(active_path, algorithm, EXPECTED_STATE_DIMS[algorithm], EXPECTED_ACTION_DIMS[algorithm])
        if validation.get('compatible'):
            return active_path
    registry = ensure_canonical_checkpoints()
    info = registry.get(algorithm, {})
    active = info.get('active_path')
    if active:
        return Path(active)
    raise FileNotFoundError(f'No compatible active checkpoint found for {algorithm}')


def archive_incompatible_checkpoints() -> list[Dict[str, Any]]:
    _ensure_dirs()
    archive_run_dir = ARCHIVE_DIR / time.strftime('%Y%m%d_%H%M%S')
    archive_run_dir.mkdir(parents=True, exist_ok=True)
    records: list[Dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        for candidate in _candidate_paths(algorithm):
            if not candidate.exists():
                continue
            validation = validate_checkpoint(candidate, algorithm, EXPECTED_STATE_DIMS[algorithm], EXPECTED_ACTION_DIMS[algorithm])
            if validation.get('compatible'):
                continue
            archived_path = archive_run_dir / candidate.name
            shutil.copy2(candidate, archived_path)
            validation['archived_to'] = str(archived_path)
            records.append(validation)
    return records


def generate_checkpoint_health_report() -> Dict[str, Any]:
    _ensure_dirs()
    registry = ensure_canonical_checkpoints()
    archive_records = archive_incompatible_checkpoints()
    report = {
        'canonical_root': str(CANONICAL_ROOT),
        'active_dir': str(ACTIVE_DIR),
        'archive_dir': str(ARCHIVE_DIR),
        'registry': registry,
        'archived_incompatible': archive_records,
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    HEALTH_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def load_active_checkpoint(algorithm: str, map_location: str | torch.device = 'cpu') -> tuple[Any, Path]:
    path = resolve_active_checkpoint(algorithm)
    checkpoint = load_torch_checkpoint(path, map_location=map_location)
    return checkpoint, path
