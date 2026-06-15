from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence


def repo_root_from(file_path: str | Path) -> Path:
    return Path(file_path).resolve().parent.parent


def ensure_repo_root_on_path(file_path: str | Path) -> Path:
    import sys

    repo_root = repo_root_from(file_path)
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def resolve_first_existing(base_dir: str | Path, candidates: Sequence[str | Path]) -> Optional[Path]:
    base = Path(base_dir)
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = base / path
        if path.exists():
            return path.resolve()
    return None
