"""Working-tree provenance (git revision) for run metadata."""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["git_revision"]


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def git_revision(*, short: bool = True, include_dirty: bool = True) -> str | None:
    """Return the current commit hash (``-dirty`` suffixed if tracked files changed).

    Returns ``None`` when git is unavailable or the repository has no commits, so
    callers can record provenance without failing a run.
    """
    arguments = ["git", "rev-parse"] + (["--short", "HEAD"] if short else ["HEAD"])
    try:
        revision = subprocess.run(
            arguments,
            cwd=_repository_root(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return None
    if revision.returncode != 0 or not revision.stdout.strip():
        return None

    value = revision.stdout.strip()
    if include_dirty:
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=_repository_root(),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
            return value
        if status.returncode == 0 and status.stdout.strip():
            value += "-dirty"
    return value
