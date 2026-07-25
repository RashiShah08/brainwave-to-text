"""Filesystem layout, resolved relative to the repository root.

Every path in the project is derived from :func:`repo_root`. Nothing is
hardcoded to a developer's home directory, and every location can be
overridden with an environment variable so the same code runs unchanged in a
container, in CI, or on a workstation.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_PREFIX = "BWT_"


def repo_root() -> Path:
    """Return the project root.

    Resolution order:
      1. ``$BWT_ROOT`` if set.
      2. The directory containing this file, walked up until a marker is found
         (``pyproject.toml``), which makes editable installs and direct source
         checkouts behave identically.
    """
    env = os.environ.get(f"{_ENV_PREFIX}ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    # Installed as a wheel with no source tree: fall back to the CWD.
    return Path.cwd().resolve()


def _sub(name: str, default: str) -> Path:
    env = os.environ.get(f"{_ENV_PREFIX}{name.upper()}")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / default


def raw_data_dir() -> Path:
    """PhysioNet EEGMMIDB EDF files, one directory per subject (``S001``...)."""
    return _sub("raw_data", "raw_data")


def cache_dir() -> Path:
    """Epoched datasets, keyed by task + preprocessing parameters."""
    return _sub("cache", "cache")


def artifacts_dir() -> Path:
    """Trained pipelines and their model cards."""
    return _sub("artifacts", "artifacts")


def reports_dir() -> Path:
    """Evaluation reports, figures, and confusion matrices."""
    return _sub("reports", "reports")


def upload_dir() -> Path:
    """Scratch space for files received by the web service."""
    return _sub("uploads", "uploads")


def configs_dir() -> Path:
    return repo_root() / "configs"


def ensure_dirs() -> None:
    """Create the writable directories. Safe to call repeatedly."""
    for d in (cache_dir(), artifacts_dir(), reports_dir(), upload_dir()):
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "repo_root",
    "raw_data_dir",
    "cache_dir",
    "artifacts_dir",
    "reports_dir",
    "upload_dir",
    "configs_dir",
    "ensure_dirs",
]
