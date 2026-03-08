from __future__ import annotations

"""Robust repository-root discovery."""

from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """Walk upward from ``start`` until the project root is found."""
    here = start.resolve()
    if here.is_file():
        here = here.parent

    for candidate in [here, *here.parents]:
        if (candidate / "requirements.txt").exists() and (candidate / "data" / "raw").exists():
            return candidate

    raise FileNotFoundError(f"Could not locate repo root above: {start}")
