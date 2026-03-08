#!/usr/bin/env python3
from __future__ import annotations

"""Synchronize repo source files into ``report_overleaf/code``."""

import shutil
from pathlib import Path

SYNC_MAP = {
    "notebooks/archival_neural_baseline.py": "report_overleaf/code/archival_neural_baseline.py",
    "notebooks/generate_zero_submission.py": "report_overleaf/code/generate_zero_submission.py",
    "notebooks/generate_tail_variants.py": "report_overleaf/code/generate_tail_variants.py",
    "notebooks/generate_tailvar_macro_specialist.py": "report_overleaf/code/generate_tailvar_macro_specialist.py",
    "notebooks/generate_tailvar_robust_anchor.py": "report_overleaf/code/generate_tailvar_robust_anchor.py",
    "notebooks/helper/run_sanity_checks.py": "report_overleaf/code/helper/run_sanity_checks.py",
    "notebooks/helper/verify_minute27_cutoff.py": "report_overleaf/code/helper/verify_minute27_cutoff.py",
    "notebooks/helper/compare_boundary_cv.py": "report_overleaf/code/helper/compare_boundary_cv.py",
    "trademaster_core/__init__.py": "report_overleaf/code/trademaster_core/__init__.py",
    "trademaster_core/constants.py": "report_overleaf/code/trademaster_core/constants.py",
    "trademaster_core/paths.py": "report_overleaf/code/trademaster_core/paths.py",
    "trademaster_core/leak_math.py": "report_overleaf/code/trademaster_core/leak_math.py",
    "trademaster_core/submission_io.py": "report_overleaf/code/trademaster_core/submission_io.py",
    "trademaster_core/tail_models.py": "report_overleaf/code/trademaster_core/tail_models.py",
}


def find_repo_root(start: Path) -> Path:
    """Walk upward from ``start`` until the project root is found."""
    here = start.resolve()
    if here.is_file():
        here = here.parent

    for candidate in [here, *here.parents]:
        if (candidate / "requirements.txt").exists() and (candidate / "data" / "raw").exists():
            return candidate

    raise FileNotFoundError(f"Could not locate repo root above: {start}")


def main() -> None:
    root = find_repo_root(Path(__file__))
    for src_rel, dst_rel in SYNC_MAP.items():
        src = root / src_rel
        dst = root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"synced {src_rel} -> {dst_rel}")


if __name__ == "__main__":
    main()
