#!/usr/bin/env python3
from __future__ import annotations

"""Generate active tail-variant submissions from deterministic leak base.

This script intentionally generates only the currently active variants:
- `zero`
- `expanding_all`

Submission naming convention for manual output names in docs/comments:
`submission_<PIPELINE>_<VARIANT>_CV<LOCAL_CV>.csv`
"""

from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trademaster_core.constants import SUBMISSION_NAMING_CONVENTION
from trademaster_core.leak_math import build_known_region_predictions, patch_tail_from_future_f16
from trademaster_core.submission_io import save_submission
from trademaster_core.tail_models import compute_minute_mean_curve, future_from_minute_mean, future_zero

ACTIVE_METHODS = ("zero", "expanding_all")


def extrapolate_future_f16(
    f16: np.ndarray,
    minute_id: np.ndarray,
    method: str,
    minute_mean_curve: np.ndarray | None = None,
    horizon: int = 240,
) -> np.ndarray:
    """Build 240-step future `feature_16` under the selected tail prior."""
    if method == "zero":
        return future_zero(horizon)

    if method == "expanding_all":
        if minute_mean_curve is None:
            raise ValueError("minute_mean_curve is required for method='expanding_all'")
        last_minute = int(minute_id[-1])
        return future_from_minute_mean(last_minute, minute_mean_curve, horizon)

    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    """Entry point."""
    root = REPO_ROOT
    test_path = root / "data" / "raw" / "test_v2.csv"
    train_path = root / "data" / "raw" / "train_v2.csv"
    sample_path = root / "data" / "raw" / "sample_submission.csv"
    out_dir = root / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(test_path, usecols=["id", "minute_id", "feature_16"])
    train_df = pd.read_csv(train_path, usecols=["minute_id", "feature_16"])
    sample_df = pd.read_csv(sample_path, usecols=["id"])

    leak_base = build_known_region_predictions(test_df)
    minute_mean_curve = compute_minute_mean_curve(train_df)
    f16 = test_df["feature_16"].to_numpy(dtype="float64")
    minute_id = test_df["minute_id"].to_numpy(dtype="int64")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated: list[Path] = []

    for method in ACTIVE_METHODS:
        future = extrapolate_future_f16(
            f16=f16,
            minute_id=minute_id,
            method=method,
            minute_mean_curve=minute_mean_curve,
        )
        pred = patch_tail_from_future_f16(test_df=test_df, known_pred=leak_base, future=future)
        out_path = out_dir / f"submission_TAILVAR_{method.upper()}_{ts}.csv"
        save_submission(out_path, sample_df["id"], pred)
        generated.append(out_path)

    print("Generated tail-variant submissions:")
    for p in generated:
        print(p)
    print(f"Naming convention (docs/examples): {SUBMISSION_NAMING_CONVENTION}")


if __name__ == "__main__":
    main()
