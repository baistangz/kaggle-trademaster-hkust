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

import numpy as np
import pandas as pd

TARGET_COLS = ["target_short", "target_medium", "target_long"]
ACTIVE_METHODS = ("zero", "expanding_all")
SUBMISSION_NAMING_CONVENTION = "submission_<PIPELINE>_<VARIANT>_CV<LOCAL_CV>.csv"


def compounded_target(feature_16: pd.Series, blocks: int) -> pd.Series:
    """Compute compounded return over `blocks` 10-minute steps."""
    out = pd.Series(1.0, index=feature_16.index, dtype="float64")
    for k in range(1, blocks + 1):
        out *= 1.0 + feature_16.shift(-10 * k)
    return out - 1.0


def build_leak_base(test_df: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic predictions where leak coverage exists."""
    f16 = test_df["feature_16"]
    return pd.DataFrame(
        {
            "id": test_df["id"],
            "target_short": f16.shift(-10),
            "target_medium": compounded_target(f16, 6),
            "target_long": compounded_target(f16, 24),
        }
    )


def compute_minute_mean_curve(train_df: pd.DataFrame) -> np.ndarray:
    """Compute global minute-of-day mean curve (length 240)."""
    return (
        train_df.groupby("minute_id")["feature_16"]
        .mean()
        .reindex(range(240))
        .fillna(0.0)
        .to_numpy(dtype="float64")
    )


def extrapolate_future_f16(
    f16: np.ndarray,
    minute_id: np.ndarray,
    method: str,
    minute_mean_curve: np.ndarray | None = None,
    horizon: int = 240,
) -> np.ndarray:
    """Build 240-step future `feature_16` under the selected tail prior."""
    if method == "zero":
        return np.zeros(horizon, dtype="float64")

    if method == "expanding_all":
        if minute_mean_curve is None:
            raise ValueError("minute_mean_curve is required for method='expanding_all'")
        if len(minute_mean_curve) != 240:
            raise ValueError("minute_mean_curve must have length 240")
        last_minute = int(minute_id[-1])
        future_minutes = (last_minute + np.arange(1, horizon + 1)) % 240
        return np.asarray(minute_mean_curve[future_minutes], dtype="float64")

    raise ValueError(f"Unknown method: {method}")


def patch_tail_from_future_f16(
    test_df: pd.DataFrame,
    leak_base: pd.DataFrame,
    future: np.ndarray,
) -> pd.DataFrame:
    """Fill unknown short/medium/long tail windows using provided future curve."""
    out = leak_base.copy()
    f16 = test_df["feature_16"].to_numpy(dtype="float64")
    n = len(f16)

    if len(future) != 240:
        raise ValueError("future must have length 240")

    for i in range(n - 10, n):
        h = i + 10 - n
        out.at[i, "target_short"] = future[h]

    for i in range(n - 60, n):
        blocks = []
        for k in range(1, 7):
            j = i + 10 * k
            blocks.append(f16[j] if j < n else future[j - n])
        out.at[i, "target_medium"] = np.prod(1.0 + np.asarray(blocks)) - 1.0

    for i in range(n - 240, n):
        blocks = []
        for k in range(1, 25):
            j = i + 10 * k
            blocks.append(f16[j] if j < n else future[j - n])
        out.at[i, "target_long"] = np.prod(1.0 + np.asarray(blocks)) - 1.0

    return out


def save_submission(path: Path, sample_ids: pd.Series, pred: pd.DataFrame) -> None:
    """Save predictions with required Kaggle schema/order."""
    out = pd.DataFrame({"id": sample_ids})
    out = out.merge(pred[["id", *TARGET_COLS]], on="id", how="left")
    out.to_csv(path, index=False, float_format="%.18g")


def main() -> None:
    """Entry point."""
    root = Path(__file__).resolve().parents[1]
    test_path = root / "data" / "raw" / "test_v2.csv"
    train_path = root / "data" / "raw" / "train_v2.csv"
    sample_path = root / "data" / "raw" / "sample_submission.csv"
    out_dir = root / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(test_path, usecols=["id", "minute_id", "feature_16"])
    train_df = pd.read_csv(train_path, usecols=["minute_id", "feature_16"])
    sample_df = pd.read_csv(sample_path, usecols=["id"])

    leak_base = build_leak_base(test_df)
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
        pred = patch_tail_from_future_f16(test_df=test_df, leak_base=leak_base, future=future)
        out_path = out_dir / f"submission_TAILVAR_{method.upper()}_{ts}.csv"
        save_submission(out_path, sample_df["id"], pred)
        generated.append(out_path)

    print("Generated tail-variant submissions:")
    for p in generated:
        print(p)
    print(f"Naming convention (docs/examples): {SUBMISSION_NAMING_CONVENTION}")


if __name__ == "__main__":
    main()
