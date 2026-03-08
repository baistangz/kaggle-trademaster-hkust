from __future__ import annotations

"""Deterministic target reconstruction and tail patching logic."""

import numpy as np
import pandas as pd

from .constants import TARGET_COLS


def compounded_target(feature_16: pd.Series, blocks: int) -> pd.Series:
    """Compute compounded return over ``blocks`` 10-minute steps."""
    out = pd.Series(1.0, index=feature_16.index, dtype="float64")
    for k in range(1, blocks + 1):
        out *= 1.0 + feature_16.shift(-10 * k)
    return out - 1.0


def build_known_region_predictions(test_df: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic predictions wherever future ``feature_16`` is visible."""
    f16 = test_df["feature_16"]
    return pd.DataFrame(
        {
            "id": test_df["id"],
            "target_short": f16.shift(-10),
            "target_medium": compounded_target(f16, 6),
            "target_long": compounded_target(f16, 24),
        }
    )


def patch_tail_from_future_f16(
    test_df: pd.DataFrame,
    known_pred: pd.DataFrame,
    future: np.ndarray,
) -> pd.DataFrame:
    """Fill unknown short/medium/long windows using a 240-step future curve."""
    out = known_pred.copy()
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


def validate_leak_mae(train_df: pd.DataFrame) -> dict[str, float]:
    """Return scored-range MAEs for the deterministic leak equations on train."""
    f16 = train_df["feature_16"]
    n = len(train_df)

    pred_short = f16.shift(-10)
    pred_medium = compounded_target(f16, 6)
    pred_long = compounded_target(f16, 24)

    mask_short = np.arange(n) < (n - 10)
    mask_medium = np.arange(n) < (n - 60)
    mask_long = np.arange(n) < (n - 240)

    return {
        "target_short": float((train_df.loc[mask_short, "target_short"] - pred_short[mask_short]).abs().mean()),
        "target_medium": float((train_df.loc[mask_medium, "target_medium"] - pred_medium[mask_medium]).abs().mean()),
        "target_long": float((train_df.loc[mask_long, "target_long"] - pred_long[mask_long]).abs().mean()),
    }


def known_region_counts(pred_df: pd.DataFrame) -> dict[str, int]:
    """Count deterministic non-NaN coverage for each target."""
    return {col: int(pred_df[col].notna().sum()) for col in TARGET_COLS}
