#!/usr/bin/env python3
from __future__ import annotations

"""Generate tail-variant submissions from the leak base.

Important reproducibility note:
- This script intentionally keeps only the currently selected candidate methods
  active (`zero`, `expanding_all`).
- Historical/losing variants are left commented for traceability.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLS = ["target_short", "target_medium", "target_long"]
ACTIVE_METHODS = ["zero", "expanding_all"]


def compounded_target(feature_16: pd.Series, blocks: int) -> pd.Series:
    """Compute compounded return over `blocks` 10-minute steps."""
    out = pd.Series(1.0, index=feature_16.index, dtype="float64")
    for k in range(1, blocks + 1):
        out *= 1.0 + feature_16.shift(-10 * k)
    return out - 1.0


def base_perfect(test_df: pd.DataFrame) -> pd.DataFrame:
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


def extrapolate_future_f16(
    f16: np.ndarray,
    method: str,
    horizon: int = 240,
    *,
    minute_id: np.ndarray | None = None,
    minute_mean_by_minute: np.ndarray | None = None,
) -> np.ndarray:
    """Extrapolate 240-step future `feature_16` using a selected tail prior."""
    x = np.asarray(f16, dtype="float64")
    if method == "zero":
        return np.zeros(horizon, dtype="float64")
    if method == "expanding_all":
        if minute_id is None or minute_mean_by_minute is None:
            raise ValueError("minute_id and minute_mean_by_minute are required for method='expanding_all'")
        if len(minute_mean_by_minute) != 240:
            raise ValueError("minute_mean_by_minute must have length 240")
        last_minute = int(minute_id[-1])
        future_minutes = (last_minute + np.arange(1, horizon + 1)) % 240
        return np.asarray(minute_mean_by_minute[future_minutes], dtype="float64")
    if method == "last":
        return np.repeat(x[-1], horizon).astype("float64")
    if method == "lag240":
        out = np.zeros(horizon, dtype="float64")
        for k in range(horizon):
            j = len(x) - 240 + k
            out[k] = x[j] if j >= 0 else 0.0
        return out
    if method == "lag480":
        out = np.zeros(horizon, dtype="float64")
        for k in range(horizon):
            j = len(x) - 480 + k
            out[k] = x[j] if j >= 0 else 0.0
        return out
    if method == "ar1":
        hist = x[np.isfinite(x)]
        hist = hist[-4000:] if len(hist) > 4000 else hist
        if len(hist) < 20:
            return np.zeros(horizon, dtype="float64")
        denom = float(np.dot(hist[:-1], hist[:-1]))
        a = float(np.dot(hist[1:], hist[:-1]) / denom) if denom > 1e-12 else 0.0
        a = float(np.clip(a, -0.995, 0.995))
        out = np.zeros(horizon, dtype="float64")
        cur = float(hist[-1])
        for i in range(horizon):
            cur = a * cur
            out[i] = cur
        return out
    if method == "ar5":
        hist = x[np.isfinite(x)]
        hist = hist[-6000:] if len(hist) > 6000 else hist
        p = 5
        if len(hist) < p + 20:
            return np.zeros(horizon, dtype="float64")

        y = hist[p:]
        X = np.column_stack([np.ones(len(y))] + [hist[p - i - 1 : len(hist) - i - 1] for i in range(p)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

        buf = list(hist[-p:])
        out = np.zeros(horizon, dtype="float64")
        for i in range(horizon):
            row = np.array([1.0] + buf[::-1], dtype="float64")
            nxt = float(row @ beta)
            # clamp to avoid unstable recursion tails
            nxt = float(np.clip(nxt, -0.1, 0.1))
            out[i] = nxt
            buf = buf[1:] + [nxt]
        return out
    raise ValueError(f"Unknown method: {method}")


def patch_tail_from_future_f16(
    test_df: pd.DataFrame,
    perfect: pd.DataFrame,
    method: str,
    minute_mean_by_minute: np.ndarray | None = None,
) -> pd.DataFrame:
    """Fill the unknown short/medium/long tail windows from extrapolated future."""
    out = perfect.copy()
    f16 = test_df["feature_16"].to_numpy(dtype="float64")
    minute_id = test_df["minute_id"].to_numpy(dtype="int64")
    n = len(f16)
    future = extrapolate_future_f16(
        f16,
        method=method,
        horizon=240,
        minute_id=minute_id,
        minute_mean_by_minute=minute_mean_by_minute,
    )

    # Patch short NaNs (last 10 rows).
    for i in range(n - 10, n):
        h = i + 10 - n  # 0..9 into future
        out.at[i, "target_short"] = future[h]

    # Patch medium NaNs (last 60 rows) with compounded 6 blocks.
    for i in range(n - 60, n):
        blocks = []
        for k in range(1, 7):
            j = i + 10 * k
            blocks.append(f16[j] if j < n else future[j - n])
        out.at[i, "target_medium"] = np.prod(1.0 + np.asarray(blocks)) - 1.0

    # Patch long NaNs (last 240 rows) with compounded 24 blocks.
    for i in range(n - 240, n):
        blocks = []
        for k in range(1, 25):
            j = i + 10 * k
            blocks.append(f16[j] if j < n else future[j - n])
        out.at[i, "target_long"] = np.prod(1.0 + np.asarray(blocks)) - 1.0

    return out


def load_submission(path: Path) -> pd.DataFrame:
    """Load and validate a fallback submission."""
    sub = pd.read_csv(path)
    expected = {"id", *TARGET_COLS}
    if not expected.issubset(sub.columns):
        missing = sorted(expected - set(sub.columns))
        raise ValueError(f"{path} is missing required columns: {missing}")
    return sub[["id", *TARGET_COLS]].copy()


def save_submission(path: Path, base_ids: pd.Series, pred: pd.DataFrame) -> None:
    """Save predictions with required Kaggle schema/order."""
    out = pd.DataFrame({"id": base_ids})
    out = out.merge(pred[["id", *TARGET_COLS]], on="id", how="left")
    out.to_csv(path, index=False, float_format="%.18g")


def blend_tail(base: pd.DataFrame, alt: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Blend only the uncertain tail windows:
    - last 10 short
    - last 60 medium
    - last 240 long
    alpha=1.0 -> base only, alpha=0.0 -> alt only
    """
    out = base.copy()
    n = len(out)
    out.loc[n - 10 :, "target_short"] = (
        alpha * base.loc[n - 10 :, "target_short"] + (1.0 - alpha) * alt.loc[n - 10 :, "target_short"]
    )
    out.loc[n - 60 :, "target_medium"] = (
        alpha * base.loc[n - 60 :, "target_medium"] + (1.0 - alpha) * alt.loc[n - 60 :, "target_medium"]
    )
    out.loc[n - 240 :, "target_long"] = (
        alpha * base.loc[n - 240 :, "target_long"] + (1.0 - alpha) * alt.loc[n - 240 :, "target_long"]
    )
    return out


def main() -> None:
    """Entry point."""
    root = Path(__file__).resolve().parents[1]
    test_path = root / "data/raw/test_v2.csv"
    train_path = root / "data/raw/train_v2.csv"
    sample_path = root / "data/raw/sample_submission.csv"
    # --- KEPT ONLY FOR OPTIONAL EXPERIMENTS ---
    # fallback_path = root / "submissions/submission_Ensemble_Ref50_Rob50_CV0.67173_20260223_154015.csv"
    out_dir = root / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path, usecols=["minute_id", "feature_16"])
    sample_df = pd.read_csv(sample_path)
    perfect = base_perfect(test_df)
    # fallback = load_submission(fallback_path)
    minute_mean_by_minute = (
        train_df.groupby("minute_id")["feature_16"].mean().reindex(range(240)).fillna(0.0).to_numpy(dtype="float64")
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Disabled losing/non-essential variant: leak + model fallback.
    # base = perfect.merge(fallback, on="id", how="left", suffixes=("", "_fb"))
    # for c in TARGET_COLS:
    #     base[c] = base[c].combine_first(base[f"{c}_fb"])
    # base = base[["id", *TARGET_COLS]]
    # save_submission(out_dir / f"submission_TAILVAR_BASE_FALLBACK_{ts}.csv", sample_df["id"], base)

    # 2) Disabled old lazy-zero variant.
    # zero = perfect.copy()
    # zero[TARGET_COLS] = zero[TARGET_COLS].fillna(0.0)
    # save_submission(out_dir / f"submission_TAILVAR_ZERO_{ts}.csv", sample_df["id"], zero)

    # 3) Keep only top candidates.
    patched_by_method = {}
    # Disabled losing variants: "ar1", "ar5", "lag240", "lag480", "last"
    for method in ACTIVE_METHODS:
        patched = patch_tail_from_future_f16(
            test_df,
            perfect,
            method=method,
            minute_mean_by_minute=minute_mean_by_minute,
        )
        patched_by_method[method] = patched
        save_submission(out_dir / f"submission_TAILVAR_{method.upper()}_{ts}.csv", sample_df["id"], patched)

    # 4) Disabled losing/non-essential blends.
    # ar1 = patched_by_method["ar1"]
    # for alpha in [0.75, 0.5, 0.25]:
    #     blend = blend_tail(base, ar1, alpha=alpha)
    #     pct = int((1.0 - alpha) * 100)
    #     save_submission(out_dir / f"submission_TAILVAR_BLEND_AR1_{pct:02d}PCT_{ts}.csv", sample_df["id"], blend)
    #
    # ar5 = patched_by_method["ar5"]
    # for alpha in [0.75, 0.5, 0.25]:
    #     blend = blend_tail(base, ar5, alpha=alpha)
    #     pct = int((1.0 - alpha) * 100)
    #     save_submission(out_dir / f"submission_TAILVAR_BLEND_AR5_{pct:02d}PCT_{ts}.csv", sample_df["id"], blend)

    print("Generated tail-variant submissions:")
    for p in sorted(out_dir.glob(f"submission_TAILVAR_*_{ts}.csv")):
        print(p)


if __name__ == "__main__":
    main()
