#!/usr/bin/env python3
from __future__ import annotations

"""Generate robust-anchor tail submission.

Robust-anchor = weighted blend of:
- macro specialist future (ridge next-day curve)
- expanding minute-mean future
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLS = ["target_short", "target_medium", "target_long"]


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


def save_submission(path: Path, base_ids: pd.Series, pred: pd.DataFrame) -> None:
    """Save predictions with required Kaggle schema/order."""
    out = pd.DataFrame({"id": base_ids})
    out = out.merge(pred[["id", *TARGET_COLS]], on="id", how="left")
    out.to_csv(path, index=False, float_format="%.18g")


def day_features(curve: np.ndarray) -> np.ndarray:
    """Extract compact day-level features for macro next-day regression."""
    o = float(curve[0])
    c = float(curve[-1])
    h = float(np.max(curve))
    l = float(np.min(curve))
    s = float(np.std(curve))
    m = float(np.mean(curve))
    am = float(np.mean(np.abs(curve)))
    r = h - l
    return np.array([1.0, o, c, h, l, s, m, am, r], dtype="float64")


def build_train_day_curves(train_df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Collect finite full-day (240-row) feature_16 curves keyed by date_id."""
    curves: dict[int, np.ndarray] = {}
    dates = train_df["date_id"].to_numpy()
    f16 = train_df["feature_16"].to_numpy(dtype="float64")
    for d in np.unique(dates):
        idx = np.where(dates == d)[0]
        if len(idx) != 240:
            continue
        vals = f16[idx]
        if not np.all(np.isfinite(vals)):
            continue
        curves[int(d)] = vals
    return curves


def fit_macro_nextday_ridge(day_curves: dict[int, np.ndarray], ridge_lambda: float) -> np.ndarray:
    """Fit ridge model mapping day features -> next-day 240-minute curve."""
    days = sorted(day_curves.keys())
    x_rows = []
    y_rows = []
    for d in days:
        nxt = d + 1
        if nxt not in day_curves:
            continue
        x_rows.append(day_features(day_curves[d]))
        y_rows.append(day_curves[nxt])

    if not x_rows:
        raise ValueError("No train day->next-day transitions found.")

    x = np.stack(x_rows, axis=0)  # n x p
    y = np.stack(y_rows, axis=0)  # n x 240
    xtx = x.T @ x
    a = xtx + ridge_lambda * np.eye(xtx.shape[0], dtype="float64")
    beta = np.linalg.solve(a, x.T @ y)  # p x 240
    return beta


def macro_future_from_test(test_df: pd.DataFrame, beta: np.ndarray, clip_abs: float) -> np.ndarray:
    """Predict next 240-minute curve from the last complete test day."""
    last_date = int(test_df["date_id"].iloc[-1])
    curve = test_df.loc[test_df["date_id"] == last_date, "feature_16"].to_numpy(dtype="float64")
    if len(curve) < 240:
        raise ValueError(f"Last test day {last_date} has only {len(curve)} rows; need 240.")
    curve = curve[-240:]
    if not np.all(np.isfinite(curve)):
        raise ValueError("Last test day feature_16 has non-finite values.")

    pred = day_features(curve) @ beta
    if clip_abs > 0:
        pred = np.clip(pred, -clip_abs, clip_abs)
    return pred.astype("float64")


def expanding_future_from_train(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Build 240-minute future using train minute-of-day means."""
    minute_mean = (
        train_df.groupby("minute_id")["feature_16"].mean().reindex(range(240)).fillna(0.0).to_numpy(dtype="float64")
    )
    last_minute = int(test_df["minute_id"].iloc[-1])
    fut_minutes = (last_minute + np.arange(1, 241)) % 240
    return minute_mean[fut_minutes].astype("float64")


def patch_tail_from_future_f16(test_df: pd.DataFrame, perfect: pd.DataFrame, future: np.ndarray) -> pd.DataFrame:
    """Fill unknown short/medium/long windows using provided future curve."""
    out = perfect.copy()
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Generate robust-anchor tailvar submission (macro+expanding blend).")
    p.add_argument("--macro-weight", type=float, default=0.75, help="Weight on macro future (0..1).")
    p.add_argument("--ridge-lambda", type=float, default=0.1, help="Ridge lambda for macro specialist.")
    p.add_argument("--clip-abs", type=float, default=0.1, help="Clamp macro future to +/- clip_abs.")
    p.add_argument("--output-name", type=str, default="", help="Optional output filename.")
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    if not (0.0 <= args.macro_weight <= 1.0):
        raise ValueError("--macro-weight must be between 0 and 1")

    root = Path(__file__).resolve().parents[1]
    train_path = root / "data" / "raw" / "train_v2.csv"
    test_path = root / "data" / "raw" / "test_v2.csv"
    sample_path = root / "data" / "raw" / "sample_submission.csv"
    out_dir = root / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path, usecols=["date_id", "minute_id", "feature_16"])
    test_df = pd.read_csv(test_path, usecols=["id", "date_id", "minute_id", "feature_16"])
    sample_df = pd.read_csv(sample_path, usecols=["id"])

    perfect = base_perfect(test_df)
    day_curves = build_train_day_curves(train_df)
    beta = fit_macro_nextday_ridge(day_curves, ridge_lambda=args.ridge_lambda)
    fut_macro = macro_future_from_test(test_df, beta=beta, clip_abs=args.clip_abs)
    fut_expand = expanding_future_from_train(train_df, test_df)

    # Robust-anchor blend (requested):
    # future = w * macro + (1-w) * expanding_all
    w = float(args.macro_weight)
    future = w * fut_macro + (1.0 - w) * fut_expand

    pred = patch_tail_from_future_f16(test_df, perfect, future=future)

    if args.output_name:
        out_path = out_dir / args.output_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        w_tag = str(args.macro_weight).replace(".", "p")
        lam_tag = str(args.ridge_lambda).replace(".", "p")
        out_path = out_dir / f"submission_TAILVAR_ROBUST_ANCHOR_W{w_tag}_LAM{lam_tag}_{ts}.csv"

    save_submission(out_path, sample_df["id"], pred[["id", *TARGET_COLS]])

    print(f"Saved: {out_path}")
    print(f"macro_weight={w:.6f}")
    print(f"expand_weight={1.0 - w:.6f}")
    print(f"weight_sum={(w + (1.0 - w)):.6f}")
    print(f"ridge_lambda={args.ridge_lambda}")
    print(f"clip_abs={args.clip_abs}")
    print(f"future_range=[{future.min():.8f}, {future.max():.8f}]")


if __name__ == "__main__":
    main()
