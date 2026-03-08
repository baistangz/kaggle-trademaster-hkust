#!/usr/bin/env python3
from __future__ import annotations

"""Generate macro-specialist tail submission.

Macro specialist fits a ridge model that maps previous-day aggregate features
to the next-day 240-minute `feature_16` curve, then patches unknown tail rows.
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
    for d in np.unique(train_df["date_id"].to_numpy()):
        idx = np.where(train_df["date_id"].to_numpy() == d)[0]
        if len(idx) != 240:
            continue
        vals = train_df.iloc[idx]["feature_16"].to_numpy(dtype="float64")
        if not np.all(np.isfinite(vals)):
            continue
        curves[int(d)] = vals
    return curves


def fit_macro_nextday_ridge(
    day_curves: dict[int, np.ndarray],
    ridge_lambda: float,
) -> tuple[np.ndarray, int]:
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
    return beta, x.shape[0]


def predict_future_f16_macro(
    test_df: pd.DataFrame,
    beta: np.ndarray,
    clip_abs: float,
) -> np.ndarray:
    """Predict next 240-minute curve from the last complete test day."""
    # Use the latest complete day in test as "yesterday".
    last_date = int(test_df["date_id"].iloc[-1])
    day_rows = test_df.loc[test_df["date_id"] == last_date, "feature_16"].to_numpy(dtype="float64")

    if len(day_rows) < 240:
        raise ValueError(f"Last test day {last_date} has only {len(day_rows)} rows; need 240.")
    curve = day_rows[-240:]
    if not np.all(np.isfinite(curve)):
        raise ValueError("Last test day feature_16 contains non-finite values.")

    x = day_features(curve)  # p
    pred = x @ beta  # 240
    if clip_abs > 0:
        pred = np.clip(pred, -clip_abs, clip_abs)
    return pred.astype("float64")


def patch_tail_from_future_f16(
    test_df: pd.DataFrame,
    perfect: pd.DataFrame,
    future: np.ndarray,
) -> pd.DataFrame:
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
    p = argparse.ArgumentParser(description="Generate tailvar submission with macro next-day specialist.")
    p.add_argument("--ridge-lambda", type=float, default=1.0, help="L2 regularization for ridge fit.")
    p.add_argument("--clip-abs", type=float, default=0.1, help="Clamp predicted future f16 to +/-clip-abs.")
    p.add_argument(
        "--output-name",
        type=str,
        default="",
        help="Optional output filename. If omitted, script auto-generates one in submissions/.",
    )
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

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
    beta, n_transitions = fit_macro_nextday_ridge(day_curves, ridge_lambda=args.ridge_lambda)
    future = predict_future_f16_macro(test_df, beta=beta, clip_abs=args.clip_abs)
    pred = patch_tail_from_future_f16(test_df, perfect, future=future)

    if args.output_name:
        out_path = out_dir / args.output_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        lam = f"{args.ridge_lambda:g}".replace(".", "p")
        out_path = out_dir / f"submission_TAILVAR_MACRO_SPECIALIST_LAM{lam}_{ts}.csv"

    save_submission(out_path, sample_df["id"], pred[["id", *TARGET_COLS]])

    print(f"Saved: {out_path}")
    print(f"Train day transitions used: {n_transitions}")
    print(f"Ridge lambda: {args.ridge_lambda}")
    print(f"Clip abs: {args.clip_abs}")


if __name__ == "__main__":
    main()
