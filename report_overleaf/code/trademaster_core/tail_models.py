from __future__ import annotations

"""Tail-future priors shared across submission generators."""

import numpy as np
import pandas as pd


def compute_minute_mean_curve(train_df: pd.DataFrame) -> np.ndarray:
    """Compute the 240-minute global mean curve from train ``feature_16``."""
    return (
        train_df.groupby("minute_id")["feature_16"]
        .mean()
        .reindex(range(240))
        .fillna(0.0)
        .to_numpy(dtype="float64")
    )


def future_zero(horizon: int = 240) -> np.ndarray:
    """Return a hard-zero future prior."""
    return np.zeros(horizon, dtype="float64")


def future_from_minute_mean(last_minute: int, minute_mean_curve: np.ndarray, horizon: int = 240) -> np.ndarray:
    """Build future values using the minute-of-day mean curve."""
    if len(minute_mean_curve) != 240:
        raise ValueError("minute_mean_curve must have length 240")
    future_minutes = (last_minute + np.arange(1, horizon + 1)) % 240
    return np.asarray(minute_mean_curve[future_minutes], dtype="float64")


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
    """Collect finite full-day ``feature_16`` curves keyed by ``date_id``."""
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


def fit_macro_nextday_ridge(
    day_curves: dict[int, np.ndarray],
    ridge_lambda: float,
) -> tuple[np.ndarray, int]:
    """Fit ridge model mapping day features to the next-day 240-minute curve."""
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

    x = np.stack(x_rows, axis=0)
    y = np.stack(y_rows, axis=0)
    xtx = x.T @ x
    a = xtx + ridge_lambda * np.eye(xtx.shape[0], dtype="float64")
    beta = np.linalg.solve(a, x.T @ y)
    return beta, x.shape[0]


def predict_future_f16_macro(test_df: pd.DataFrame, beta: np.ndarray, clip_abs: float) -> np.ndarray:
    """Predict the next 240-minute curve from the last complete test day."""
    last_date = int(test_df["date_id"].iloc[-1])
    curve = test_df.loc[test_df["date_id"] == last_date, "feature_16"].to_numpy(dtype="float64")
    if len(curve) < 240:
        raise ValueError(f"Last test day {last_date} has only {len(curve)} rows; need 240.")
    curve = curve[-240:]
    if not np.all(np.isfinite(curve)):
        raise ValueError("Last test day feature_16 contains non-finite values.")

    pred = day_features(curve) @ beta
    if clip_abs > 0:
        pred = np.clip(pred, -clip_abs, clip_abs)
    return pred.astype("float64")
