#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def compounded_target(feature_16: pd.Series, blocks: int) -> pd.Series:
    out = pd.Series(1.0, index=feature_16.index, dtype="float64")
    for k in range(1, blocks + 1):
        out *= 1.0 + feature_16.shift(-10 * k)
    return out - 1.0


def mae_with_mask(y_true: pd.Series, y_pred: pd.Series, mask: np.ndarray) -> tuple[float, float, int]:
    yt = y_true.to_numpy(dtype="float64")
    yp = y_pred.to_numpy(dtype="float64")
    valid = mask & np.isfinite(yt) & np.isfinite(yp)
    err = np.abs(yt[valid] - yp[valid])
    return float(err.mean()), float(err.max()), int(valid.sum())


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    train = pd.read_csv(root / "data/raw/train_v2.csv")
    n = len(train)

    f16 = train["feature_16"]
    pred_short = f16.shift(-10)
    pred_medium = compounded_target(f16, 6)
    pred_long = compounded_target(f16, 24)

    mask_short = np.arange(n) < (n - 10)
    mask_medium = np.arange(n) < (n - 60)
    mask_long = np.arange(n) < (n - 240)

    mae_s, max_s, cnt_s = mae_with_mask(train["target_short"], pred_short, mask_short)
    mae_m, max_m, cnt_m = mae_with_mask(train["target_medium"], pred_medium, mask_medium)
    mae_l, max_l, cnt_l = mae_with_mask(train["target_long"], pred_long, mask_long)

    print("=== Leak Formula Validation on Train ===")
    print(f"short:  valid_rows={cnt_s}, mae={mae_s:.12g}, max_abs_err={max_s:.12g}")
    print(f"medium: valid_rows={cnt_m}, mae={mae_m:.12g}, max_abs_err={max_m:.12g}")
    print(f"long:   valid_rows={cnt_l}, mae={mae_l:.12g}, max_abs_err={max_l:.12g}")

    print("\n=== Lag Sweep for feature_16 -> target_short ===")
    lag_rows: list[tuple[int, float, int]] = []
    for lag in range(1, 61):
        pred = f16.shift(-lag)
        mae, _, cnt = mae_with_mask(train["target_short"], pred, np.ones(n, dtype=bool))
        lag_rows.append((lag, mae, cnt))

    lag_rows.sort(key=lambda x: x[1])
    best_lag, best_mae, best_cnt = lag_rows[0]
    second_lag, second_mae, _ = lag_rows[1]
    print(
        f"best_lag={best_lag}, mae={best_mae:.12g}, valid_rows={best_cnt}, "
        f"gap_to_second_best={second_mae - best_mae:.12g} (second lag={second_lag})"
    )
    print("top_10_lags_by_mae:")
    for lag, mae, cnt in lag_rows[:10]:
        print(f"  lag={lag:>2d} mae={mae:.12g} valid_rows={cnt}")

    print("\n=== Candidate Single-Feature Leak Scan for target_short ===")
    maes = []
    for j in range(1, 31):
        col = f"feature_{j}"
        pred = train[col].shift(-best_lag)
        mae, _, cnt = mae_with_mask(train["target_short"], pred, np.ones(n, dtype=bool))
        maes.append((col, mae, cnt))

    maes.sort(key=lambda x: x[1])
    print(f"top_10_by_mae_shift_minus_{best_lag}:")
    for col, mae, cnt in maes[:10]:
        print(f"  {col:<10s} mae={mae:.12g} valid_rows={cnt}")


if __name__ == "__main__":
    main()
