#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Legacy helper equivalent of notebooks/helper/compare_boundary_cv.py
# with root path fixed for the report package.
def fut_zero(_: int, f16: np.ndarray) -> np.ndarray:
    return np.zeros(240, dtype="float64")


def fut_ar1(e: int, f16: np.ndarray) -> np.ndarray:
    hist = f16[: e + 1]
    hist = hist[np.isfinite(hist)]
    hist = hist[-4000:] if len(hist) > 4000 else hist
    if len(hist) < 20:
        return np.zeros(240, dtype="float64")

    denom = float(np.dot(hist[:-1], hist[:-1]))
    a = float(np.dot(hist[1:], hist[:-1]) / denom) if denom > 1e-12 else 0.0
    a = float(np.clip(a, -0.995, 0.995))

    out = np.zeros(240, dtype="float64")
    cur = float(hist[-1])
    for i in range(240):
        cur = a * cur
        out[i] = cur
    return out


def fut_ar5(e: int, f16: np.ndarray) -> np.ndarray:
    hist = f16[: e + 1]
    hist = hist[np.isfinite(hist)]
    hist = hist[-6000:] if len(hist) > 6000 else hist
    p = 5
    if len(hist) < p + 20:
        return np.zeros(240, dtype="float64")

    y = hist[p:]
    x = np.column_stack([np.ones(len(y))] + [hist[p - i - 1 : len(hist) - i - 1] for i in range(p)])
    try:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(240, dtype="float64")

    buf = list(hist[-p:])
    out = np.zeros(240, dtype="float64")
    for i in range(240):
        row = np.array([1.0] + buf[::-1], dtype="float64")
        nxt = float(row @ beta)
        nxt = float(np.clip(nxt, -0.1, 0.1))
        out[i] = nxt
        buf = buf[1:] + [nxt]
    return out


def boundary_score_for_future(tr: pd.DataFrame, f16: np.ndarray, e: int, fut: np.ndarray) -> float:
    err_short = []
    for i in range(e - 9, e + 1):
        err_short.append(abs(tr.at[i, "target_short"] - fut[i + 10 - e - 1]))

    err_medium = []
    for i in range(e - 59, e + 1):
        vals = []
        for k in range(1, 7):
            j = i + 10 * k
            vals.append(f16[j] if j <= e else fut[j - e - 1])
        pred = np.prod(1 + np.array(vals)) - 1
        err_medium.append(abs(tr.at[i, "target_medium"] - pred))

    err_long = []
    for i in range(e - 239, e + 1):
        vals = []
        for k in range(1, 25):
            j = i + 10 * k
            vals.append(f16[j] if j <= e else fut[j - e - 1])
        pred = np.prod(1 + np.array(vals)) - 1
        err_long.append(abs(tr.at[i, "target_long"] - pred))

    return 0.5 * np.mean(err_short) + 0.3 * np.mean(err_medium) + 0.2 * np.mean(err_long)


def build_eval_indices(
    tr: pd.DataFrame, f16: np.ndarray, by_day: dict[int, np.ndarray], cutoff_minute: int
) -> list[int]:
    eval_idx: list[int] = []
    for d, idx in by_day.items():
        if len(idx) <= cutoff_minute:
            continue

        e = int(idx[cutoff_minute])
        if e < 239:
            continue
        if e + 240 >= len(tr):
            continue

        pref = f16[idx[: cutoff_minute + 1]]
        if not np.all(np.isfinite(pref)):
            continue

        eval_idx.append(e)
    return eval_idx


def run_cutoff(
    tr: pd.DataFrame, f16: np.ndarray, by_day: dict[int, np.ndarray], cutoff_minute: int
) -> tuple[int, dict[str, dict[str, float]]]:
    methods = {
        "zero": fut_zero,
        "ar1": fut_ar1,
        "ar5": fut_ar5,
    }
    eval_idx = build_eval_indices(tr, f16, by_day, cutoff_minute)

    out: dict[str, dict[str, float]] = {}
    for name, fn in methods.items():
        scores = []
        for e in eval_idx:
            fut = fn(e, f16)
            s = boundary_score_for_future(tr, f16, e, fut)
            scores.append(s)
        arr = np.asarray(scores, dtype="float64")
        out[name] = {
            "mean_mae": float(arr.mean()),
            "median_mae": float(np.median(arr)),
            "std_mae": float(arr.std()),
        }
    return len(eval_idx), out


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    tr = pd.read_csv(root / "data" / "raw" / "train_v2.csv")
    f16 = tr["feature_16"].to_numpy(dtype="float64")
    date = tr["date_id"].to_numpy()
    by_day = {int(d): np.where(date == d)[0] for d in np.unique(date)}

    rows = []
    print("=== Legacy Helper Output: compare_boundary_cv ===")
    print("Comparing cutoff minute 27 vs 239 on train boundary simulation...")
    for cutoff in [27, 239]:
        n_days, result = run_cutoff(tr, f16, by_day, cutoff)
        print(f"\ncutoff={cutoff} | eval_days={n_days}")
        for method, stats in result.items():
            print(
                f"  {method.upper():<4} mean={stats['mean_mae']:.8f} "
                f"median={stats['median_mae']:.8f} std={stats['std_mae']:.8f}"
            )
            rows.append(
                {
                    "cutoff": cutoff,
                    "method": method,
                    "eval_days": n_days,
                    "mean_mae": stats["mean_mae"],
                    "median_mae": stats["median_mae"],
                    "std_mae": stats["std_mae"],
                }
            )

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="method", columns="cutoff", values="mean_mae")
    if 27 in pivot.columns and 239 in pivot.columns:
        pivot["delta_239_minus_27"] = pivot[239] - pivot[27]
    print("\nSide-by-side mean MAE:")
    print(pivot.sort_values(by=27).to_string(float_format=lambda x: f"{x:.8f}"))


if __name__ == "__main__":
    main()
