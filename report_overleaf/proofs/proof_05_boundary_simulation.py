#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Env:
    tr: pd.DataFrame
    f16: np.ndarray
    minute: np.ndarray
    date: np.ndarray
    by_day: dict[int, np.ndarray]
    day_curves: dict[int, np.ndarray]


def load_env() -> Env:
    root = Path(__file__).resolve().parents[2]
    tr = pd.read_csv(root / "data/raw/train_v2.csv")
    f16 = tr["feature_16"].to_numpy(dtype="float64")
    minute = tr["minute_id"].to_numpy(dtype="int64")
    date = tr["date_id"].to_numpy(dtype="int64")
    by_day = {int(d): np.where(date == d)[0] for d in np.unique(date)}

    day_curves: dict[int, np.ndarray] = {}
    for d, idx in by_day.items():
        if len(idx) != 240:
            continue
        vals = f16[idx]
        if np.all(np.isfinite(vals)):
            day_curves[d] = vals.copy()

    return Env(tr=tr, f16=f16, minute=minute, date=date, by_day=by_day, day_curves=day_curves)


def boundary_score_for_future(env: Env, e: int, fut: np.ndarray) -> float:
    tr, f16 = env.tr, env.f16
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


def fut_zero(_: int, env: Env) -> np.ndarray:
    return np.zeros(240, dtype="float64")


def fut_ar1(e: int, env: Env) -> np.ndarray:
    hist = env.f16[: e + 1]
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


def fut_ar5(e: int, env: Env) -> np.ndarray:
    hist = env.f16[: e + 1]
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


def minute_mean_from_prefix(env: Env, e: int) -> np.ndarray:
    hist = pd.DataFrame({"minute_id": env.minute[: e + 1], "feature_16": env.f16[: e + 1]})
    mm = hist.groupby("minute_id")["feature_16"].mean().reindex(range(240)).fillna(0.0).to_numpy(dtype="float64")
    return mm


def fut_expanding_all_causal(e: int, env: Env) -> np.ndarray:
    mm = minute_mean_from_prefix(env, e)
    last_minute = int(env.minute[e])
    future_minutes = (last_minute + np.arange(1, 241)) % 240
    return mm[future_minutes]


def fut_expanding_all_global(e: int, env: Env, mm_global: np.ndarray) -> np.ndarray:
    last_minute = int(env.minute[e])
    future_minutes = (last_minute + np.arange(1, 241)) % 240
    return mm_global[future_minutes]


def day_features(curve: np.ndarray) -> np.ndarray:
    o = float(curve[0])
    c = float(curve[-1])
    h = float(np.max(curve))
    l = float(np.min(curve))
    s = float(np.std(curve))
    m = float(np.mean(curve))
    am = float(np.mean(np.abs(curve)))
    r = h - l
    return np.array([1.0, o, c, h, l, s, m, am, r], dtype="float64")


def fit_macro_beta(day_curves: dict[int, np.ndarray], day_limit: int | None, lam: float = 0.1) -> np.ndarray | None:
    days = sorted(day_curves.keys())
    if day_limit is not None:
        days = [d for d in days if d <= day_limit]

    day_set = set(days)
    x_rows, y_rows = [], []
    for d in days:
        nxt = d + 1
        if nxt in day_set:
            x_rows.append(day_features(day_curves[d]))
            y_rows.append(day_curves[nxt])

    if len(x_rows) < 5:
        return None

    x = np.stack(x_rows, axis=0)
    y = np.stack(y_rows, axis=0)
    xtx = x.T @ x
    beta = np.linalg.solve(xtx + lam * np.eye(xtx.shape[0], dtype="float64"), x.T @ y)
    return beta


def fut_macro_from_beta(curve_today: np.ndarray, beta: np.ndarray, clip_abs: float = 0.1) -> np.ndarray:
    pred = day_features(curve_today) @ beta
    return np.clip(pred, -clip_abs, clip_abs).astype("float64")


def build_eval_days(env: Env, cutoff_minute: int) -> list[int]:
    eval_days: list[int] = []
    for d, idx in env.by_day.items():
        if len(idx) <= cutoff_minute:
            continue
        e = int(idx[cutoff_minute])
        if e < 239 or e + 240 >= len(env.tr):
            continue

        pref = env.f16[idx[: cutoff_minute + 1]]
        if not np.all(np.isfinite(pref)):
            continue

        eval_days.append(int(d))
    return sorted(eval_days)


def score_methods_basic(env: Env) -> pd.DataFrame:
    rows = []
    for cutoff in [27, 239]:
        eval_days = build_eval_days(env, cutoff)
        for method_name, fn in [("zero", fut_zero), ("ar1", fut_ar1), ("ar5", fut_ar5)]:
            vals = []
            for d in eval_days:
                e = int(env.by_day[d][cutoff])
                fut = fn(e, env)
                vals.append(boundary_score_for_future(env, e, fut))
            a = np.asarray(vals, dtype="float64")
            rows.append(
                {
                    "scenario": "basic",
                    "cutoff": cutoff,
                    "method": method_name,
                    "eval_days": len(eval_days),
                    "mean_mae": float(a.mean()),
                    "median_mae": float(np.median(a)),
                    "std_mae": float(a.std()),
                }
            )
    return pd.DataFrame(rows)


def score_methods_advanced_cutoff239(env: Env) -> pd.DataFrame:
    cutoff = 239
    eval_days = build_eval_days(env, cutoff)

    mm_global = (
        pd.DataFrame({"minute_id": env.minute, "feature_16": env.f16})
        .groupby("minute_id")["feature_16"]
        .mean()
        .reindex(range(240))
        .fillna(0.0)
        .to_numpy(dtype="float64")
    )
    beta_global = fit_macro_beta(env.day_curves, day_limit=None, lam=0.1)
    if beta_global is None:
        raise RuntimeError("Global macro beta fit failed.")

    beta_cache: dict[int, np.ndarray | None] = {}

    out = {
        "zero": [],
        "expanding_all_causal": [],
        "expanding_all_global": [],
        "macro_lam0p1_causal": [],
        "macro_lam0p1_global": [],
        "robust_anchor_causal": [],
        "robust_anchor_global": [],
    }

    for d in eval_days:
        idx = env.by_day[d]
        e = int(idx[cutoff])
        curve_today = env.day_curves.get(d)
        if curve_today is None:
            continue

        fut_zero_arr = fut_zero(e, env)
        fut_exp_c = fut_expanding_all_causal(e, env)
        fut_exp_g = fut_expanding_all_global(e, env, mm_global)

        if d not in beta_cache:
            beta_cache[d] = fit_macro_beta(env.day_curves, day_limit=d, lam=0.1)
        beta_c = beta_cache[d]
        if beta_c is None:
            fut_mac_c = np.zeros(240, dtype="float64")
        else:
            fut_mac_c = fut_macro_from_beta(curve_today, beta_c, clip_abs=0.1)

        fut_mac_g = fut_macro_from_beta(curve_today, beta_global, clip_abs=0.1)

        fut_rob_c = 0.75 * fut_mac_c + 0.25 * fut_exp_c
        fut_rob_g = 0.75 * fut_mac_g + 0.25 * fut_exp_g

        out["zero"].append(boundary_score_for_future(env, e, fut_zero_arr))
        out["expanding_all_causal"].append(boundary_score_for_future(env, e, fut_exp_c))
        out["expanding_all_global"].append(boundary_score_for_future(env, e, fut_exp_g))
        out["macro_lam0p1_causal"].append(boundary_score_for_future(env, e, fut_mac_c))
        out["macro_lam0p1_global"].append(boundary_score_for_future(env, e, fut_mac_g))
        out["robust_anchor_causal"].append(boundary_score_for_future(env, e, fut_rob_c))
        out["robust_anchor_global"].append(boundary_score_for_future(env, e, fut_rob_g))

    rows = []
    for method, vals in out.items():
        a = np.asarray(vals, dtype="float64")
        rows.append(
            {
                "scenario": "advanced_cutoff239",
                "cutoff": 239,
                "method": method,
                "eval_days": len(vals),
                "mean_mae": float(a.mean()),
                "median_mae": float(np.median(a)),
                "std_mae": float(a.std()),
            }
        )

    df = pd.DataFrame(rows)
    score_map = {k: np.asarray(v, dtype="float64") for k, v in out.items()}
    return df, score_map


def bootstrap_mean_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 4000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return bootstrap 95% CI for mean(a-b) under paired resampling."""
    if len(a) != len(b):
        raise ValueError("Arrays must have the same length for paired CI.")
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    diffs = a - b
    boot = np.empty(n_boot, dtype="float64")
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(diffs[idx]))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def pairwise_confidence_table(
    score_map: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """Build pairwise day-level comparison table for selected methods."""
    rows = []
    for a_name, b_name in pairs:
        a = score_map[a_name]
        b = score_map[b_name]
        if len(a) != len(b):
            raise ValueError(f"Score length mismatch for pair {a_name} vs {b_name}")
        diff = a - b
        ci_lo, ci_hi = bootstrap_mean_diff_ci(a, b)
        rows.append(
            {
                "method_a": a_name,
                "method_b": b_name,
                "eval_days": len(diff),
                "mean_diff_a_minus_b": float(np.mean(diff)),
                "median_diff_a_minus_b": float(np.median(diff)),
                "win_rate_a_better": float(np.mean(a < b)),
                "tie_rate": float(np.mean(a == b)),
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    env = load_env()

    print("=== Boundary Simulation: Basic (ZERO/AR1/AR5) ===")
    basic = score_methods_basic(env)
    basic_print = basic.rename(columns={"eval_days": "valid_backtest_days"})
    print(basic_print.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

    print("\n=== Boundary Simulation: Advanced (Cutoff 239) ===")
    adv, score_map = score_methods_advanced_cutoff239(env)
    adv_print = adv.sort_values("mean_mae").rename(columns={"eval_days": "valid_backtest_days"})
    print(adv_print.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

    pairs = [
        ("macro_lam0p1_global", "robust_anchor_global"),
        ("macro_lam0p1_global", "expanding_all_global"),
        ("robust_anchor_global", "expanding_all_global"),
        ("zero", "expanding_all_global"),
        ("zero", "robust_anchor_global"),
        ("zero", "macro_lam0p1_global"),
    ]
    pw = pairwise_confidence_table(score_map, pairs)
    print("\n=== Pairwise Confidence (Cutoff 239, day-level paired) ===")
    method_alias = {
        "macro_lam0p1_global": "macro_g",
        "robust_anchor_global": "robust_g",
        "expanding_all_global": "expand_g",
        "zero": "zero",
    }
    pw_print = pw.copy()
    pw_print["method_a"] = pw_print["method_a"].map(method_alias).fillna(pw_print["method_a"])
    pw_print["method_b"] = pw_print["method_b"].map(method_alias).fillna(pw_print["method_b"])
    pw_print = pw_print.rename(
        columns={
            "method_a": "a",
            "method_b": "b",
            "eval_days": "n",
            "mean_diff_a_minus_b": "mean_d",
            "median_diff_a_minus_b": "med_d",
            "win_rate_a_better": "win_a",
            "ci95_low": "ci_lo",
            "ci95_high": "ci_hi",
        }
    )
    pw_print = pw_print[["a", "b", "n", "mean_d", "med_d", "win_a", "ci_lo", "ci_hi"]]
    print(pw_print.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

    out_dir = Path(__file__).resolve().parents[1] / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    basic_print.to_csv(out_dir / "boundary_basic.csv", index=False)
    adv_print.to_csv(out_dir / "boundary_advanced_cutoff239.csv", index=False)
    pw.to_csv(out_dir / "boundary_pairwise_confidence_cutoff239.csv", index=False)


if __name__ == "__main__":
    main()
