#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TARGET_COLS = ["target_short", "target_medium", "target_long"]
WINDOWS = {"target_short": 10, "target_medium": 60, "target_long": 240}


def load_sub(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"id", *TARGET_COLS}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in {path}: {sorted(miss)}")
    return df[["id", *TARGET_COLS]].copy()


def describe_window(x: np.ndarray) -> dict[str, float | int | bool]:
    same_as_prev = np.mean(np.isclose(x[1:], x[:-1])) if len(x) > 1 else 1.0
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "zeros": int(np.isclose(x, 0.0).sum()),
        "unique_rounded_12": int(len(np.unique(np.round(x, 12)))),
        "all_zero": bool(np.all(np.isclose(x, 0.0))),
        "constant": bool(np.all(np.isclose(x, x[0]))),
        "ffill_like_ratio": float(same_as_prev),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tail behavior of two submissions.")
    parser.add_argument(
        "--a",
        type=Path,
        default=Path("submissions/submission_FINAL_LEAK_20260222_040009.csv"),
    )
    parser.add_argument(
        "--b",
        type=Path,
        default=Path("submissions/tailvar/submission_TAILVAR_EXPANDING_ALL_20260224_030331.csv"),
    )
    args = parser.parse_args()

    a = load_sub(args.a)
    b = load_sub(args.b)
    if len(a) != len(b):
        raise ValueError("Submissions have different row counts.")

    print("=== Tail Behavior Comparison ===")
    print(f"A={args.a}")
    print(f"B={args.b}")

    n = len(a)
    for col in TARGET_COLS:
        w = WINDOWS[col]
        xa = a[col].iloc[n - w :].to_numpy(dtype="float64")
        xb = b[col].iloc[n - w :].to_numpy(dtype="float64")
        sa = describe_window(xa)
        sb = describe_window(xb)

        print(f"\n[{col}] last_{w}")
        print(
            f"A: all_zero={sa['all_zero']} constant={sa['constant']} zeros={sa['zeros']} "
            f"std={sa['std']:.12g} mean={sa['mean']:.12g}"
        )
        print(
            f"B: all_zero={sb['all_zero']} constant={sb['constant']} zeros={sb['zeros']} "
            f"std={sb['std']:.12g} mean={sb['mean']:.12g}"
        )
        print(
            f"A_vs_B: mae={float(np.mean(np.abs(xa-xb))):.12g} "
            f"equal_count={int(np.isclose(xa, xb).sum())}/{w}"
        )


if __name__ == "__main__":
    main()
