#!/usr/bin/env python3
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

SUBS = {
    "final_leak": "submissions/submission_FINAL_LEAK_20260222_040009.csv",
    "expanding_all": "submissions/tailvar/submission_TAILVAR_EXPANDING_ALL_20260224_030331.csv",
    "robust_anchor": "submissions/submission_TAILVAR_ROBUST_ANCHOR_W0p75_LAM0p1_20260224_214808.csv",
    "macro_lam0p1": "submissions/submission_TAILVAR_MACRO_SPECIALIST_LAM0p1_CHECK.csv",
    "zero_patched": "submissions/tailvar/submission_TAILVAR_ZERO_PATCHED_20260223_164342.csv",
}

WINDOWS = {
    "target_short": 10,
    "target_medium": 60,
    "target_long": 240,
}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    dfs = {k: pd.read_csv(root / v) for k, v in SUBS.items()}

    print("=== Submission Correlation in Unknown Tail Windows ===")
    for col, w in WINDOWS.items():
        print(f"\n[{col}] last_{w}")
        for a, b in combinations(dfs.keys(), 2):
            xa = dfs[a][col].iloc[-w:].to_numpy(dtype="float64")
            xb = dfs[b][col].iloc[-w:].to_numpy(dtype="float64")

            if np.std(xa) == 0 or np.std(xb) == 0:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(xa, xb)[0, 1])

            mae = float(np.mean(np.abs(xa - xb)))
            print(f"{a:12s} vs {b:12s} | corr={corr:.6f} | mae={mae:.8f}")


if __name__ == "__main__":
    main()
