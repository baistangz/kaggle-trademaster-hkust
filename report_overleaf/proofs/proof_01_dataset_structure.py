#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    train = pd.read_csv(root / "data/raw/train_v2.csv")
    test = pd.read_csv(root / "data/raw/test_v2.csv")

    print("=== Dataset Structure Proof ===")
    print(f"train_shape={train.shape}")
    print(f"test_shape={test.shape}")
    print(f"train_columns={len(train.columns)}")
    print(f"test_columns={len(test.columns)}")

    print("\n=== Date/Minute Structure ===")
    print(f"train_unique_date_id={train['date_id'].nunique()}")
    print(f"test_unique_date_id={test['date_id'].nunique()}")
    print(f"train_minute_range=[{int(train['minute_id'].min())}, {int(train['minute_id'].max())}]")
    print(f"test_minute_range=[{int(test['minute_id'].min())}, {int(test['minute_id'].max())}]")

    test_counts = test.groupby("date_id").size()
    partial_days = test_counts[test_counts != 240].to_dict()
    print(f"test_partial_days={partial_days}")

    first_date = int(test.iloc[0]["date_id"])
    first_min = int(test.iloc[0]["minute_id"])
    last_date = int(test.iloc[-1]["date_id"])
    last_min = int(test.iloc[-1]["minute_id"])
    print(f"test_first_row=(date_id={first_date}, minute_id={first_min})")
    print(f"test_last_row=(date_id={last_date}, minute_id={last_min})")

    n = len(test)
    phase_last = (n - 1) % 240
    phase_next = n % 240
    minute_from_phase_last = (first_min + phase_last) % 240
    minute_from_phase_next = (first_min + phase_next) % 240

    print("\n=== Cutoff Phase Alignment Check ===")
    print(f"len_test={n}")
    print(f"len_test_mod_240={n % 240}")
    print(f"(len_test_minus_1)_mod_240={phase_last}")
    print(f"minute_from_phase_last={minute_from_phase_last}")
    print(f"minute_from_phase_next={minute_from_phase_next}")

    print("\n=== NaN/Inf Audit ===")
    for name, df in [("train", train), ("test", test)]:
        numeric = df.select_dtypes(include=[np.number]).to_numpy(dtype="float64")
        print(
            f"{name}: nan_count={int(np.isnan(numeric).sum())}, "
            f"inf_count={int(np.isinf(numeric).sum())}"
        )

    for col in ["feature_16", "target_short", "target_medium", "target_long"]:
        if col not in train.columns:
            continue
        s = train[col].to_numpy(dtype="float64")
        print(
            f"train_{col}: nan={int(np.isnan(s).sum())}, inf={int(np.isinf(s).sum())}, "
            f"min={float(np.nanmin(s)):.12g}, max={float(np.nanmax(s)):.12g}"
        )


if __name__ == "__main__":
    main()
