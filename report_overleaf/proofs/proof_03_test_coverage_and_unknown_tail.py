#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd


def compounded_target(feature_16: pd.Series, blocks: int) -> pd.Series:
    out = pd.Series(1.0, index=feature_16.index, dtype="float64")
    for k in range(1, blocks + 1):
        out *= 1.0 + feature_16.shift(-10 * k)
    return out - 1.0


def describe_tail(test: pd.DataFrame, pred: pd.Series, name: str, tail: int) -> None:
    na_idx = pred[pred.isna()].index
    print(f"{name}: non_na={pred.notna().sum()}, na={pred.isna().sum()}, expected_tail={tail}")
    if len(na_idx) > 0:
        first_i = int(na_idx.min())
        last_i = int(na_idx.max())
        first_row = test.iloc[first_i]
        last_row = test.iloc[last_i]
        print(
            f"  na_index_range=[{first_i}, {last_i}] "
            f"first=(date_id={int(first_row['date_id'])}, minute_id={int(first_row['minute_id'])}) "
            f"last=(date_id={int(last_row['date_id'])}, minute_id={int(last_row['minute_id'])})"
        )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    test = pd.read_csv(root / "data/raw/test_v2.csv", usecols=["date_id", "minute_id", "feature_16"])
    f16 = test["feature_16"]

    short = f16.shift(-10)
    medium = compounded_target(f16, 6)
    long_ = compounded_target(f16, 24)

    print("=== Leak Coverage on Test ===")
    print(f"test_rows={len(test)}")
    describe_tail(test, short, "target_short", 10)
    describe_tail(test, medium, "target_medium", 60)
    describe_tail(test, long_, "target_long", 240)


if __name__ == "__main__":
    main()
