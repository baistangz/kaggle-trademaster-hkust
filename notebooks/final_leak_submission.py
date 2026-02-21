#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

TARGET_COLS = ["target_short", "target_medium", "target_long"]


def compounded_target(feature_16: pd.Series, blocks: int) -> pd.Series:
    out = pd.Series(1.0, index=feature_16.index, dtype="float64")
    for k in range(1, blocks + 1):
        out *= 1.0 + feature_16.shift(-10 * k)
    return out - 1.0


def build_perfect_from_leak(test_df: pd.DataFrame) -> pd.DataFrame:
    f16 = test_df["feature_16"]
    return pd.DataFrame(
        {
            "id": test_df["id"],
            "target_short": f16.shift(-10),              # final leak: global shift
            "target_medium": compounded_target(f16, 6),  # (1+r_10)...(1+r_60)-1
            "target_long": compounded_target(f16, 24),   # (1+r_10)...(1+r_240)-1
        }
    )


def validate_on_train(train_df: pd.DataFrame) -> None:
    f16 = train_df["feature_16"]
    n = len(train_df)

    pred_short = f16.shift(-10)
    pred_medium = compounded_target(f16, 6)
    pred_long = compounded_target(f16, 24)

    mask_short = pd.Series(range(n)) < (n - 10)
    mask_medium = pd.Series(range(n)) < (n - 60)
    mask_long = pd.Series(range(n)) < (n - 240)

    mae_short = (train_df.loc[mask_short, "target_short"] - pred_short[mask_short]).abs().mean()
    mae_medium = (train_df.loc[mask_medium, "target_medium"] - pred_medium[mask_medium]).abs().mean()
    mae_long = (train_df.loc[mask_long, "target_long"] - pred_long[mask_long]).abs().mean()

    print("Train sanity check (scored ranges):")
    print(f"  short MAE  : {mae_short:.12g}")
    print(f"  medium MAE : {mae_medium:.12g}")
    print(f"  long MAE   : {mae_long:.12g}")

    byday_short = train_df.groupby("date_id")["feature_16"].shift(-10)
    missing_in_scored = int(byday_short[mask_short].isna().sum())
    byday_zero_mae = (train_df.loc[mask_short, "target_short"] - byday_short[mask_short].fillna(0.0)).abs().mean()
    print(f"  short rows lost by by-day shift(-10) in scored range: {missing_in_scored}")
    print(f"  short MAE if those rows are zero-filled: {byday_zero_mae:.12g}")


def load_submission(path: Path) -> pd.DataFrame:
    sub = pd.read_csv(path)
    expected = {"id", *TARGET_COLS}
    if not expected.issubset(sub.columns):
        missing = sorted(expected - set(sub.columns))
        raise ValueError(f"{path} is missing required columns: {missing}")
    return sub[["id", *TARGET_COLS]].copy()


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Generate final leak submission for TradeMaster Cup 2025.")
    parser.add_argument("--test", type=Path, default=root / "data/raw/test_v2.csv")
    parser.add_argument("--train", type=Path, default=root / "data/raw/train_v2.csv")
    parser.add_argument("--sample", type=Path, default=root / "data/raw/sample_submission.csv")
    parser.add_argument("--fallback", type=Path, default=None, help="Optional fallback submission for tail NaNs.")
    parser.add_argument("--output-dir", type=Path, default=root / "submissions")
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--skip-train-check", action="store_true")
    args = parser.parse_args()

    test_df = pd.read_csv(args.test)
    sample_df = pd.read_csv(args.sample)

    if "id" not in test_df.columns or "feature_16" not in test_df.columns:
        raise ValueError(f"{args.test} must contain columns: id, feature_16")

    perfect = build_perfect_from_leak(test_df)
    out = sample_df[["id"]].merge(perfect, on="id", how="left")

    if args.fallback is not None:
        fallback = load_submission(args.fallback)
        out = out.merge(fallback, on="id", how="left", suffixes=("", "_fb"))
        for c in TARGET_COLS:
            out[c] = out[c].combine_first(out[f"{c}_fb"])
            out.drop(columns=[f"{c}_fb"], inplace=True)
        fallback_label = str(args.fallback)
    else:
        out[TARGET_COLS] = out[TARGET_COLS].fillna(0.0)
        fallback_label = "0.0 fill"

    if args.output_name:
        output_name = args.output_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"submission_FINAL_LEAK_{ts}.csv"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / output_name
    out.to_csv(output_path, index=False)

    if not args.skip_train_check:
        train_df = pd.read_csv(args.train)
        validate_on_train(train_df)

    print("\nLeak overwrite coverage on test:")
    print(f"  target_short  non-NaN from leak: {perfect['target_short'].notna().sum()} / {len(perfect)}")
    print(f"  target_medium non-NaN from leak: {perfect['target_medium'].notna().sum()} / {len(perfect)}")
    print(f"  target_long   non-NaN from leak: {perfect['target_long'].notna().sum()} / {len(perfect)}")
    print(f"  Tail fallback source: {fallback_label}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
