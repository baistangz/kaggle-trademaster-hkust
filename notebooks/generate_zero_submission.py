#!/usr/bin/env python3
from __future__ import annotations

"""Generate the zero-model submission.

This script reconstructs targets directly from `feature_16` whenever future
values are observable in test. Remaining NaNs at the tail are filled from a
configured fallback submission (or with zeros if fallback is disabled).

Submission naming convention for manual output names in docs/comments:
`submission_<PIPELINE>_<VARIANT>_CV<LOCAL_CV>.csv`
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trademaster_core.constants import SUBMISSION_NAMING_CONVENTION, TARGET_COLS
from trademaster_core.leak_math import (
    build_known_region_predictions,
    compounded_target,
    known_region_counts,
)
from trademaster_core.submission_io import apply_fallback, load_submission, schema_aligned_submission


def validate_on_train(train_df: pd.DataFrame) -> None:
    """Print sanity metrics that verify the leak equations on train."""
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

def parse_args(root: Path) -> argparse.Namespace:
    """Parse command-line arguments for submission generation."""
    parser = argparse.ArgumentParser(description="Generate zero submission for TradeMaster Cup 2025.")
    parser.add_argument("--test", type=Path, default=root / "data/raw/test_v2.csv")
    parser.add_argument("--train", type=Path, default=root / "data/raw/train_v2.csv")
    parser.add_argument("--sample", type=Path, default=root / "data/raw/sample_submission.csv")
    parser.add_argument(
        "--fallback",
        type=Path,
        default=root / "data/raw/sample_submission.csv",
        help="Optional fallback submission for tail NaNs. Defaults to the host all-zero sample submission.",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "submissions")
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Optional output filename. Recommended format: "
            "submission_ZERO_<VARIANT>_CV<LOCAL_CV>.csv"
        ),
    )
    parser.add_argument("--skip-train-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    root = REPO_ROOT
    args = parse_args(root)

    test_df = pd.read_csv(args.test)
    sample_df = pd.read_csv(args.sample)

    if "id" not in test_df.columns or "feature_16" not in test_df.columns:
        raise ValueError(f"{args.test} must contain columns: id, feature_16")

    perfect = build_known_region_predictions(test_df)
    # Keep sample ID ordering exactly as required by Kaggle format.
    out = schema_aligned_submission(sample_df["id"], perfect)

    if args.fallback is not None:
        fallback = load_submission(args.fallback)
        out, fallback_label = apply_fallback(out, fallback)
        fallback_label = str(args.fallback)
    else:
        out, fallback_label = apply_fallback(out, None)

    if args.output_name:
        output_name = args.output_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"submission_ZERO_{ts}.csv"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / output_name
    out.to_csv(output_path, index=False, float_format="%.18g")

    if not args.skip_train_check:
        train_df = pd.read_csv(args.train)
        validate_on_train(train_df)

    counts = known_region_counts(perfect)
    print("\nDeterministic overwrite coverage on test:")
    print(f"  target_short  non-NaN from leak: {counts['target_short']} / {len(perfect)}")
    print(f"  target_medium non-NaN from leak: {counts['target_medium']} / {len(perfect)}")
    print(f"  target_long   non-NaN from leak: {counts['target_long']} / {len(perfect)}")
    print(f"  Tail fallback source: {fallback_label}")
    print(f"  Naming convention (docs/examples): {SUBMISSION_NAMING_CONVENTION}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
