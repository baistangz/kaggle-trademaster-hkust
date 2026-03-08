#!/usr/bin/env python3
from __future__ import annotations

"""Lightweight deterministic sanity checks for leakage and tail coverage."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trademaster_core.constants import TARGET_COLS
from trademaster_core.leak_math import build_known_region_predictions, compounded_target, validate_leak_mae
from trademaster_core.submission_io import schema_aligned_submission


def check_leak_equations(train_df: pd.DataFrame) -> None:
    """Verify deterministic target equations on scored-valid windows."""
    stats = validate_leak_mae(train_df)
    mae_short = stats["target_short"]
    mae_medium = stats["target_medium"]
    mae_long = stats["target_long"]

    assert mae_short <= 1e-12, f"target_short leak check failed: MAE={mae_short}"
    assert mae_medium <= 1e-11, f"target_medium leak check failed: MAE={mae_medium}"
    assert mae_long <= 1e-11, f"target_long leak check failed: MAE={mae_long}"

    print(f"[PASS] Leak equations: short={mae_short:.3e} medium={mae_medium:.3e} long={mae_long:.3e}")


def check_test_tail_coverage(test_df: pd.DataFrame) -> None:
    """Verify unknown tail sizes implied by shift/compound windows."""
    f16 = test_df["feature_16"]
    short_na = int(f16.shift(-10).isna().sum())
    medium_na = int(compounded_target(f16, 6).isna().sum())
    long_na = int(compounded_target(f16, 24).isna().sum())

    assert short_na == 10, f"Expected 10 short NaNs, got {short_na}"
    assert medium_na == 60, f"Expected 60 medium NaNs, got {medium_na}"
    assert long_na == 240, f"Expected 240 long NaNs, got {long_na}"

    print(f"[PASS] Test unknown tails: short={short_na} medium={medium_na} long={long_na}")


def check_submission_schema(sample_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Build leak-base frame and verify schema/order compatibility."""
    pred = build_known_region_predictions(test_df)
    out = schema_aligned_submission(sample_df["id"], pred)

    assert list(out.columns) == ["id", *TARGET_COLS], "Submission schema mismatch"
    assert out["id"].equals(sample_df["id"]), "Submission id ordering mismatch"
    assert len(out) == len(sample_df), "Submission length mismatch"

    print("[PASS] Submission schema/order check")


def check_known_inf_location(train_df: pd.DataFrame) -> None:
    """Ensure known Inf footprint remains limited to feature_2."""
    num = train_df.select_dtypes(include=["number"])
    arr = num.to_numpy()
    rows, cols = np.where(np.isinf(arr))
    inf_cols = sorted(set(num.columns[cols]))
    assert inf_cols == ["feature_2"], f"Unexpected Inf columns: {inf_cols}"
    print(f"[PASS] Inf audit: columns={inf_cols}, count={len(rows)}")


def main() -> None:
    """Run all sanity checks."""
    root = REPO_ROOT
    train_df = pd.read_csv(root / "data" / "raw" / "train_v2.csv")
    test_df = pd.read_csv(root / "data" / "raw" / "test_v2.csv")
    sample_df = pd.read_csv(root / "data" / "raw" / "sample_submission.csv")

    check_leak_equations(train_df)
    check_test_tail_coverage(test_df)
    check_submission_schema(sample_df, test_df)
    check_known_inf_location(train_df)
    print("ALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()
