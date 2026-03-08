#!/usr/bin/env python3
from __future__ import annotations

"""Generate macro-specialist tail submission.

Macro specialist fits a ridge model that maps previous-day aggregate features
to the next-day 240-minute `feature_16` curve, then patches unknown tail rows.

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
from trademaster_core.leak_math import build_known_region_predictions, patch_tail_from_future_f16
from trademaster_core.submission_io import save_submission
from trademaster_core.tail_models import (
    build_train_day_curves,
    fit_macro_nextday_ridge,
    predict_future_f16_macro,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Generate tailvar submission with macro next-day specialist.")
    p.add_argument("--ridge-lambda", type=float, default=1.0, help="L2 regularization for ridge fit.")
    p.add_argument("--clip-abs", type=float, default=0.1, help="Clamp predicted future f16 to +/-clip-abs.")
    p.add_argument(
        "--output-name",
        type=str,
        default="",
        help=(
            "Optional output filename. Recommended format: "
            "submission_TAILVAR_MACRO_SPECIALIST_LAM<val>_CV<LOCAL_CV>.csv. "
            "If omitted, script auto-generates one in submissions/."
        ),
    )
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    root = REPO_ROOT
    train_path = root / "data" / "raw" / "train_v2.csv"
    test_path = root / "data" / "raw" / "test_v2.csv"
    sample_path = root / "data" / "raw" / "sample_submission.csv"
    out_dir = root / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path, usecols=["date_id", "minute_id", "feature_16"])
    test_df = pd.read_csv(test_path, usecols=["id", "date_id", "minute_id", "feature_16"])
    sample_df = pd.read_csv(sample_path, usecols=["id"])

    perfect = build_known_region_predictions(test_df)
    day_curves = build_train_day_curves(train_df)
    beta, n_transitions = fit_macro_nextday_ridge(day_curves, ridge_lambda=args.ridge_lambda)
    future = predict_future_f16_macro(test_df, beta=beta, clip_abs=args.clip_abs)
    pred = patch_tail_from_future_f16(test_df, perfect, future=future)

    if args.output_name:
        out_path = out_dir / args.output_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        lam = f"{args.ridge_lambda:g}".replace(".", "p")
        out_path = out_dir / f"submission_TAILVAR_MACRO_SPECIALIST_LAM{lam}_{ts}.csv"

    save_submission(out_path, sample_df["id"], pred[["id", *TARGET_COLS]])

    print(f"Saved: {out_path}")
    print(f"Train day transitions used: {n_transitions}")
    print(f"Ridge lambda: {args.ridge_lambda}")
    print(f"Clip abs: {args.clip_abs}")
    print(f"Naming convention (docs/examples): {SUBMISSION_NAMING_CONVENTION}")


if __name__ == "__main__":
    main()
