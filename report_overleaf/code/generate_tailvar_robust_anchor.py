#!/usr/bin/env python3
from __future__ import annotations

"""Generate robust-anchor tail submission.

Robust-anchor = weighted blend of:
- macro specialist future (ridge next-day curve)
- expanding minute-mean future

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
    compute_minute_mean_curve,
    fit_macro_nextday_ridge,
    future_from_minute_mean,
    predict_future_f16_macro,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Generate robust-anchor tailvar submission (macro+expanding blend).")
    p.add_argument("--macro-weight", type=float, default=0.75, help="Weight on macro future (0..1).")
    p.add_argument("--ridge-lambda", type=float, default=0.1, help="Ridge lambda for macro specialist.")
    p.add_argument("--clip-abs", type=float, default=0.1, help="Clamp macro future to +/- clip_abs.")
    p.add_argument(
        "--output-name",
        type=str,
        default="",
        help=(
            "Optional output filename. Recommended format: "
            "submission_TAILVAR_ROBUST_ANCHOR_W<val>_LAM<val>_CV<LOCAL_CV>.csv"
        ),
    )
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    if not (0.0 <= args.macro_weight <= 1.0):
        raise ValueError("--macro-weight must be between 0 and 1")

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
    beta, _ = fit_macro_nextday_ridge(day_curves, ridge_lambda=args.ridge_lambda)
    fut_macro = predict_future_f16_macro(test_df, beta=beta, clip_abs=args.clip_abs)
    minute_mean_curve = compute_minute_mean_curve(train_df)
    fut_expand = future_from_minute_mean(int(test_df["minute_id"].iloc[-1]), minute_mean_curve)

    # Robust-anchor blend (requested):
    # future = w * macro + (1-w) * expanding_all
    w = float(args.macro_weight)
    future = w * fut_macro + (1.0 - w) * fut_expand

    pred = patch_tail_from_future_f16(test_df, perfect, future=future)

    if args.output_name:
        out_path = out_dir / args.output_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        w_tag = str(args.macro_weight).replace(".", "p")
        lam_tag = str(args.ridge_lambda).replace(".", "p")
        out_path = out_dir / f"submission_TAILVAR_ROBUST_ANCHOR_W{w_tag}_LAM{lam_tag}_{ts}.csv"

    save_submission(out_path, sample_df["id"], pred[["id", *TARGET_COLS]])

    print(f"Saved: {out_path}")
    print(f"macro_weight={w:.6f}")
    print(f"expand_weight={1.0 - w:.6f}")
    print(f"weight_sum={(w + (1.0 - w)):.6f}")
    print(f"ridge_lambda={args.ridge_lambda}")
    print(f"clip_abs={args.clip_abs}")
    print(f"future_range=[{future.min():.8f}, {future.max():.8f}]")
    print(f"Naming convention (docs/examples): {SUBMISSION_NAMING_CONVENTION}")


if __name__ == "__main__":
    main()
