from __future__ import annotations

"""Submission schema helpers."""

from pathlib import Path

import pandas as pd

from .constants import TARGET_COLS


def load_submission(path: Path) -> pd.DataFrame:
    """Load a submission CSV and validate the Kaggle schema."""
    sub = pd.read_csv(path)
    expected = {"id", *TARGET_COLS}
    if not expected.issubset(sub.columns):
        missing = sorted(expected - set(sub.columns))
        raise ValueError(f"{path} is missing required columns: {missing}")
    return sub[["id", *TARGET_COLS]].copy()


def schema_aligned_submission(sample_ids: pd.Series, pred: pd.DataFrame) -> pd.DataFrame:
    """Return predictions aligned to sample-submission ID order."""
    out = pd.DataFrame({"id": sample_ids})
    return out.merge(pred[["id", *TARGET_COLS]], on="id", how="left")


def save_submission(path: Path, sample_ids: pd.Series, pred: pd.DataFrame) -> None:
    """Write schema-aligned predictions to disk."""
    schema_aligned_submission(sample_ids, pred).to_csv(path, index=False, float_format="%.18g")


def apply_fallback(pred: pd.DataFrame, fallback: pd.DataFrame | None) -> tuple[pd.DataFrame, str]:
    """Fill remaining NaNs from ``fallback`` or hard zero when fallback is absent."""
    out = pred.copy()
    if fallback is not None:
        out = out.merge(fallback[["id", *TARGET_COLS]], on="id", how="left", suffixes=("", "_fb"))
        for col in TARGET_COLS:
            out[col] = out[col].combine_first(out[f"{col}_fb"])
            out.drop(columns=[f"{col}_fb"], inplace=True)
        return out, "fallback submission"

    out[TARGET_COLS] = out[TARGET_COLS].fillna(0.0)
    return out, "0.0 fill"
