#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trademaster_core.leak_math import build_known_region_predictions, patch_tail_from_future_f16
from trademaster_core.paths import find_repo_root
from trademaster_core.tail_models import (
    build_train_day_curves,
    compute_minute_mean_curve,
    fit_macro_nextday_ridge,
    future_from_minute_mean,
    future_zero,
    predict_future_f16_macro,
)


class TailModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = find_repo_root(Path(__file__))
        cls.train_df = pd.read_csv(root / "data" / "raw" / "train_v2.csv", usecols=["date_id", "minute_id", "feature_16"])
        cls.test_df = pd.read_csv(root / "data" / "raw" / "test_v2.csv", usecols=["id", "date_id", "minute_id", "feature_16"])

    def test_zero_and_expanding_futures_are_finite_and_length_240(self) -> None:
        minute_mean_curve = compute_minute_mean_curve(self.train_df)
        fut_zero = future_zero()
        fut_expand = future_from_minute_mean(int(self.test_df["minute_id"].iloc[-1]), minute_mean_curve)

        self.assertEqual(len(fut_zero), 240)
        self.assertEqual(len(fut_expand), 240)
        self.assertTrue(np.isfinite(fut_zero).all())
        self.assertTrue(np.isfinite(fut_expand).all())

    def test_macro_future_and_tail_patch_are_finite(self) -> None:
        day_curves = build_train_day_curves(self.train_df)
        beta, n_transitions = fit_macro_nextday_ridge(day_curves, ridge_lambda=0.1)
        future = predict_future_f16_macro(self.test_df, beta=beta, clip_abs=0.1)
        known = build_known_region_predictions(self.test_df)
        patched = patch_tail_from_future_f16(self.test_df, known, future)

        self.assertGreater(n_transitions, 0)
        self.assertEqual(len(future), 240)
        self.assertTrue(np.isfinite(future).all())
        self.assertFalse(patched[["target_short", "target_medium", "target_long"]].isna().any().any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
