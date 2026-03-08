#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trademaster_core.leak_math import build_known_region_predictions, compounded_target
from trademaster_core.paths import find_repo_root


class StructuralClaimsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = find_repo_root(Path(__file__))
        cls.train_df = pd.read_csv(root / "data" / "raw" / "train_v2.csv")
        cls.test_df = pd.read_csv(root / "data" / "raw" / "test_v2.csv")
        cls.sample_df = pd.read_csv(root / "data" / "raw" / "sample_submission.csv")

    def test_feature16_leak_equations(self) -> None:
        train_df = self.train_df
        f16 = train_df["feature_16"]
        n = len(train_df)

        pred_short = f16.shift(-10)
        pred_medium = compounded_target(f16, 6)
        pred_long = compounded_target(f16, 24)

        mask_short = np.arange(n) < (n - 10)
        mask_medium = np.arange(n) < (n - 60)
        mask_long = np.arange(n) < (n - 240)

        mae_short = float((train_df.loc[mask_short, "target_short"] - pred_short[mask_short]).abs().mean())
        mae_medium = float((train_df.loc[mask_medium, "target_medium"] - pred_medium[mask_medium]).abs().mean())
        mae_long = float((train_df.loc[mask_long, "target_long"] - pred_long[mask_long]).abs().mean())

        self.assertLessEqual(mae_short, 1e-12)
        self.assertLessEqual(mae_medium, 1e-11)
        self.assertLessEqual(mae_long, 1e-11)

    def test_unknown_tail_sizes(self) -> None:
        f16 = self.test_df["feature_16"]
        self.assertEqual(int(f16.shift(-10).isna().sum()), 10)
        self.assertEqual(int(compounded_target(f16, 6).isna().sum()), 60)
        self.assertEqual(int(compounded_target(f16, 24).isna().sum()), 240)

    def test_cutoff_phase_alignment(self) -> None:
        test_df = self.test_df
        self.assertEqual(len(test_df) % 240, 28)
        self.assertEqual((len(test_df) - 1) % 240, 27)

        first_row = test_df.iloc[0]
        last_row = test_df.iloc[-1]
        self.assertEqual((int(first_row["date_id"]), int(first_row["minute_id"])), (582, 212))
        self.assertEqual((int(last_row["date_id"]), int(last_row["minute_id"])), (725, 239))

    def test_submission_schema_matches_sample(self) -> None:
        sample_df = self.sample_df
        pred = build_known_region_predictions(self.test_df)
        out = sample_df[["id"]].merge(pred, on="id", how="left")
        self.assertEqual(list(out.columns), ["id", "target_short", "target_medium", "target_long"])
        self.assertTrue(out["id"].equals(sample_df["id"]))
        self.assertEqual(len(out), len(sample_df))


if __name__ == "__main__":
    unittest.main(verbosity=2)
