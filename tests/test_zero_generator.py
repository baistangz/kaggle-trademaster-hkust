#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trademaster_core.paths import find_repo_root


class ZeroGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = find_repo_root(Path(__file__))
        cls.script = cls.root / "notebooks" / "generate_zero_submission.py"
        cls.sample_df = pd.read_csv(cls.root / "data" / "raw" / "sample_submission.csv")

    def test_zero_generator_creates_schema_aligned_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_name = "submission_ZERO_TEST.csv"
            cmd = [
                "python3",
                str(self.script),
                "--skip-train-check",
                "--output-dir",
                tmpdir,
                "--output-name",
                out_name,
            ]
            subprocess.run(cmd, check=True, cwd=self.root, capture_output=True, text=True)

            out_df = pd.read_csv(Path(tmpdir) / out_name)
            self.assertEqual(list(out_df.columns), list(self.sample_df.columns))
            self.assertTrue(out_df["id"].equals(self.sample_df["id"]))
            self.assertFalse(out_df[["target_short", "target_medium", "target_long"]].isna().any().any())

            tail_short = out_df["target_short"].tail(10).to_numpy(dtype="float64")
            tail_medium = out_df["target_medium"].tail(60).to_numpy(dtype="float64")
            tail_long = out_df["target_long"].tail(240).to_numpy(dtype="float64")

            self.assertTrue(np.allclose(tail_short, 0.0))
            self.assertTrue(np.allclose(tail_medium, 0.0))
            self.assertTrue(np.allclose(tail_long, 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
