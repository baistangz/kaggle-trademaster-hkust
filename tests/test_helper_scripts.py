#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from trademaster_core.paths import find_repo_root


class HelperScriptsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = find_repo_root(Path(__file__))

    def run_helper(self, rel_path: str) -> str:
        proc = subprocess.run(
            ["python3", rel_path],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout

    def test_verify_minute27_cutoff_runs(self) -> None:
        out = self.run_helper("notebooks/helper/verify_minute27_cutoff.py")
        self.assertIn("Equivalent cycle phase at endpoint is 27", out)

    def test_compare_boundary_cv_runs(self) -> None:
        out = self.run_helper("notebooks/helper/compare_boundary_cv.py")
        self.assertIn("Comparing cutoff minute 27 vs 239", out)
        self.assertIn("Side-by-side mean MAE", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
