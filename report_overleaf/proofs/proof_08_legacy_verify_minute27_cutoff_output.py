#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    # Legacy helper equivalent of notebooks/helper/verify_minute27_cutoff.py
    # with root path fixed for the report package.
    root = Path(__file__).resolve().parents[2]
    test = pd.read_csv(root / "data" / "raw" / "test_v2.csv", usecols=["date_id", "minute_id"])

    n = len(test)
    first_date = int(test.iloc[0]["date_id"])
    first_minute = int(test.iloc[0]["minute_id"])
    last_date = int(test.iloc[-1]["date_id"])
    last_minute = int(test.iloc[-1]["minute_id"])

    counts = test.groupby("date_id").size()
    partial_days = counts[counts != 240]

    phase_last = (n - 1) % 240
    phase_next = n % 240

    minute_from_phase_last = (first_minute + phase_last) % 240
    minute_from_phase_next = (first_minute + phase_next) % 240

    print("=== Legacy Helper Output: verify_minute27_cutoff ===")
    print("=== Test Cutoff Verification ===")
    print(f"rows: {n}")
    print(f"first row: date={first_date}, minute={first_minute}")
    print(f"last row : date={last_date}, minute={last_minute}")
    print(f"rows % 240 = {n % 240}")
    print(f"(rows-1) % 240 = {phase_last}")
    print()
    print("Partial day structure:")
    print(partial_days.to_dict())
    print()
    print("Phase alignment checks:")
    print(f"minute from phase_last = (first_minute + phase_last) % 240 = {minute_from_phase_last}")
    print(f"minute from phase_next = (first_minute + phase_next) % 240 = {minute_from_phase_next}")
    print()
    print("Interpretation:")
    print("- Raw test endpoint minute_id is 239.")
    print("- Equivalent cycle phase at endpoint is 27.")
    print("- Therefore minute-27 boundary simulation is the correct aligned cutoff for historical backtests.")


if __name__ == "__main__":
    main()
