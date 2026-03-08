#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def extract_function_block(lines: list[str], func_name: str) -> list[str]:
    """Extract one top-level function block by name from source lines."""
    start = None
    marker = f"def {func_name}("
    for i, ln in enumerate(lines):
        if ln.startswith(marker):
            start = i
            break
    if start is None:
        raise ValueError(f"Function not found: {func_name}")

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("def "):
            end = j
            break
    return lines[start:end]


def main() -> None:
    """Build a baseline artifact from key training blocks in solution.py."""
    root = Path(__file__).resolve().parents[2]
    src_path = root / "notebooks/solution.py"
    lines = src_path.read_text().splitlines()
    keep_funcs = [
        "train_champion_refinery",
        "train_purist",
        "train_robust",
        "blend_refinery_and_robust",
    ]

    out_lines = []
    out_lines.append("# Extracted from notebooks/solution.py")
    out_lines.append("# Included as baseline ML proof artifact for report appendix")
    out_lines.append("")

    for func_name in keep_funcs:
        block = extract_function_block(lines, func_name)
        out_lines.append(f"# ===== BASELINE FUNCTION: {func_name} =====")
        out_lines.extend(block)
        out_lines.append("")

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "extracted_baseline_training_blocks.py"
    out_path.write_text("\n".join(out_lines) + "\n")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
