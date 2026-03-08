#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROOF_DIR="$ROOT_DIR/report_overleaf/proofs"
RESULT_DIR="$ROOT_DIR/report_overleaf/results"

mkdir -p "$RESULT_DIR"

python3 "$PROOF_DIR/proof_01_dataset_structure.py" > "$RESULT_DIR/proof_01_dataset_structure.txt"
python3 "$PROOF_DIR/proof_02_leak_formula_validation.py" > "$RESULT_DIR/proof_02_leak_formula_validation.txt"
python3 "$PROOF_DIR/proof_03_test_coverage_and_unknown_tail.py" > "$RESULT_DIR/proof_03_test_coverage_and_unknown_tail.txt"
python3 "$PROOF_DIR/proof_04_tail_submission_behavior.py" > "$RESULT_DIR/proof_04_tail_submission_behavior.txt"
python3 "$PROOF_DIR/proof_05_boundary_simulation.py" > "$RESULT_DIR/proof_05_boundary_simulation.txt"
python3 "$PROOF_DIR/proof_06_submission_correlation.py" > "$RESULT_DIR/proof_06_submission_correlation.txt"
python3 "$PROOF_DIR/proof_07_extract_baseline_training_blocks.py" > "$RESULT_DIR/proof_07_extract_baseline_training_blocks.txt"
python3 "$PROOF_DIR/proof_08_legacy_verify_minute27_cutoff_output.py" > "$RESULT_DIR/proof_08_legacy_verify_minute27_cutoff_output.txt"
python3 "$PROOF_DIR/proof_09_legacy_compare_boundary_cv_output.py" > "$RESULT_DIR/proof_09_legacy_compare_boundary_cv_output.txt"

echo "All proof scripts finished. Outputs are in $RESULT_DIR"
