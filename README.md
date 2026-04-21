# TradeMaster Cup 2025 - 4th Place Solution

This repository contains my solution for the TradeMaster Cup 2025 university Kaggle competition.

The core idea was:
- reconstruct as much of the target as possible from dataset structure,
- then forecast only the small unknown tail at the end.

The repo keeps the final submission generators, diagnostics, tests, and the report source in one place.

## What Is In This Repo

### Main scripts
- `notebooks/solution.py` - baseline pure-ML XGBoost pipeline.
- `notebooks/generate_zero_submission.py` - generates the hard-zero fallback submission.
- `notebooks/generate_tail_variants.py` - generates simple tail variants such as `zero` and `expanding_all`.
- `notebooks/generate_tailvar_macro_specialist.py` - generates the macro ridge tail model.
- `notebooks/generate_tailvar_robust_anchor.py` - generates the robust blended tail model.
- `notebooks/archival_neural_baseline.py` - archival neural baseline kept for completeness.

### Diagnostics and checks
- `notebooks/helper/run_sanity_checks.py` - quick structural checks.
- `notebooks/helper/verify_minute27_cutoff.py` - verifies the cutoff alignment logic.
- `notebooks/helper/compare_boundary_cv.py` - compares local boundary backtests.
- `tests/` - lightweight regression tests.

### Shared code
- `trademaster_core/` - shared utilities for leak math, tail models, submission IO, and path handling.

### Report material
- `report_overleaf/` - LaTeX report source, proof scripts, outputs, and mirrored code listings.
- `scripts/sync_report_code.py` - syncs repo code into `report_overleaf/code/`.

## Expected Data
Place the competition files under `data/raw/`:
- `data/raw/train_v2.csv`
- `data/raw/test_v2.csv`
- `data/raw/sample_submission.csv`

Raw data is not included in this repo.

## Setup
Run from the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Quick Start

Run the main checks:

```bash
make sanity
make tests
make proofs
```

If you want to keep the report appendix code in sync with the repo code:

```bash
make sync-report-code
```

## Common Commands

### Generate the zero submission
```bash
python3 notebooks/generate_zero_submission.py \
  --output-name submission_ZERO_CV0.00005.csv
```

### Generate simple tail variants
```bash
python3 notebooks/generate_tail_variants.py
```

### Generate the macro tail model
```bash
python3 notebooks/generate_tailvar_macro_specialist.py \
  --ridge-lambda 0.1 \
  --clip-abs 0.1 \
  --output-name submission_TAILVAR_MACRO_SPECIALIST_LAM0p1_CV0.00708.csv
```

### Generate the robust blended tail model
```bash
python3 notebooks/generate_tailvar_robust_anchor.py \
  --macro-weight 0.75 \
  --ridge-lambda 0.1 \
  --clip-abs 0.1 \
  --output-name submission_TAILVAR_ROBUST_ANCHOR_W0p75_LAM0p1_CV0.00705.csv
```

### Run the baseline XGBoost pipeline
```bash
python3 notebooks/solution.py
```

### Run diagnostics
```bash
python3 notebooks/helper/verify_minute27_cutoff.py
python3 notebooks/helper/compare_boundary_cv.py
```

## Submission Naming
When saving manual outputs, I use:

`submission_<PIPELINE>_<VARIANT>_CV<LOCAL_CV>.csv`

Examples:
- `submission_ZERO_CV0.00005.csv`
- `submission_TAILVAR_EXPANDING_ALL_CV0.00712.csv`
- `submission_TAILVAR_ROBUST_ANCHOR_W0p75_LAM0p1_CV0.00705.csv`

## Notes
- `submissions/` is git-ignored.
- Raw data is git-ignored.
- The repo is organized for reproducibility rather than as a production package.
