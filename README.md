# TradeMaster Cup 2025 - Reproducible Pipeline

This repository contains the final reproducible scripts used in competition.

## Repository Layout
- `notebooks/solution.py`: baseline pure-ML XGBoost pipeline (feature engineering + ensemble).
- `notebooks/archival_neural_baseline.py`: archival ResNet-style NN baseline used during early exploration.
- `notebooks/generate_zero_submission.py`: zero-model generator with hard-zero fill for the unknown tail rows.
- `notebooks/generate_tail_variants.py`: deterministic tail priors (`zero`, `expanding_all`).
- `notebooks/generate_tailvar_macro_specialist.py`: macro ridge tail model.
- `notebooks/generate_tailvar_robust_anchor.py`: robust anchor blend (macro + expanding).
- `notebooks/helper/run_sanity_checks.py`: deterministic sanity checks (leak formulas, tail coverage, schema).
- `notebooks/helper/compare_boundary_cv.py`: boundary CV comparison.
- `notebooks/helper/verify_minute27_cutoff.py`: cutoff alignment check.

## Data Location
Expected raw files:
- `data/raw/train_v2.csv`
- `data/raw/test_v2.csv`
- `data/raw/sample_submission.csv`

## Environment
```bash
# Run from the repository root
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Quickstart
```bash
make sanity
make tests
make proofs
make sync-report-code
```
These targets provide the shortest reproducible path for structural checks, regression checks, report-proof regeneration, and report-code synchronization.

## Engineering Structure
- `notebooks/` contains CLI entrypoints for baseline and submission generation.
- `trademaster_core/` contains shared leak math, tail priors, path resolution, and submission IO utilities.
- `scripts/sync_report_code.py` synchronizes repo source files into `report_overleaf/code/` so appendix code mirrors do not have to be maintained manually.

## Submission Naming Convention
Use one naming format in all manual output names:

`submission_<PIPELINE>_<VARIANT>_CV<LOCAL_CV>.csv`

Examples:
- `submission_PURE_XGB_REFINERY_CV0.00666.csv`
- `submission_ZERO_CV0.00005.csv`
- `submission_TAILVAR_EXPANDING_ALL_CV0.00712.csv`
- `submission_TAILVAR_ROBUST_ANCHOR_W0p75_LAM0p1_CV0.00705.csv`

If a script is run without `--output-name`, it may append a timestamp for uniqueness. For report/code examples, keep the naming convention above.

## Run Commands

### 0) Sanity Checks
```bash
python3 notebooks/helper/run_sanity_checks.py
```

### 0.5) Lightweight Regression Tests
```bash
python3 -m unittest discover -s tests -v
```

### 0.6) Synchronize Report Code Mirror
```bash
python3 scripts/sync_report_code.py
```

### 1) Baseline Pure-ML
```bash
python3 notebooks/solution.py
```
Long-running training script. Regenerates the pure-ML XGBoost family and 50/50 blend.

### 2) Archival Neural Baseline
```bash
python3 notebooks/archival_neural_baseline.py \
  --output-name submission_resnet_honest.csv
```
Long-running archival baseline. Included for report completeness rather than final submission generation.

### 3) Zero Submission
```bash
python3 notebooks/generate_zero_submission.py \
  --output-name submission_ZERO_CV0.00005.csv
```

### 4) Tail Variants (zero, expanding_all)
```bash
python3 notebooks/generate_tail_variants.py
```

### 5) Macro Specialist Tail
```bash
python3 notebooks/generate_tailvar_macro_specialist.py \
  --ridge-lambda 0.1 \
  --clip-abs 0.1 \
  --output-name submission_TAILVAR_MACRO_SPECIALIST_LAM0p1_CV0.00708.csv
```

### 6) Robust Anchor Tail
```bash
python3 notebooks/generate_tailvar_robust_anchor.py \
  --macro-weight 0.75 \
  --ridge-lambda 0.1 \
  --clip-abs 0.1 \
  --output-name submission_TAILVAR_ROBUST_ANCHOR_W0p75_LAM0p1_CV0.00705.csv
```

### 7) Boundary Diagnostics
```bash
python3 notebooks/helper/verify_minute27_cutoff.py
python3 notebooks/helper/compare_boundary_cv.py
```

## Notes
- All scripts keep logic reproducible and deterministic where possible.
- `submissions/` and raw data are git-ignored by design.
