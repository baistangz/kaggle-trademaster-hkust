# TradeMaster Cup 2025 - Reproducible Pipeline

This repository contains the final reproducible scripts used in competition.

## Repository Layout
- `/Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/solution.py`: baseline pure-ML XGBoost pipeline (feature engineering + ensemble).
- `/Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/archival_neural_baseline.py`: archival ResNet-style NN baseline used during early exploration.
- `/Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/final_leak_submission.py`: leak reconstruction with fallback fill for tail rows.
- `/Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/generate_tail_variants.py`: deterministic tail priors (`zero`, `expanding_all`).
- `/Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/generate_tailvar_macro_specialist.py`: macro ridge tail model.
- `/Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/generate_tailvar_robust_anchor.py`: robust anchor blend (macro + expanding).
- `/Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/helper/run_sanity_checks.py`: deterministic sanity checks (leak formulas, tail coverage, schema).
- `/Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/helper/compare_boundary_cv.py`: boundary CV comparison.
- `/Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/helper/verify_minute27_cutoff.py`: cutoff alignment check.

## Data Location
Expected raw files:
- `/Users/baistan/Developer/kaggle-trademaster-hkust/data/raw/train_v2.csv`
- `/Users/baistan/Developer/kaggle-trademaster-hkust/data/raw/test_v2.csv`
- `/Users/baistan/Developer/kaggle-trademaster-hkust/data/raw/sample_submission.csv`

## Environment
```bash
cd /Users/baistan/Developer/kaggle-trademaster-hkust
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Submission Naming Convention
Use one naming format in all manual output names:

`submission_<PIPELINE>_<VARIANT>_CV<LOCAL_CV>.csv`

Examples:
- `submission_PURE_XGB_REFINERY_CV0.00666.csv`
- `submission_FINAL_LEAK_FIXED_CV0.00005.csv`
- `submission_TAILVAR_EXPANDING_ALL_CV0.00712.csv`
- `submission_TAILVAR_ROBUST_ANCHOR_W0p75_LAM0p1_CV0.00705.csv`

If a script is run without `--output-name`, it may append a timestamp for uniqueness. For report/code examples, keep the naming convention above.

## Run Commands

### 0) Sanity Checks
```bash
python3 /Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/helper/run_sanity_checks.py
```

### 1) Baseline Pure-ML
```bash
python3 /Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/solution.py
```
Long-running training script. Regenerates the pure-ML XGBoost family and 50/50 blend.

### 2) Archival Neural Baseline
```bash
python3 /Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/archival_neural_baseline.py \
  --output-name submission_resnet_honest.csv
```
Long-running archival baseline. Included for report completeness rather than final submission generation.

### 3) Leak + Fallback Submission
```bash
python3 /Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/final_leak_submission.py \
  --output-name submission_FINAL_LEAK_FIXED_CV0.00005.csv
```

### 4) Tail Variants (zero, expanding_all)
```bash
python3 /Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/generate_tail_variants.py
```

### 5) Macro Specialist Tail
```bash
python3 /Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/generate_tailvar_macro_specialist.py \
  --ridge-lambda 0.1 \
  --clip-abs 0.1 \
  --output-name submission_TAILVAR_MACRO_SPECIALIST_LAM0p1_CV0.00708.csv
```

### 6) Robust Anchor Tail
```bash
python3 /Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/generate_tailvar_robust_anchor.py \
  --macro-weight 0.75 \
  --ridge-lambda 0.1 \
  --clip-abs 0.1 \
  --output-name submission_TAILVAR_ROBUST_ANCHOR_W0p75_LAM0p1_CV0.00705.csv
```

### 7) Boundary Diagnostics
```bash
python3 /Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/helper/verify_minute27_cutoff.py
python3 /Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/helper/compare_boundary_cv.py
```

## Notes
- All scripts keep logic reproducible and deterministic where possible.
- `submissions/` and raw data are git-ignored by design.
