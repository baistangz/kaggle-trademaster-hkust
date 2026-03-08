# TradeMaster Cup 2025 - Reproducible Pipeline

This repository contains the final reproducible scripts used in competition.

## Repository Layout
- `/Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/solution.py`: baseline pure-ML XGBoost pipeline (feature engineering + ensemble).
- `/Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/final_leak_submission.py`: leak reconstruction with fallback fill for tail rows.
- `/Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/generate_tail_variants.py`: deterministic tail priors (`zero`, `expanding_all`).
- `/Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/generate_tailvar_macro_specialist.py`: macro ridge tail model.
- `/Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/generate_tailvar_robust_anchor.py`: robust anchor blend (macro + expanding).
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

### 1) Baseline Pure-ML
```bash
python3 /Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/solution.py
```

### 2) Leak + Fallback Submission
```bash
python3 /Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/final_leak_submission.py \
  --fallback /Users/baistan/Developer/kaggle-trademaster-hkust/submissions/submission_Ensemble_Ref50_Rob50_CV0.67173.csv \
  --output-name submission_FINAL_LEAK_FIXED_CV0.00005.csv
```

### 3) Tail Variants (zero, expanding_all)
```bash
python3 /Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/generate_tail_variants.py
```

### 4) Macro Specialist Tail
```bash
python3 /Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/generate_tailvar_macro_specialist.py \
  --ridge-lambda 0.1 \
  --clip-abs 0.1 \
  --output-name submission_TAILVAR_MACRO_SPECIALIST_LAM0p1_CV0.00708.csv
```

### 5) Robust Anchor Tail
```bash
python3 /Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/generate_tailvar_robust_anchor.py \
  --macro-weight 0.75 \
  --ridge-lambda 0.1 \
  --clip-abs 0.1 \
  --output-name submission_TAILVAR_ROBUST_ANCHOR_W0p75_LAM0p1_CV0.00705.csv
```

### 6) Boundary Diagnostics
```bash
python3 /Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/helper/verify_minute27_cutoff.py
python3 /Users/baistan/Developer/kaggle-trademaster-hkust/notebooks/helper/compare_boundary_cv.py
```

## Notes
- All scripts keep logic reproducible and deterministic where possible.
- `submissions/` and raw data are git-ignored by design.
