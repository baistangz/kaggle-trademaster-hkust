PYTHON ?= python3

.PHONY: help sanity tests proofs boundary zero tailvars macro robust baseline nn all-checks

help:
	@echo "Available targets:"
	@echo "  sanity     Run deterministic formula/coverage/schema checks"
	@echo "  tests      Run lightweight regression tests"
	@echo "  proofs     Regenerate report proof outputs"
	@echo "  boundary   Run cutoff-alignment and boundary-CV diagnostics"
	@echo "  zero       Generate the zero submission"
	@echo "  tailvars   Generate zero and expanding_all tail variants"
	@echo "  macro      Generate the macro-specialist tail submission"
	@echo "  robust     Generate the robust-anchor tail submission"
	@echo "  baseline   Run the pure-ML baseline pipeline"
	@echo "  nn         Run the archival neural baseline"
	@echo "  all-checks Run sanity checks, tests, and proof generation"

sanity:
	$(PYTHON) notebooks/helper/run_sanity_checks.py

tests:
	$(PYTHON) -m unittest discover -s tests -v

proofs:
	bash report_overleaf/proofs/run_all_proofs.sh

boundary:
	$(PYTHON) notebooks/helper/verify_minute27_cutoff.py
	$(PYTHON) notebooks/helper/compare_boundary_cv.py

zero:
	$(PYTHON) notebooks/generate_zero_submission.py --skip-train-check

tailvars:
	$(PYTHON) notebooks/generate_tail_variants.py

macro:
	$(PYTHON) notebooks/generate_tailvar_macro_specialist.py --ridge-lambda 0.1 --clip-abs 0.1

robust:
	$(PYTHON) notebooks/generate_tailvar_robust_anchor.py --macro-weight 0.75 --ridge-lambda 0.1 --clip-abs 0.1

baseline:
	$(PYTHON) notebooks/solution.py

nn:
	$(PYTHON) notebooks/archival_neural_baseline.py

all-checks: sanity tests proofs
