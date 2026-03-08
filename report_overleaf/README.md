# Overleaf Report Package

This folder is a self-contained report package for TradeMaster Cup 2025.

## Structure

- `main.tex`: entry point
- `sections/`: report sections and appendices
- `proofs/`: executable scripts that prove each core claim
- `results/`: raw outputs from proof scripts
- `tables/`: CSV summaries from boundary simulation
- `code/`: core pipeline scripts copied from project (including helper and sanity-check scripts)

## Re-run all proofs

From repository root:

```bash
bash report_overleaf/proofs/run_all_proofs.sh
```

## Compile locally (optional)

If LaTeX is installed:

```bash
cd report_overleaf
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Upload to Overleaf

Upload entire `report_overleaf/` directory contents to an Overleaf project.
