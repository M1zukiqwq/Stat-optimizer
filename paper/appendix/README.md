# Appendix Materials

This directory contains the supplementary appendix snippets referenced by the
paper as well as extra material kept in the repository for the longer artifact
version.

## Scope

These files are **not automatically included** in `paper/main.tex` for the
current conference-length submission. Instead, the main paper points to them as
supplementary materials, and they are kept synchronized with the live scripts
and official experiment outputs.

## Files

### `mcv_adapter.tex`

Portable statistics-format conversion appendix:

- formal three-phase reconstruction / correction / decomposition pattern,
- PostgreSQL `MCV + histogram_bounds` worked example,
- consistency and overhead discussion aligned with
  `experiments/results/mcv_validation_results.json`.

### `system_details.tex`

System-details appendix trimmed from the main paper:

- attention-pooled MLP architecture,
- training details and implementation notes,
- per-conjunct instrumentation and bounded feedback collection behavior.

### `mcv_adapter_interface.py`

Documentation-oriented Python reference interface for the conversion adapter.

## Update Rules

- If `mcv_adapter.tex` changes, keep it numerically aligned with
  `experiments/results/mcv_validation_results.json`.
- If `system_details.tex` changes, keep it aligned with the current main-text
  claims about lightweight feedback collection and non-invasive integration.

## Useful Commands

### Rebuild the main paper

```bash
cd ../
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

### TPC-DS drift experiments

`SCD Type 2` / `Fact growth` 的真实评测不再保留在 synthetic appendix 中，
请改用 `tpcds_experiment/EXPERIMENT_MANUAL.md`。

### Re-run MCV validation

```bash
cd ..
python3 experiments/mcv_adapter_validation.py
python3 experiments/generate_mcv_figures.py
```
