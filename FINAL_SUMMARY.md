# Final Summary: OASIS Paper and MCV Adapter State

## Scope

This summary reflects the **current authoritative state** of the OASIS paper and the PostgreSQL statistics-format conversion materials as of 2026-03-11.

Two rules should guide any future edits:

1. **The paper is authoritative for claims, framing, and default parameters.**
2. **`experiments/results/mcv_validation_results.json` is authoritative for exact MCV validation numbers.**
3. **`experiments/results/synthetic_paper_suite_tree_isomer_v3/` is authoritative for the current cleaned synthetic-paper reruns used by the paper text.**

## What Was Fixed

The earlier MCV validation workflow had a modeling bug:

- it treated PostgreSQL histograms as if they stored explicit bucket-frequency vectors;
- real PostgreSQL single-column statistics store **ordered histogram boundary values** for the residual non-MCV population.

The validation script, figures, appendix, and paper text were updated so that all of them now use the same PostgreSQL model.

## Current Paper-Level Claims

The main paper now supports the following claims consistently:

- OASIS corrects stale column statistics from execution feedback without rescanning tables.
- The deployed model is a lightweight attention-pooled MLP with about `38K` parameters.
- On synthetic drift workloads, OASIS achieves up to `62.0%` Q-Error reduction over the stale prior.
- Across unseen initial distributions at `q=10`, OASIS improves over the stale prior by `44.4%--60.8%`.
- On PostgreSQL 14 / TPC-DS, OASIS reduces execution time from `364.2s` to `339.3s` and recovers `52.0%` of Full ANALYZE's gain over the stale baseline.
- The statistics-format conversion interface supports PostgreSQL-style statistics while preserving probability mass and keeping conversion overhead low.

## Clean Synthetic Suite Status

The synthetic experiment pipeline was consolidated on 2026-03-11 to eliminate conflicting runners.

- authoritative entrypoint: `experiments/run_synthetic_paper_suite.py`
- authoritative outputs: `experiments/results/synthetic_paper_suite_tree_isomer_v3/main/summary.json`, `experiments/results/synthetic_paper_suite_tree_isomer_v3/distribution/summary.json`, and `experiments/results/synthetic_paper_suite_tree_isomer_v3/sensitivity/summary.json`
- archived legacy material: `legacy_experiments_backup/2026-03-11/`
- important paper rule: the fresh sensitivity rerun exists, but the paper intentionally keeps the `K=16` subsection narrative by author choice

## Current MCV Adapter Results

### Round-Trip Fidelity

- total mass error: `0.0`
- MCV mass error: `0.0`
- residual mass error: `0.0`
- residual quantile MAE mean: `2.392887868050384e-17`
- residual quantile MAE max: `9.178576490712927e-17`

### Selectivity Consistency

- global MAE: `1.0720625000010417e-04`
- global P95: `1.0e-04`
- global P99: `3.788937499999999e-03`
- global max: `3.8387500000000006e-03`

### Interface Conversion Overhead

- PostgreSQL-like `100 MCV / 10 bins`: `0.0660048660 ms`
- PostgreSQL-like `100 MCV / 20 bins`: `0.0731946860 ms`
- PostgreSQL-like `100 MCV / 50 bins`: `0.0641457660 ms`
- worst tested `50 MCV / 50 bins`: `1.0270490040 ms`

### Threshold Sensitivity

| Threshold | MCV Count | Selectivity MAE | P99 |
|-----------|-----------|-----------------|-----|
| `0.1%` | `20` | `0.0010266594` | `0.0038621985` |
| `0.5%` | `20` | `0.0010266594` | `0.0038621985` |
| `1%`   | `17` | `0.0015147205` | `0.0061021000` |
| `2%`   | `9`  | `0.0029720909` | `0.0138112435` |
| `5%`   | `3`  | `0.0045435225` | `0.0195120508` |

## Files That Were Updated

### Paper

- `paper/main.tex`
- `paper/main.pdf`
- `paper/appendix/mcv_adapter.tex`
- `paper/appendix/system_details.tex`
- `paper/appendix/mcv_adapter_interface.py`
- `paper/appendix/README.md`

### Experiments

- `experiments/mcv_adapter_validation.py`
- `experiments/generate_mcv_figures.py`
- `experiments/results/mcv_validation_results.json`
- `experiments/run_synthetic_paper_suite.py`
- `experiments/README_SYNTHETIC_PAPER_SUITE.md`
- `experiments/results/synthetic_paper_suite_tree_isomer_v3/manifest.json`
- `experiments/results/synthetic_paper_suite_tree_isomer_v3/main/summary.json`
- `experiments/results/synthetic_paper_suite_tree_isomer_v3/distribution/summary.json`
- `experiments/results/synthetic_paper_suite_tree_isomer_v3/sensitivity/summary.json`

### Figures

- `paper/figures/mcv_validation_summary.pdf`
- `paper/figures/mcv_performance_overhead.pdf`
- `paper/figures/mcv_threshold_sensitivity.pdf`
- `paper/figures/ablation_qerror.pdf`
- `paper/figures/ablation_selerror.pdf`
- `paper/figures/ablation_mae.pdf`

### Documentation

- `MCV_ADAPTER_SUMMARY.md`
- `MCV_ADAPTER_CHECKLIST.md`
- `MCV_PROBLEM_AND_SOLUTION.md`
- `MCV_VERIFICATION_REPORT.md`
- `experiments/README_MCV_VALIDATION.md`
- `FIGURE_IMPROVEMENTS.md`
- `FINAL_SUMMARY.md`
- `PAPER_INTEGRATED_HANDOFF.md`

## Validation Commands Run

```bash
python3 -m py_compile experiments/mcv_adapter_validation.py
python3 -m py_compile experiments/generate_mcv_figures.py
python3 -m py_compile paper/appendix/mcv_adapter_interface.py
cd experiments && python3 mcv_adapter_validation.py
cd experiments && python3 generate_mcv_figures.py
cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Final Status

- The PostgreSQL histogram-modeling bug in the old validation workflow is fixed.
- The paper, appendix, scripts, figures, and markdown summaries are now aligned.
- Legacy conflicting synthetic runners are archived, and the cleaned synthetic paper numbers now come from `experiments/results/synthetic_paper_suite_tree_isomer_v3/`.
- The paper keeps the existing `K=16` sensitivity narrative intentionally, even though a fresh sensitivity rerun also exists for reference.
- Future models should start from `PAPER_INTEGRATED_HANDOFF.md` and treat it as the primary orientation document.
