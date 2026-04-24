# MCV Adapter Validation Checklist

## Model Alignment

- [x] Re-read PostgreSQL histogram semantics in the code and paper workflow
- [x] Confirm that PostgreSQL stores ordered histogram bounds, not explicit per-bucket frequencies
- [x] Treat histogram bounds as describing the residual non-MCV population only
- [x] Reconstruct residual mass by assigning equal implicit mass to each adjacent interval
- [x] Decompose corrected residual distributions by resampling quantile boundaries

## Script Updates

- [x] Fix `reconstruct_full_histogram()` to expand `histogram_bounds` into equal-mass residual ranges
- [x] Fix `decompose_to_postgres_format()` to emit histogram bounds instead of bucket-frequency vectors
- [x] Keep experiments 1-3 threshold-free (`mcv_threshold=0.0`) for fidelity checks
- [x] Keep experiment 4 as the threshold-sensitivity study with simulated correction
- [x] Regenerate `experiments/results/mcv_validation_results.json`

## Figures and Paper

- [x] Regenerate `paper/figures/mcv_validation_summary.pdf`
- [x] Regenerate `paper/figures/mcv_performance_overhead.pdf`
- [x] Regenerate `paper/figures/mcv_threshold_sensitivity.pdf`
- [x] Update the histogram-conversion text in `paper/main.tex`
- [x] Update appendix notes to use histogram-bound terminology
- [x] Sync `MCV*.md` and `experiments/README_MCV_VALIDATION.md` with the new data

## Recorded Results

- [x] Round-trip total mass error = `0.0`
- [x] Round-trip MCV mass error = `0.0`
- [x] Round-trip residual mass error = `0.0`
- [x] Residual quantile MAE mean = `2.392887868050384e-17`
- [x] Residual quantile MAE max = `9.178576490712927e-17`
- [x] Selectivity global MAE = `1.0720625000010417e-04`
- [x] Selectivity global P95 = `1.0e-04`
- [x] Selectivity global P99 = `3.788937499999999e-03`
- [x] Default-like latency (`100/10`) = `0.0660048660 ms`
- [x] Default-like latency (`100/20`) = `0.0731946860 ms`
- [x] Worst tested latency (`50/50`) = `1.0270490040 ms`
- [x] Threshold sweep confirms monotonic error growth from `0.1%` to `5%`

## Validation Commands Run

- [x] `python3 -m py_compile experiments/mcv_adapter_validation.py`
- [x] `python3 -m py_compile experiments/generate_mcv_figures.py`
- [x] `python3 -m py_compile paper/appendix/mcv_adapter_interface.py`
- [x] `python3 experiments/mcv_adapter_validation.py`
- [x] `python3 experiments/generate_mcv_figures.py`
- [x] `pdflatex -interaction=nonstopmode -halt-on-error main.tex`

## Final Judgment

- [x] The old validation logic had a PostgreSQL-format modeling bug
- [x] The updated script now matches PostgreSQL histogram semantics
- [x] The new measurements are internally consistent across script, figures, and paper
- [x] The main paper claims remain valid after correction
