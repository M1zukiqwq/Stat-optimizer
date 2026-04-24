# MCV Adapter Verification Report

## Scope

This report verifies whether the PostgreSQL MCV adapter description, validation script, and paper text are mutually consistent after correcting the histogram model.

## Finding

**Conclusion:** the earlier validation draft contained a bug.

The bug was that PostgreSQL histograms were modeled as explicit bucket-frequency vectors. The corrected implementation now matches PostgreSQL's actual representation: ordered histogram bounds over the residual non-MCV distribution.

## What Was Checked

### Code

- `experiments/mcv_adapter_validation.py`
- `experiments/generate_mcv_figures.py`
- `paper/appendix/mcv_adapter_interface.py`

### Paper text

- `paper/main.tex`
- `paper/appendix/mcv_adapter.tex`

### Derived artifacts

- `experiments/results/mcv_validation_results.json`
- `paper/figures/mcv_validation_summary.pdf`
- `paper/figures/mcv_performance_overhead.pdf`
- `paper/figures/mcv_threshold_sensitivity.pdf`

## Correct PostgreSQL Interpretation

For the single-column statistics path used here:

1. `mcv_list` stores explicit frequencies.
2. `histogram_bounds` stores ordered residual sample boundaries.
3. Consecutive bounds define equi-depth residual intervals.
4. Residual interval mass is implicit and equals:

   `p_res / (len(histogram_bounds) - 1)`

   where `p_res = 1 - null_fraction - sum(mcv frequencies)`.

## Code-Level Resolution

### Reconstruction

`reconstruct_full_histogram()` now:
- materializes each MCV entry as a singleton bucket,
- computes residual mass from the non-MCV remainder,
- expands adjacent histogram bounds into equal-mass residual range buckets.

### Decomposition

`decompose_to_postgres_format()` now:
- selects MCV candidates from singleton buckets,
- merges the remaining singletons and ranges into a residual distribution,
- writes PostgreSQL histogram bounds by sampling residual quantiles.

### Experiment separation

- Experiments 1-3 validate translation fidelity and overhead using threshold-free round trips.
- Experiment 4 isolates `tau_MCV` sensitivity after a simulated correction.

## Measured Results

### Round-trip fidelity

- total mass error max = `0.0`
- MCV mass error max = `0.0`
- residual mass error max = `0.0`
- residual quantile MAE mean = `2.392887868050384e-17`
- residual quantile MAE max = `9.178576490712927e-17`

### Selectivity consistency

- global MAE = `1.0720625000010417e-04`
- global P95 = `1.0e-04`
- global P99 = `3.788937499999999e-03`
- global max = `3.8387500000000006e-03`

### Performance overhead

- `100/10`: `0.0660048660 ms`
- `100/20`: `0.0731946860 ms`
- `100/50`: `0.0641457660 ms`
- worst tested `50/50`: `1.0270490040 ms`

### Threshold sensitivity

| Threshold | MCV Count | MAE | P99 |
|-----------|-----------|-----|-----|
| `0.1%` | `20` | `0.0010266594` | `0.0038621985` |
| `0.5%` | `20` | `0.0010266594` | `0.0038621985` |
| `1%`   | `17` | `0.0015147205` | `0.0061021000` |
| `2%`   | `9`  | `0.0029720909` | `0.0138112435` |
| `5%`   | `3`  | `0.0045435225` | `0.0195120508` |

## Paper Consistency Check

The paper is now consistent with the corrected script in the histogram-conversion subsection:

- PostgreSQL histograms are described as ordered residual bounds.
- Reconstruction is phrased as conversion to a canonical quantile/CDF view.
- Decomposition is phrased as resampling back to histogram bounds.
- The validation section reports the regenerated metrics rather than the old component-frequency numbers.

## Validation Commands

The following commands were run successfully during verification:

```bash
python3 -m py_compile experiments/mcv_adapter_validation.py
python3 -m py_compile experiments/generate_mcv_figures.py
python3 -m py_compile paper/appendix/mcv_adapter_interface.py
python3 experiments/mcv_adapter_validation.py
python3 experiments/generate_mcv_figures.py
cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Verdict

The corrected workflow is internally consistent.

- The bug in the old validation has been fixed.
- The new data have been generated.
- The paper text and figures now match the corrected PostgreSQL model.
