# MCV Adapter Implementation Summary

## Overview

This summary reflects the corrected PostgreSQL model used by the validation and paper update on 2026-03-11.

The key fix is that PostgreSQL single-column histograms do **not** store explicit bucket frequencies. They store an ordered list of histogram boundary values for the **non-MCV residual population**. Each interval between adjacent bounds therefore carries an equal share of the residual mass implicitly.

## What Changed

1. `experiments/mcv_adapter_validation.py` now reconstructs a canonical full distribution from:
   - explicit MCV entries, and
   - residual histogram bounds with equal residual mass per interval.
2. Decomposition now converts the corrected residual distribution back into PostgreSQL-style histogram bounds by sampling residual quantiles.
3. Experiments 1-3 use `mcv_threshold=0.0` during round-trip validation so that translation fidelity is evaluated independently of threshold tuning.
4. Experiment 4 isolates `tau_MCV` sensitivity by applying a simulated correction first and then varying the decomposition threshold.
5. `paper/main.tex` and the appendix now describe PostgreSQL histograms as ordered bounds rather than explicit bucket-frequency vectors.

## Files Updated

- `experiments/mcv_adapter_validation.py`
- `experiments/generate_mcv_figures.py`
- `paper/main.tex`
- `paper/appendix/mcv_adapter.tex`
- `paper/appendix/mcv_adapter_interface.py`
- `paper/figures/mcv_validation_summary.pdf`
- `paper/figures/mcv_performance_overhead.pdf`
- `paper/figures/mcv_threshold_sensitivity.pdf`

## Current Results

### 1. Round-Trip Translation Fidelity

- Total non-null mass error: `0.0`
- MCV mass error: `0.0`
- Residual mass error: `0.0`
- Residual quantile MAE (mean): `2.392887868050384e-17`
- Residual quantile MAE (max): `9.178576490712927e-17`

Interpretation: reconstruction and decomposition are exact for the tested synthetic PostgreSQL statistics once threshold effects are removed from the round-trip check.

### 2. Selectivity Consistency

- Global MAE: `1.0720625000010417e-04`
- Global P95: `1.0e-04`
- Global P99: `3.788937499999999e-03`
- Global max: `3.8387500000000006e-03`

Interpretation: predicate estimates remain close after translation, with small tail errors concentrated around discrete/high-skew regions.

### 3. Conversion Overhead

- PostgreSQL-like `100 MCV / 10 bins`: `0.0660048660 ms`
- PostgreSQL-like `100 MCV / 20 bins`: `0.0731946860 ms`
- PostgreSQL-like `100 MCV / 50 bins`: `0.0641457660 ms`
- Worst tested configuration `50 MCV / 50 bins`: `1.0270490040 ms`

Interpretation: the default-like PostgreSQL configurations remain comfortably sub-millisecond; only the densest tested synthetic setting crosses 1 ms.

### 4. Threshold Sensitivity

| Threshold | MCV Count | Selectivity MAE | P99 Error |
|-----------|-----------|-----------------|-----------|
| `0.1%`    | `20`      | `0.0010266594`  | `0.0038621985` |
| `0.5%`    | `20`      | `0.0010266594`  | `0.0038621985` |
| `1%`      | `17`      | `0.0015147205`  | `0.0061021000` |
| `2%`      | `9`       | `0.0029720909`  | `0.0138112435` |
| `5%`      | `3`       | `0.0045435225`  | `0.0195120508` |

Interpretation: lower thresholds preserve more singleton mass as MCV entries; higher thresholds compress more of the distribution back into the residual histogram and therefore degrade equality-heavy selectivity estimates.

## Main Takeaways

- The previous validation bug was in the **model of PostgreSQL histograms**, not in the overall adapter idea.
- The paper's main qualitative claims still hold:
  - exact or near-exact round-trip translation,
  - low adapter overhead,
  - graceful degradation under more aggressive MCV thresholds.
- The paper's emphasis remains on PostgreSQL-like settings (`K=100`, 10-20 residual bins, `tau≈1%`), with updated numbers now grounded in a faithful PostgreSQL representation.

## Reproduction

```bash
cd postgresql-14.17/postgres-cdf-simulation/experiments
python3 mcv_adapter_validation.py
python3 generate_mcv_figures.py
```

Results are written to `experiments/results/mcv_validation_results.json` and figures are written to `paper/figures/`.

## Status

- Validation script corrected
- New results generated
- Paper text updated
- Figures regenerated
- LaTeX build passes
