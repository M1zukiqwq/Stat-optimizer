# MCV Adapter Problem and Solution

## Executive Summary

The bug was not in the high-level idea of the MCV adapter. The bug was in the **validation model** used to mimic PostgreSQL statistics.

The earlier draft treated PostgreSQL histograms as if they explicitly stored bucket frequencies. PostgreSQL actually stores only an ordered list of histogram boundary values for the residual non-MCV population. Once the validation script was corrected to follow that representation, the paper text, figures, and markdown summaries had to be updated accordingly.

## What the Original Validation Got Wrong

### Incorrect assumption

The old validation effectively modeled PostgreSQL statistics as:

- MCV list with explicit frequencies, plus
- residual histogram with explicit bucket probabilities.

That is not how `pg_statistic` stores single-column histograms.

### PostgreSQL's actual layout

For the single-column case relevant here:

- MCV entries store explicit frequencies.
- Histogram data stores only sorted boundary values.
- The histogram is built **after removing MCV values**.
- Each adjacent pair of histogram bounds implicitly represents an equal share of the residual mass.

So the faithful interpretation is:

`MCV list + residual bounds -> canonical full distribution -> residual bounds`

not

`MCV list + explicit residual bucket vector -> explicit residual bucket vector`.

## Corrected Solution

### Phase 1: Reconstruction

Given:
- `mcv_list = [(v_i, f_i)]`
- `histogram_bounds = [u_0, u_1, ..., u_m]`

we compute:
- residual mass `p_res = 1 - null_fraction - sum(f_i)`
- `m` residual intervals, each with mass `p_res / m`
- a canonical full distribution made of singleton buckets for MCVs and range buckets for residual intervals

### Phase 2: Correction

OASIS continues to operate on the canonical full distribution without any change to the core correction logic.

### Phase 3: Decomposition

Given a corrected full distribution:
- extract singleton MCV candidates using `tau_MCV`
- keep the top `K` entries
- merge the remaining singletons and ranges into a residual distribution
- emit PostgreSQL-style histogram bounds by sampling residual quantiles

This is the important change: decomposition now writes **histogram bounds**, not explicit bucket masses.

## Experiment Design After the Fix

### Experiments 1-3

These experiments validate translation fidelity and overhead, so they use `mcv_threshold=0.0` during round-trip decomposition. That removes threshold-induced MCV migration from the fidelity test.

### Experiment 4

This experiment is the dedicated threshold study. It first applies a simulated correction and then varies `tau_MCV` to show how threshold choices affect the final PostgreSQL decomposition.

## Current Results

### Round-trip fidelity

- total mass error: `0.0`
- MCV mass error: `0.0`
- residual mass error: `0.0`
- residual quantile MAE mean: `2.392887868050384e-17`
- residual quantile MAE max: `9.178576490712927e-17`

### Selectivity consistency

- global MAE: `1.0720625000010417e-04`
- global P95: `1.0e-04`
- global P99: `3.788937499999999e-03`
- global max: `3.8387500000000006e-03`

### Performance

- `100 MCV / 10 bins`: `0.0660048660 ms`
- `100 MCV / 20 bins`: `0.0731946860 ms`
- `100 MCV / 50 bins`: `0.0641457660 ms`
- worst tested (`50 MCV / 50 bins`): `1.0270490040 ms`

### Threshold sweep

- `0.1%` and `0.5%`: `20` MCVs, MAE `0.0010266594`
- `1%`: `17` MCVs, MAE `0.0015147205`
- `2%`: `9` MCVs, MAE `0.0029720909`
- `5%`: `3` MCVs, MAE `0.0045435225`

## Impact on the Paper

The paper's main claims remain intact:

- the adapter preserves mass exactly in translation,
- the conversion overhead is very small for PostgreSQL-like settings,
- threshold choice mainly affects post-correction decomposition quality.

What changed is the wording and evidence for the histogram conversion subsection before the end-to-end experiments:

- it now describes PostgreSQL histograms as ordered residual bounds,
- it explains the implicit equal-mass interpretation,
- it reports the corrected numbers produced by the new validation script.

## Files That Matter Most

- `experiments/mcv_adapter_validation.py`
- `experiments/generate_mcv_figures.py`
- `paper/main.tex`
- `paper/appendix/mcv_adapter.tex`
- `paper/appendix/mcv_adapter_interface.py`
- `experiments/results/mcv_validation_results.json`

## Bottom Line

Yes — there was a bug in the earlier validation workflow. It came from using the wrong PostgreSQL histogram model. That bug has now been corrected, the data were regenerated, and the paper text has been synchronized to the corrected workflow.
