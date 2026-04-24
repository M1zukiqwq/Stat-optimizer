# Figure Notes for the Corrected MCV Validation

## Purpose

This note explains the **current** figure set used by the corrected PostgreSQL MCV adapter workflow.

It replaces the older discussion that referred to interpolated performance curves and a removed standalone round-trip figure.

## Current Figure Set

### 1. `paper/figures/mcv_validation_summary.pdf`

This is the main paper figure for the MCV adapter section.

It contains three panels:

- **(a) Round-trip fidelity**: residual-quantile MAE by distribution
- **(b) Performance overhead**: representative round-trip latency by MCV count
- **(c) Threshold sensitivity**: selectivity MAE as `tau_MCV` increases

This figure is the one referenced in `paper/main.tex` for the pre-end-to-end MCV validation subsection.

### 2. `paper/figures/mcv_performance_overhead.pdf`

This figure shows the measured round-trip latency curves for:

- `MCV=10`
- `MCV=50`
- `MCV=100`

across residual bin counts `{10, 20, 50}`.

Important note: the current script plots **measured points only**. It does **not** add interpolated bucket counts.

### 3. `paper/figures/mcv_threshold_sensitivity.pdf`

This figure shows:

- how many singleton masses remain in the MCV list after decomposition, and
- how selectivity MAE changes as the MCV threshold increases.

It isolates the effect of `tau_MCV` after a simulated correction step.

## Design Rationale

### Why the round-trip panel now uses residual-quantile MAE

The corrected workflow validates PostgreSQL translation fidelity through the residual quantile representation, because PostgreSQL histograms are stored as ordered residual bounds rather than explicit bucket-frequency vectors.

Therefore, the key fidelity metric is now:

- exact total mass preservation, and
- machine-precision residual-quantile reconstruction error.

This is a better fit than the old component-frequency view.

### Why there is no standalone `mcv_roundtrip_accuracy.pdf`

The old standalone figure was based on the earlier bucket-frequency interpretation and is no longer the right visual summary.

After the correction, the combined summary figure already captures the round-trip result in the correct form, so a separate standalone figure is unnecessary.

### Why panel (b) uses representative points in the combined figure

The combined summary figure is meant to stay compact inside the paper.

So panel (b) uses one representative latency per MCV count, preferring the `20-bin` configuration when available. The detailed multi-line view remains available in `mcv_performance_overhead.pdf`.

## Numbers Reflected by the Current Figures

### Round-trip fidelity

- residual quantile MAE mean: `2.392887868050384e-17`
- residual quantile MAE max: `9.178576490712927e-17`
- total mass error max: `0.0`

### Performance

- `100 / 10`: `0.0660048660 ms`
- `100 / 20`: `0.0731946860 ms`
- `100 / 50`: `0.0641457660 ms`
- worst tested `50 / 50`: `1.0270490040 ms`

### Threshold sensitivity

- `0.1%`: MAE `0.0010266594`, `20` MCVs
- `0.5%`: MAE `0.0010266594`, `20` MCVs
- `1%`: MAE `0.0015147205`, `17` MCVs
- `2%`: MAE `0.0029720909`, `9` MCVs
- `5%`: MAE `0.0045435225`, `3` MCVs

## Source of Truth

The figure script reads from:

- `experiments/results/mcv_validation_results.json`

If figure captions, markdown notes, and JSON disagree, treat the JSON and the regenerated PDFs as correct.

## Regeneration Commands

```bash
cd experiments
python3 mcv_adapter_validation.py
python3 generate_mcv_figures.py
```

## Bottom Line

The current figure set is intentionally simple and measurement-driven:

- no interpolated performance points,
- no obsolete standalone round-trip figure,
- one combined summary figure for the paper,
- two supporting figures for detail.
