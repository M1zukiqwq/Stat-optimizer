# OASIS Paper Integrated Handoff

## What This Document Is For

This is the **single orientation document** that future models should read first before touching the OASIS paper, the PostgreSQL statistics-format interface validation, or the related markdown summaries.

It integrates:

- the paper's main story,
- the repository locations that implement each part,
- the statistics-format modeling bug that was found and fixed,
- the current validated numbers,
- the rules for future edits.

If this document conflicts with an older markdown note, prefer this document, `paper/main.tex`, and the current experiment outputs.

## One-Paragraph Summary

OASIS is a feedback-driven system that repairs stale column statistics for cost-based query optimizers without rescanning tables and without changing the optimizer's core logic. Its central idea is to treat histogram correction as a regression problem over stale statistics plus execution feedback, implemented with a lightweight attention-pooled MLP. The paper's main evidence is: up to `62.0%` Q-Error reduction on synthetic drift workloads, `44.4%--60.8%` Q-Error reduction across unseen initial distributions at `q=10`, recovery of `52.0%` of Full ANALYZE's execution-time gain in a PostgreSQL 14 / TPC-DS end-to-end study, and low integration overhead. A key portability contribution is a three-phase statistics-format conversion interface that converts heterogeneous engine statistics into a canonical full distribution and back again; the paper instantiates and validates this interface on PostgreSQL-style `MCV + residual histogram bounds`.

## Repository Map

### Primary paper files

- `paper/main.tex`: authoritative paper source
- `paper/main.pdf`: latest compiled paper
- `paper/references.bib`: bibliography

### Statistics-format interface appendix and support

- `paper/appendix/mcv_adapter.tex`: formal portability/interface appendix text
- `paper/appendix/system_details.tex`: model and instrumentation appendix details
- `paper/appendix/mcv_adapter_interface.py`: abstract interface sketch
- `paper/appendix/README.md`: appendix overview

### Unified synthetic suite

- `experiments/run_synthetic_paper_suite.py`: authoritative runner for the cleaned synthetic paper suites
- covered suites: `main`, `sensitivity`, `distribution`
- authoritative clean rerun root for the current paper text: `experiments/results/synthetic_paper_suite_tree_isomer_v3/`
- archived but intentionally not revived: `legacy_experiments_backup/2026-03-11/cdf_kll_ml_pipeline/run_drift_pattern_experiments.py`

### PostgreSQL interface experiments

- `experiments/mcv_adapter_validation.py`: authoritative validation script
- `experiments/generate_mcv_figures.py`: figure generator
- `experiments/results/mcv_validation_results.json`: authoritative numeric output

### Generated MCV figures

- `paper/figures/mcv_validation_summary.pdf`
- `paper/figures/mcv_performance_overhead.pdf`
- `paper/figures/mcv_threshold_sensitivity.pdf`

### Supporting markdown

- `MCV_PROBLEM_AND_SOLUTION.md`
- `MCV_VERIFICATION_REPORT.md`
- `MCV_ADAPTER_SUMMARY.md`
- `MCV_ADAPTER_CHECKLIST.md`
- `FINAL_SUMMARY.md`
- `FIGURE_IMPROVEMENTS.md`
- `experiments/README_MCV_VALIDATION.md`

## Paper Structure

### 1. Introduction

Core claims introduced in the paper:

- stale column statistics create a persistent optimizer staleness gap;
- OASIS corrects statistics from implicit query feedback;
- OASIS is non-invasive and deploys through lightweight hooks;
- a single pre-trained `38K`-parameter checkpoint supports zero-shot deployment;
- a statistics-format conversion interface extends the design beyond standalone histograms to tightly coupled engines such as PostgreSQL.

### 2. Background and Problem Formulation

This section explains:

- equi-depth histograms and the staleness gap,
- selectivity feedback,
- why statistics-level correction differs from plan-level feedback,
- the formal regression problem that OASIS solves.

### 3. System Design

This section contains:

- system overview,
- feature representation,
- attention-pooled MLP design,
- feedback collection architecture,
- overhead analysis,
- non-invasive integration,
- statistics-format conversion interface design.

### 4. Experimental Evaluation

The current experiment flow is intentionally staged in the following order:

1. **Experimental Setup**
2. **Statistics-Format Interface Validation**
3. **Main Results: Q-Error vs. Drift Intensity**
4. **Structural Accuracy: Quantile and Selectivity Error**
5. **Observation Window Ablation**
6. **Initial-Distribution Generalization**
7. **End-to-End Evaluation on TPC-DS**

This order matters: it moves from portability validation, to controlled single-column accuracy, to structural/ablation/generalization evidence, and only then to systems payoff.
The synthetic `SCD Type 2` / `Fact Table Growth` transfer suites have been removed from the paper-facing workflow because those patterns should be evaluated on the real `TPC-DS` pipeline under `tpcds_experiment/`, not with synthetic surrogates.
The appendix snippets are not compiled into the current 17-page submission by default, so consistency is maintained by keeping their prose aligned with the live result files rather than by cross-references inside `main.tex`.
The statistics-format interface subsection should not be allowed to drift away from the experiment JSON and figure scripts.

### 5. Related Work and Conclusion

These sections position OASIS against classical histogram tuning, feedback-based methods, and learned cardinality estimation / learned optimization work.

## The Most Important Paper Claims to Preserve

Future edits should preserve the following central claims unless new experiments explicitly replace them:

- up to `62.0%` Q-Error reduction over the stale prior on synthetic drift,
- `44.4%--60.8%` Q-Error reduction across unseen initial distributions at `q=10`,
- zero-shot deployment from one pre-trained checkpoint,
- OASIS recovers `52.0%` of Full ANALYZE's end-to-end execution-time gain on PostgreSQL 14 / TPC-DS,
- a statistics-format conversion interface that makes histogram correction portable across tightly coupled statistics architectures.

## The Statistics-Format Interface: Correct Mental Model

### The old mistake

The earlier validation workflow mistakenly treated PostgreSQL histograms as if they stored explicit bucket probabilities.

### The correct PostgreSQL model

For the single-column statistics path relevant to this paper:

- the MCV list stores explicit frequencies,
- the histogram stores **ordered residual boundary values**,
- those bounds are built after removing MCV values,
- each adjacent pair of bounds implicitly represents an equal share of the residual mass.

This means the right abstraction is:

`PostgreSQL statistics -> canonical full distribution -> PostgreSQL statistics`

not

`MCV list + explicit residual bucket vector -> explicit residual bucket vector`.

## Three-Phase Conversion Pipeline

### Phase 1: Reconstruction

Input:

- `mcv_list = [(value, frequency)]`
- `histogram_bounds = [u_0, ..., u_m]`

Behavior:

- convert each MCV entry into a singleton bucket,
- compute residual mass `1 - null_fraction - sum(mcv frequencies)`,
- expand adjacent histogram bounds into equal-mass residual range buckets,
- produce the canonical OASIS-facing full distribution.

### Phase 2: OASIS Correction

Behavior:

- run the normal OASIS correction logic on the canonical full distribution,
- do **not** bake PostgreSQL-specific logic into the core correction model.

### Phase 3: Decomposition

Behavior:

- extract singleton MCV candidates using `tau_MCV`,
- keep the top `K` entries,
- merge remaining singletons and residual ranges into a residual distribution,
- emit PostgreSQL `histogram_bounds` by sampling residual quantiles.

## Why the Validation Design Looks the Way It Does

### Experiments 1-3 are threshold-free

Round-trip fidelity should test format translation, not threshold policy.

So experiments 1-3 use `mcv_threshold=0.0` during decomposition to remove threshold-induced MCV migration from the fidelity check.

### Experiment 4 isolates threshold choice

Threshold sensitivity matters after correction.

So experiment 4 first applies a simulated correction to the canonical distribution, then varies `tau_MCV` to measure the trade-off between explicit MCV retention and residual compression.

## Current Authoritative MCV Results

These values come from `experiments/results/mcv_validation_results.json` and should be treated as the exact source for the MCV subsection.

### Round-trip fidelity

- total mass error max: `0.0`
- MCV mass error max: `0.0`
- residual mass error max: `0.0`
- residual quantile MAE mean: `2.392887868050384e-17`
- residual quantile MAE max: `9.178576490712927e-17`

### Selectivity consistency

- global MAE: `1.0720625000010417e-04`
- global P95: `1.0e-04`
- global P99: `3.788937499999999e-03`
- global max: `3.8387500000000006e-03`

### Performance overhead

- `100 / 10`: `0.0660048660 ms`
- `100 / 20`: `0.0731946860 ms`
- `100 / 50`: `0.0641457660 ms`
- worst tested `50 / 50`: `1.0270490040 ms`

### Threshold sensitivity

| Threshold | MCV Count | MAE | P99 |
|-----------|-----------|-----|-----|
| `0.1%` | `20` | `0.0010266594` | `0.0038621985` |
| `0.5%` | `20` | `0.0010266594` | `0.0038621985` |
| `1%`   | `17` | `0.0015147205` | `0.0061021000` |
| `2%`   | `9`  | `0.0029720909` | `0.0138112435` |
| `5%`   | `3`  | `0.0045435225` | `0.0195120508` |

## Authoritative Synthetic Suite

The clean synthetic-paper results now come from the unified runner and its official outputs:

- `experiments/run_synthetic_paper_suite.py`: clean synthetic experiment entrypoint after the 2026-03-11 cleanup
- `experiments/results/synthetic_paper_suite_tree_isomer_v3/main/summary.json`: authoritative main Q-Error and structural-metric numbers
- `experiments/results/synthetic_paper_suite_tree_isomer_v3/distribution/summary.json`: authoritative initial-distribution generalization numbers
- `experiments/results/synthetic_paper_suite_tree_isomer_v3/sensitivity/summary.json`: fresh sensitivity rerun; keep it for reference, but the paper intentionally retains the `K=16` narrative by author choice
- `legacy_experiments_backup/2026-03-11/`: archived conflicting synthetic runners and outputs that should no longer feed paper tables

## Current Figure Semantics

### `mcv_validation_summary.pdf`

This is the paper-facing summary figure.

- panel (a): residual-quantile fidelity,
- panel (b): representative performance points,
- panel (c): threshold sensitivity.

### `mcv_performance_overhead.pdf`

This shows the full measured latency curves for the tested MCV/bin combinations.

### `mcv_threshold_sensitivity.pdf`

This shows how `tau_MCV` changes MCV count and selectivity error.

### Important note

The current figure generator uses **measured points only**. Older notes about interpolated extra bucket counts are obsolete.

## Current Main Paper Results Outside the MCV Section

These come from `paper/main.tex` and are the primary high-level results of the paper.

### Synthetic drift results

- the paper now takes synthetic numbers from `experiments/results/synthetic_paper_suite_tree_isomer_v3/main/summary.json`,
- OASIS achieves up to `62.0%` Q-Error reduction over the stale prior,
- at `q=1` and `q=3`, ISOMER is marginally better; from `q>=5`, OASIS is best at every tested drift level,
- the old cross-pattern drift figure is no longer part of the active paper because its legacy runner was archived during the 2026-03-11 cleanup.

### Sensitivity analysis

- the paper keeps the `K=16` operating-point narrative by explicit author choice,
- a fresh rerun exists in `experiments/results/synthetic_paper_suite_tree_isomer_v3/sensitivity/summary.json`, but it is reference-only unless the author decides to rewrite that subsection.

### End-to-end TPC-DS results

- OASIS reduces total execution time from `364.2s` to `339.3s` on PostgreSQL 14 / TPC-DS,
- full `ANALYZE` reaches `316.3s`, so OASIS recovers `52.0%` of the total available gain over the stale baseline,
- these results are reported in terms of execution time rather than workload-level aggregate Q-Error,
- the drift is injected through SCD Type 2 growth on `item` and `customer` plus synchronized `store_sales` growth, rather than through synthetic single-column distribution reversal.

## Rules for Future Models

### Rule 1: Use the paper as the source of narrative truth

If you are updating wording, framing, contributions, or the paper's main claims, start from `paper/main.tex`.

### Rule 2: Use JSON as the source of exact MCV numbers

If you are updating the MCV validation subsection, figure captions, or markdown summaries, use `experiments/results/mcv_validation_results.json`.

### Rule 3: Do not revert to the old PostgreSQL bucket-frequency interpretation

The corrected workflow depends on `histogram_bounds` meaning residual boundary values, not explicit per-bucket probabilities.

### Rule 4: Keep the MCV section scoped narrowly

The statistics-format interface subsection exists to validate conversion fidelity and overhead before the end-to-end experiments. It should not silently change the paper's main experimental claims.

### Rule 5: Expect mild timing jitter

Latency numbers may shift slightly between runs. When updating prose, prefer short ranges for typical settings and avoid overclaiming precision beyond what the latest JSON supports.

## Recommended Reading Order for a Future Model

1. `PAPER_INTEGRATED_HANDOFF.md`
2. `paper/main.tex`
3. `experiments/mcv_adapter_validation.py`
4. `experiments/results/mcv_validation_results.json`
5. `experiments/generate_mcv_figures.py`
6. `paper/appendix/mcv_adapter.tex`
7. `paper/appendix/system_details.tex`
8. `paper/appendix/mcv_adapter_interface.py`

## Minimal Reproduction Commands

```bash
cd experiments
python3 mcv_adapter_validation.py
python3 generate_mcv_figures.py
cd ../paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Final Takeaway

For future work, the safest mental model is:

- **OASIS paper claims live in `paper/main.tex`.**
- **MCV exact numbers live in the validation JSON.**
- **PostgreSQL histograms are residual boundary lists, not bucket-frequency arrays.**
- **This handoff document is the fastest way to regain context without rereading the entire repo.**
