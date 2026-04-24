# MCV Adapter Validation Experiments

This directory contains the validation workflow for the PostgreSQL MCV adapter described in Section 3.5 and evaluated before the end-to-end experiments in the paper.

## Validation Model

The experiments now match PostgreSQL's actual single-column statistics layout:

1. The MCV list stores explicit value frequencies.
2. The histogram stores only ordered boundary values for the **non-MCV residual population**.
3. Consecutive boundary values define equal-depth residual intervals.
4. The adapter reconstructs a canonical full distribution from this compressed representation, then decomposes it back to PostgreSQL format.

This is the important correction relative to the earlier validation draft, which incorrectly treated PostgreSQL histograms as explicit bucket-frequency vectors.

## Files

- `mcv_adapter_validation.py`: runs the four validation experiments and writes JSON results
- `generate_mcv_figures.py`: regenerates the paper figures from the JSON results
- `results/mcv_validation_results.json`: latest experiment output

## Running the Experiments

### Prerequisites

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib
```

### Run Validation

```bash
cd experiments
python3 mcv_adapter_validation.py
```

### Generate Figures

```bash
python3 generate_mcv_figures.py
```

Figures are written to `../paper/figures/`.

## Experiment Design

### Experiment 1: Round-Trip Fidelity

- Input: PostgreSQL-style statistics generated from synthetic data
- Pipeline: `pg_stats -> full distribution -> pg_stats`
- Decomposition setting: `mcv_threshold=0.0`
- Goal: isolate translation fidelity without threshold-induced MCV reclassification

**Current summary:**
- total mass error max = `0.0`
- MCV mass error max = `0.0`
- residual quantile MAE mean = `2.392887868050384e-17`
- residual quantile MAE max = `9.178576490712927e-17`

### Experiment 2: Selectivity Consistency

- Input: translated PostgreSQL statistics and reconstructed full histogram
- Predicates: equality, range `<`, and `BETWEEN`
- Goal: compare selectivity estimates before and after translation

**Current summary:**
- global MAE = `1.0720625000010417e-04`
- global P95 = `1.0e-04`
- global P99 = `3.788937499999999e-03`
- global max = `3.8387500000000006e-03`

### Experiment 3: Performance Overhead

- Configurations: `n_mcv ∈ {10, 50, 100}` × `n_buckets ∈ {10, 20, 50}`
- Goal: measure reconstruction and decomposition latency

**Current summary:**
- `100 MCV / 10 bins`: `0.0660048660 ms`
- `100 MCV / 20 bins`: `0.0731946860 ms`
- `100 MCV / 50 bins`: `0.0641457660 ms`
- worst tested (`50 MCV / 50 bins`): `1.0270490040 ms`

### Experiment 4: Threshold Sensitivity

- Base distribution: corrected Zipfian full histogram
- Thresholds: `0.1%`, `0.5%`, `1%`, `2%`, `5%`
- Goal: isolate how `tau_MCV` changes the decomposition quality after correction

**Current summary:**

| Threshold | MCV Count | MAE | P99 |
|-----------|-----------|-----|-----|
| `0.1%` | `20` | `0.0010266594` | `0.0038621985` |
| `0.5%` | `20` | `0.0010266594` | `0.0038621985` |
| `1%` | `17` | `0.0015147205` | `0.0061021000` |
| `2%` | `9`  | `0.0029720909` | `0.0138112435` |
| `5%` | `3`  | `0.0045435225` | `0.0195120508` |

## Interpretation

- The corrected adapter is exact for round-trip translation under threshold-free decomposition.
- Residual quantile reconstruction is effectively machine-precision accurate for the tested synthetic statistics.
- Default-like PostgreSQL settings remain sub-millisecond.
- Threshold choice primarily matters **after correction**, when deciding how much mass to keep as explicit MCV entries.

## Reproducing the Paper State

```bash
cd ../paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The paper uses:
- `mcv_validation_summary.pdf`
- `mcv_performance_overhead.pdf`
- `mcv_threshold_sensitivity.pdf`

## Notes

- `results/mcv_validation_results.json` is the authoritative source for the numbers copied into the paper and markdown summaries.
- The focus of the validation is adapter fidelity and overhead; the end-to-end OASIS experiments remain unchanged.
