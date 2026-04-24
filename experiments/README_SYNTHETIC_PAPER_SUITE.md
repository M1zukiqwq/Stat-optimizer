# Synthetic Paper Suite

`experiments/run_synthetic_paper_suite.py` is the clean synthetic experiment entrypoint after the 2026-03-11 cleanup.

It replaces multiple overlapping runners with one unified workflow:
- retrain the main OASIS model on `q in {1,3,5,10,15,20}`;
- rerun the main compound-drift comparison against `Prior`, `STHoles`, `QuickSel-H`, `ISOMER`, and `OASIS`;
- optionally rerun observation-window sensitivity;
- optionally rerun initial-distribution generalization.

Key guarantees compared with the archived scripts:
- no hard-coded absolute paths to another repository;
- no hard-coded paper values in plotting code;
- all methods on a case share the same sampled evaluation points;
- outputs are written under `experiments/results/synthetic_paper_suite/`.

Coverage after the 2026-03-11 cleanup:
- the unified runner now covers every remaining paper-facing synthetic suite after the cleanup: `main`, `sensitivity`, and `distribution`;
- the archived `run_drift_pattern_experiments.py` is intentionally not ported because its extended-pattern setup was removed from the active paper narrative and was one source of conflicting numbers;
- `experiments/mcv_adapter_validation.py` and the live `job_experiment/` workflows remain separate because they validate the DBMS interface layer and end-to-end DB behavior rather than the synthetic single-column model;
- `SCD Type 2` / `Fact Table Growth` should be evaluated with the real `TPC-DS` workflow under `tpcds_experiment/`, not with the synthetic suite.


Example:
- `python3 experiments/run_synthetic_paper_suite.py --suites main distribution --output-root experiments/results/synthetic_paper_suite_rerun`
- Use `--stholes-mode tree` to replace the flat `STHoles` baseline with the exploratory hierarchical hole-tree variant during a rerun.
