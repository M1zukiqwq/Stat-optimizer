# Task Plan

Goal: understand the Stat-optimizer project, survey algorithms that estimate multi-column statistics from single-column statistics, run the existing ablation experiments, then implement and evaluate a practical end-to-end experiment using corrected single-column statistics to influence multi-column estimates.

## Phases

| Phase | Status | Notes |
|---|---|---|
| 1. Project orientation | complete | Read paper, docs, and experiment code structure. |
| 2. Algorithm survey | complete | Search for methods related to inferring multi-column stats from single-column stats. |
| 3. Existing ablation | complete | Main synthetic ablation rerun completed under `experiments/results/synthetic_paper_suite_rerun_20260529`. |
| 4. End-to-end design | complete | Add a marginal-propagation E2E: corrected single-column marginals feed independence/max-entropy and Gaussian-copula joint estimators. |
| 5. Implementation | complete | Updated `experiments/copula_oasis_experiment.py` with marginal-propagation E2E estimators and summaries. |
| 6. Verification and conclusion | complete | `py_compile` passed; ablation and marginal-to-joint E2E summaries generated. |
| 7. Attribution and hybrid E2E | complete | Added marginal-error attribution and feedback-residual hybrid routing; formal results written under `experiments/results/marginal_joint_hybrid_e2e_20260529`. |
| 8. OASIS E2E diagnosis and targeted improvement | complete | Diagnosed weak marginal feedback-consistency and tested OASIS-Proj, which recovers most E2E loss. |
| 9. Real DBMS planner evidence | complete | Built local PostgreSQL and ran planner-only stats injection using `COUNT(*)` ground truth plus `EXPLAIN (FORMAT JSON)` plans; formal results written under `experiments/results/postgres_planner_stats_injection_20260529`. |
| 10. Multi-config PostgreSQL planner batch | complete | Extended the planner-only script with batch mode and ran 12 PostgreSQL configurations under `experiments/results/postgres_planner_stats_injection_batch_20260529`; the Information Systems draft now uses the batch table. |
| 11. Reviewer-driven safety diagnostics | complete | Added plan-change breakdown, feedback-budget sensitivity, and feedback-noise robustness diagnostics; updated `paper/main_is.tex` and supplementary materials with the new evidence. |
| 12. OOD drift realism | complete | Added and ran an out-of-distribution drift suite over batch load, range shift, skew evolution, outlier burst, multimodal, and seasonal/mixed drift; narrowed statistics-adapter claims to PostgreSQL-style validation. |
| 13. Information Systems realism/safety补强 | complete | Added a benchmark-inspired DML trace sanity check and a deployment-safety summary table; updated `paper/main_is.tex`, `paper/supplementary.tex`, and paper figures. |

## Decisions

| Decision | Rationale |
|---|---|
| Use additive experiment code where possible | Preserve existing paper/experiment contracts and avoid broad refactors. |
| Treat single-column-only multi-column estimation as maximum-entropy/independence | Exact joint stats are not identifiable from marginals alone; independence is the defensible no-extra-signal baseline. |
| Also evaluate a fixed-dependence copula path | Shows how corrected marginals help a real multi-column estimator when dependence metadata/model is already present. |
| Hybrid policy uses only feedback-window residuals | Avoids oracle/fresh-statistics leakage while testing whether deployment gating can prevent OASIS regressions. |
| Use PostgreSQL planner-only evidence instead of runtime | Avoids overclaiming end-to-end latency; directly tests whether corrected single-column statistics alter a real DBMS optimizer's estimates and plan choices. |

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
