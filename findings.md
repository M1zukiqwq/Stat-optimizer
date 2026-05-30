# Findings

## Project

- Paper title: OASIS, a feedback-driven single-column histogram/statistics correction system for cost-based optimizers.
- Core claim: repair stale single-column histograms from query feedback without rescanning tables or changing the optimizer.
- Existing limitation explicitly stated in paper/supplement: OASIS does not directly repair multi-column correlations or join cardinality errors; it is positioned as complementary to multi-column estimators.
- Existing E2E evidence is mostly counterfactual selectivity simulation plus optimizer-only PostgreSQL `EXPLAIN` plan-sensitivity checks.
- Repository already contains `experiments/copula_oasis_experiment.py`, which evaluates whether OASIS-improved single-column marginals improve a copula-style multi-column estimator.

## Algorithms

- From the survey, exact multi-column statistics cannot be identified from single-column marginals alone; a dependence assumption or extra signal is required.
- Maximum entropy is the principled default: with no cross-column constraints, it collapses to uniformity/independence assumptions. With feedback or partial multivariate constraints, it can produce a consistent joint estimate.
- ISOMER (ICDE 2006) builds multidimensional histograms from query feedback using maximum entropy and consistency with observed predicate cardinalities.
- STHoles builds multidimensional workload-aware histograms without scanning data, using query-result feedback to allocate nested buckets.
- PostgreSQL extended statistics confirm the practical DBMS boundary: normal per-column stats cannot capture cross-column correlation; multivariate stats require `ANALYZE` samples for dependencies, n-distinct, or multivariate MCV lists.
- Copula methods are a natural composition layer because they combine univariate marginals with a dependence model to estimate a joint CDF. Recent CoLSE uses copula theory plus a residual model for single-table cardinality estimation.
- For this repository, the lowest-risk E2E experiment is not "invent true correlation from single columns"; it is a marginal-propagation experiment: compare multi-column selectivity using stale marginals vs OASIS-corrected marginals vs fresh marginals under the same dependence assumption/model.

## Experiments

- Ablation entry points found: `cdf_kll_ml_pipeline/ablation_experiment.py`, `experiments/run_synthetic_paper_suite.py`, and `experiments/run_attention_ablation.py`.
- Lightweight E2E results already exist under `experiments/results/lightweight_e2e`.
- Main ablation rerun completed at `experiments/results/synthetic_paper_suite_rerun_20260529`.
- Rerun headline: OASIS improves Q-Error vs Prior by 51.3% at q=10, 57.9% at q=20, 60.6% at q=30. ISOMER is strongest at q=1/3 but OASIS dominates moderate/high drift.
- Marginal-to-joint E2E experiment completed at `experiments/results/marginal_joint_e2e_20260529`.
- E2E aggregation:
  - Independence/max-entropy estimator: OASIS improves joint Q-error by 7.88% on average; ISOMER improves by 31.90%.
  - Gaussian-copula estimator: OASIS improves joint Q-error by 2.35% on average; ISOMER improves by 22.61%.
  - OASIS improvement is mixed by configuration, with regressions at some low-drift/high-correlation points. This is evidence that corrected marginals can propagate to multi-column estimates, but not yet strong evidence that the current OASIS checkpoint is robust in this E2E setting.
- Hybrid attribution E2E completed at `experiments/results/marginal_joint_hybrid_e2e_20260529`.
- Hybrid policy uses only feedback-window residuals over stale/ISOMER/OASIS marginals. It selected ISOMER for ~98% of columns and OASIS for ~2%; no stale selections under the default margin.
- Hybrid results:
  - Independence/max-entropy: Hybrid joint Q-error 1.218 vs stale 1.820, OASIS 1.679, ISOMER 1.220, fresh 1.185; average Hybrid improvement 32.02%.
  - Gaussian copula: Hybrid joint Q-error 1.174 vs stale 1.558, OASIS 1.513, ISOMER 1.185, fresh 1.139; average Hybrid improvement 23.25%.
  - Marginal-to-joint log-improvement correlation averages 0.76 for independence and 0.65 for Gaussian copula, confirming that marginal repair quality is a strong predictor of downstream joint-estimation quality except at the highest correlation settings.
- OASIS diagnosis:
  - Plain OASIS improves marginal Q-error over stale in only 43.6% of E2E rows and beats ISOMER on marginal Q-error in only 3.9%.
  - Plain OASIS beats stale joint Q-error in 52.7% of independence rows and 47.1% of Gaussian-copula rows; it beats ISOMER joint Q-error in only 23.4% / 21.3%.
  - OASIS feedback residual beats ISOMER on both columns in only ~1.0% of rows, indicating the MLP correction is not feedback-consistent enough for this E2E setting.
  - High-correlation copula cases can amplify moderate marginal errors into extreme joint Q-error regressions.
- Targeted improvement tested: OASIS-Proj, which starts from OASIS marginals and then applies feedback-consistency projection via the ISOMER/IPF correction.
  - Independence/max-entropy: OASIS-Proj joint Q-error 1.222 vs plain OASIS 1.679, stale 1.820, ISOMER 1.220, fresh 1.185; average projected improvement 31.76%.
  - Gaussian copula: OASIS-Proj joint Q-error 1.176 vs plain OASIS 1.513, stale 1.558, ISOMER 1.185, fresh 1.139; average projected improvement 23.11%.
  - OASIS-Proj beats stale in 81.7% / 78.6% of joint rows and beats ISOMER in 50.2% / 49.2%, making it a genuine OASIS-specific repair rather than only a fallback.
  - Hybrid over stale/ISOMER/OASIS/OASIS-Proj selects ISOMER ~49.5%, OASIS-Proj ~50.3%, plain OASIS ~0.3%, stale 0%; it slightly improves over both ISOMER and OASIS-Proj on average.

## Paper revision implications

- The paper should not claim that plain OASIS directly and robustly improves multi-column E2E cardinality estimates.
- The honest framing is: OASIS corrects single-column marginal inputs; those corrections propagate to downstream independence/copula estimators only when the supplied marginals are feedback-consistent.
- OASIS-Proj is best presented as a lightweight deployment calibration step, not as a new correlation model: OASIS learns the global marginal correction, then the projection enforces the current feedback constraints before multi-column composition.
- The main paper now includes a concise marginal-to-joint table showing plain OASIS weak/mixed, OASIS-Proj/Hybrid near fresh marginals and slightly competitive with ISOMER, plus diagnostic discussion of marginal inconsistency and copula amplification.

## Optimizer-decision proxy

- A generator-driven optimizer-facing experiment can avoid wall-clock runtime claims while still testing whether better statistics improve planning signals. The new proxy evaluates scan/join choices selected from estimated selectivities and scores them by true proxy cost using generated ground-truth selectivities.
- Formal results in `experiments/results/optimizer_decision_proxy_20260529`:
  - Stale: selectivity Q-error 2.867, join regret 1.187, optimal join choice 79.0%.
  - OASIS: selectivity Q-error 1.760, join regret 1.056, optimal join choice 89.5%, resolves 63.0% of stale risky proxy decisions.
  - ISOMER: selectivity Q-error 1.664, join regret 1.063, optimal join choice 88.8%, resolves 54.3% of stale risky proxy decisions.
  - OASIS-Proj: selectivity Q-error 1.519, join regret 1.042, optimal join choice 91.1%, resolves 69.0% of stale risky proxy decisions.
  - Hybrid: selectivity Q-error 1.504, join regret 1.041, optimal join choice 91.1%, resolves 66.8% of stale risky proxy decisions.
  - Fresh: selectivity Q-error 1.000, join regret 1.000, optimal join choice 100%.
- This result is appropriate to phrase as optimizer-facing decision-signal improvement, not runtime improvement.

## PostgreSQL planner-only evidence

- Added and ran a real PostgreSQL 16.9 planner-only experiment at `experiments/postgres_planner_stats_injection_experiment.py`.
- The script builds a local database under `/Volumes/QUQ/pg/`, creates fresh current data, and injects alternative single-column `fact.x` statistics into `pg_statistic` by copying typed statistics from analyzed source tables. This avoids hand-writing `anyarray` catalog values.
- The evaluation does not measure query wall-clock runtime. It uses `COUNT(*)` only to obtain true cardinalities, and uses `EXPLAIN (FORMAT JSON)` to read PostgreSQL's estimated rows and plan shape under each injected statistics state.
- The script now supports multi-configuration batch runs. Formal batch results are in `experiments/results/postgres_planner_stats_injection_batch_20260529`, covering 12 configurations: left/right drift direction, 100K/200K fact-table rows, and three random seeds.
- Batch headline over all query instances:
  - Stale: row Q-error 27.748, fresh-plan match 56.0%.
  - Plain OASIS: row Q-error 3.306, 88.1% Q-error improvement, fresh-plan match 88.6%, plan recovery 91.7%, but 13.8% new plan deviations.
  - OASIS-Proj: row Q-error 2.360, 91.5% Q-error improvement, fresh-plan match 96.1%, plan recovery 92.6%, and 1.1% new deviations.
  - Hybrid: row Q-error 2.358, fresh-plan match 96.1%, plan recovery 92.6%, and 1.1% new deviations. In this PostgreSQL batch, Hybrid selected ISOMER for all 12 configurations.
  - Fresh: row Q-error 1.412 and 100% fresh-plan match by definition.
- Batch configuration-level means:
  - OASIS-Proj row Q-error 2.375 ± 0.275, close to ISOMER 2.372 ± 0.283 and substantially below plain OASIS 3.313 ± 0.228.
  - OASIS-Proj/Hybrid fresh-plan match 96.1% ± 1.9%, versus plain OASIS 88.6% ± 2.7%.
  - OASIS-Proj/Hybrid new deviations 1.0% ± 1.0%, versus plain OASIS 13.7% ± 2.0%.
- The original single-configuration result remains in `experiments/results/postgres_planner_stats_injection_20260529`, but the Information Systems manuscript now uses the stronger 12-configuration batch table.
- Formal results in `experiments/results/postgres_planner_stats_injection_20260529`:
  - 84 query templates were evaluated across scan, join, and dimension-filtered join families.
  - Stale and fresh statistics disagreed on 42/84 PostgreSQL plan shapes.
  - Stale row Q-error was 35.736 and matched fresh plan shape on only 50.0% of queries.
  - Plain OASIS reduced row Q-error to 3.264, improved over stale on 92.9% of queries, matched fresh plan shape on 90.5%, and recovered 92.9% of stale/fresh plan disagreements; however, it introduced new deviations in 11.9% of cases where stale had already matched fresh.
  - OASIS-Proj reduced row Q-error to 2.283, improved over stale on 96.4% of queries, matched fresh plan shape on 97.6%, recovered 95.2% of stale/fresh plan disagreements, and introduced 0.0% new deviations.
  - Hybrid selected ISOMER for this run and achieved row Q-error 2.273, fresh-plan match 96.4%, and 0.0% new deviations.
  - Fresh PostgreSQL-analyzed statistics are not perfect but are the planner reference: row Q-error 1.506 and 100% fresh-plan match by definition.
- This is the cleanest "real DBMS" evidence currently in the project: OASIS-style marginal corrections can move PostgreSQL's own optimizer estimates and plan choices toward the fresh-statistics planner, while OASIS-Proj is the safer deployment form because it removes the new-plan-deviation issue seen with plain OASIS.

## Reviewer-driven safety diagnostics

- Added `experiments/postgres_plan_change_analysis.py`, which derives family-level plan-shape recovery and representative plan-change examples from the existing PostgreSQL batch output.
- Plan family breakdown:
  - Selection: stale/fresh changed on 95/336 queries; OASIS-Proj fresh-plan match 98.5%, recovery 94.7%, new deviations 0.0%.
  - Join: stale/fresh changed on 215/336 queries; OASIS-Proj fresh-plan match 92.3%, recovery 90.7%, new deviations 5.0%.
  - Join + dimension filter: stale/fresh changed on 134/336 queries; OASIS-Proj fresh-plan match 97.6%, recovery 94.0%, new deviations 0.0%.
- Added `experiments/feedback_budget_sensitivity_experiment.py` and ran it to `experiments/results/feedback_budget_sensitivity_20260529`.
  - All methods see the same truncated window of the K most recent observations; the OASIS checkpoint remains fixed-width K=16 with padding.
  - At K=2, Hybrid selectivity Q-error is 2.233 and join-optimal match is 84.6%, better than stale 2.867 / 79.0% but not yet strong.
  - At K=16, OASIS-Proj selectivity Q-error is 1.518 and Hybrid is 1.504; Hybrid join-optimal match reaches 91.2%.
  - Hybrid choices at K=16 are ISOMER 36.8%, OASIS-Proj 47.4%, OASIS 14.1%, and stale 1.7%, showing that the deployment gate uses multiple components.
- Added `experiments/feedback_noise_robustness_experiment.py` and ran a lightweight formal diagnostic to `experiments/results/feedback_noise_robustness_20260529` with 64 held-out cases per drift intensity and one noise seed.
  - At 10% multiplicative feedback noise, OASIS-Proj selectivity Q-error is 1.580 and Hybrid is 1.565, versus stale 2.868.
  - OASIS-Proj/Hybrid maintain 90.5% join-optimal match at 10% noise, versus stale 79.4%.
  - ISOMER degrades from 1.677 clean-feedback Q-error to 1.779 at 10% noise, while plain OASIS changes from 1.755 to 1.765; this supports combining learned correction with projection and residual gating.

## OOD drift realism

- Added `experiments/ood_drift_realism_experiment.py` and ran it to `experiments/results/ood_drift_realism_20260529`.
- The experiment evaluates the same OASIS checkpoint trained on compound drift only; no OOD drift samples are used for retraining.
- Protocol: for each OOD case, capture stale quantiles, interleave the OOD drift family with 16 feedback observations, compute final post-drift quantiles, and evaluate future selectivity Q-error over 64 CDF probe points.
- Six OOD drift families, 128 cases each:
  - Batch load: stale 4.203, ISOMER 1.461, OASIS 1.280, OASIS-Proj 1.272, Hybrid 1.388.
  - Range shift: stale 1.994, ISOMER 1.210, OASIS 1.352, OASIS-Proj 1.187, Hybrid 1.210.
  - Skew evolution: stale 1.540, ISOMER 1.182, OASIS 1.437, OASIS-Proj 1.171, Hybrid 1.175.
  - Outlier burst: stale 1.258, ISOMER 1.076, OASIS 1.135, OASIS-Proj 1.055, Hybrid 1.067.
  - Multimodal: stale 1.238, ISOMER 1.066, OASIS 1.134, OASIS-Proj 1.060, Hybrid 1.064.
  - Seasonal/mixed: stale 1.212, ISOMER 1.066, OASIS 1.129, OASIS-Proj 1.057, Hybrid 1.064.
- Interpretation: plain OASIS generalizes best on the strongest unseen batch-load drift and beats ISOMER there, but OASIS-Proj is the safest non-fresh method across all six OOD drift families. This directly addresses the drift-simulator realism concern without making runtime claims.
- Wording adjustment: the paper now narrows the statistics-format claim to a PostgreSQL-style MCV+histogram instantiation that isolates engine-specific adapter logic; it no longer claims experimentally demonstrated portability across DBMS formats.

## Trace-grounded drift sanity check

- Added `experiments/trace_grounded_drift_experiment.py` and ran it to `experiments/results/trace_grounded_drift_20260529`.
- The experiment uses explicit insert/update/delete event streams anchored to analytical benchmark table/column patterns rather than the compound training generator:
  - TPC-DS-style sales-date append;
  - promotion price revisions;
  - inventory restocking;
  - returns/cancellations;
  - customer segment churn;
  - seasonal mixed maintenance.
- The same OASIS checkpoint trained on compound drift is reused unchanged; no trace-specific retraining is performed.
- Aggregate geometric mean across the six traces:
  - stale 1.295;
  - ISOMER 1.070;
  - plain OASIS 1.141;
  - OASIS-Proj 1.076;
  - Hybrid 1.072;
  - fresh 1.000.
- OASIS-Proj is the best non-fresh method on the visibly shifted sales-append and returns/cancellation traces. ISOMER is best or tied on the milder update-heavy traces, and plain OASIS can regress there. This strengthens the deployment story: OASIS-style learned marginal repair should be projected or residual-gated before planner use.

## Deployment safety summary

- Added `experiments/deployment_safety_analysis.py`, which derives `experiments/results/deployment_safety_20260529/table_deployment_safety.tex` from cached PG planner, feedback-budget, feedback-noise, and trace-grounded results.
- Key safety rows:
  - PostgreSQL planner new deviations: plain OASIS 13.8% vs OASIS-Proj 1.1%, with 96.1% fresh-plan match.
  - PostgreSQL joins new deviations: plain OASIS 19.8% vs OASIS-Proj 5.0%.
  - Sparse feedback K=2: Hybrid join-optimal choices 84.6% vs stale 79.0%, with 2.5% new-risk loss.
  - Full feedback K=16: Hybrid join-optimal choices 91.2% vs stale 79.0%, with 1.4% new-risk loss.
  - 10% feedback noise: Hybrid join-optimal choices 90.5% vs stale 79.4%, with 1.7% new-risk loss.
  - DML trace sanity: Hybrid selectivity Q-error 1.072 vs stale 1.295; the residual gate selects ISOMER 77%, OASIS-Proj 20%, OASIS 2%, and stale 1%.
