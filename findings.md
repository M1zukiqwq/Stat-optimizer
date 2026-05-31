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

## OASIS Efficiency Feasibility Reading

- `paper/main_is.tex` currently frames OASIS as a two-stage method: Stage 1 OASIS-noProj is learned marginal repair; Stage 2 feedback-consistency projection is part of the deployed/safe method. The paper explicitly says noProj can be weak for composition and harmful in FactorJoin/planner-facing settings, so efficiency work must not simply remove projection without a replacement safety check.
- Main training/evaluation entry point: `experiments/run_synthetic_paper_suite.py`.
  - It generates/loads compound-drift JSON samples, tensorizes them with `cdf_kll_ml_pipeline/tensorizer.py`, trains `MlpHistogramModelV2`, and writes the cached paper suite under `experiments/results/synthetic_paper_suite_rerun_20260529`.
  - `evaluate_main_suite` will generate data, train/load a model, evaluate Prior/LinInterp/FeedAvg/STHoles/QuickSel-H/ISOMER/OASIS-noProj, and write summary/table/figure artifacts.
- Stage 1 implementation: `cdf_kll_ml_pipeline/mlp_histogram_model_v2.py` plus `tensorizer.py`.
  - Input feature tensor = normalized prior internal quantiles, 3 meta features (`null_fraction`, visible-observation fraction, bucket-count scale), K observation slots, and a mask.
  - Each observation has predicate one-hot plus normalized endpoints, reconstructed prior estimated selectivity, actual selectivity, optional time decay, `has_upper`, and span.
  - The model is a pure-NumPy residual MLP: multi-head attention over observation slots, prior encoder, fused MLP, and output delta added to prior quantiles.
  - Loss is mean squared error between predicted normalized internal quantiles and target normalized corrected quantiles, with L2 regularization and Adam.
  - Inference currently has per-call overhead because `predict()` converts saved Python-list weights into NumPy arrays and back on every call.
- Stage 2 projection implementation: `cdf_kll_ml_pipeline/modern_baselines.py::correct_isomer`.
  - It parses feedback observations into interval selectivity constraints, builds a partition from prior/OASIS boundaries plus constraint endpoints, initializes cell probabilities from the prior histogram, and runs cyclic I-projection/IPF over active constraints.
  - If active constraints do not converge within `max_iter`, it drops older constraints first. Default `max_iter=200`, `tol=1e-4`.
  - Full OASIS is implemented across experiments by calling `correct_isomer` with OASIS-noProj boundaries as the prior; ISOMER is the same operator initialized from stale boundaries.
- Existing metrics:
  - Single-column: selectivity Q-error, selectivity MAE, quantile MAE in `run_synthetic_paper_suite.py`; single-column full projection table in `make_single_column_projection_table.py`.
  - Projection safety: feedback residual in `optimizer_decision_proxy_experiment.py`, `copula_oasis_experiment.py`, OOD/trace/noise/budget scripts.
  - Downstream: composition family, FactorJoin, optimizer decision proxy, and PostgreSQL planner-only row Q-error/plan-shape match.
- Cached paper inputs currently referenced by `paper/main_is.tex` include `single_column_projection_20260531`, `projection_locality_20260531`, `composition_family_20260531`, `factorjoin_oasis_20260531`, `postgres_planner_stats_injection_batch_20260529`, `feedback_budget_sensitivity_20260529`, `feedback_noise_robustness_20260529`, `trace_grounded_drift_20260529`, and `deployment_safety_20260529`. Scripts such as `run_synthetic_paper_suite.py`, `composition_family_experiment.py`, `factorjoin_oasis_experiment.py`, and `postgres_planner_stats_injection_experiment.py` rerun nontrivial experiments; `make_single_column_projection_table.py` is presentation-only over cached synthetic data/model.

## OASIS Efficiency Optimization Directions

- Projection-side fast IPF cap with residual guard.
  - Idea: reduce `correct_isomer` `max_iter` from 200 to 25/50/100 and keep the existing feedback-residual check as the acceptance criterion; fall back to full 200 iterations when residual/Q-error proxy degrades.
  - Why promising: projection dominates per-case runtime in the smoke (~53 ms projection vs ~1 ms Stage 1 inference), and the operator already has a tolerance-based early stop.
  - Safety risk: too-low iteration caps may leave constraints underfit on inconsistent/sparse/noisy feedback; must report feedback residual/violations and downstream planner safety, not only Q-error.
- Batch or cached Stage 1 inference.
  - Idea: avoid per-call list-to-NumPy conversion in `MlpHistogramModelV2.predict()` and/or batch feature tensors for many columns at once.
  - Why promising: low risk to accuracy because it should be numerically equivalent; useful if projection gets cheaper and Stage 1 overhead becomes visible.
  - Safety risk: mostly engineering/shape bugs; must preserve model checkpoint format and existing `predict()` behavior.
- Active-set/reduced-constraint projection.
  - Idea: pre-merge duplicate/nested feedback intervals, prioritize high-residual constraints, or cap active constraints before IPF.
  - Why promising: projection cost scales with active constraints and repeated cyclic projections.
  - Safety risk: dropping constraints changes the feedback-consistency story unless rejected constraints are explicitly reported and gated.
- Residual-triggered projection or Hybrid routing.
  - Idea: skip projection only when OASIS-noProj already has feedback residual within a tight threshold; otherwise project or route to ISOMER/OASIS-Proj.
  - Why promising: avoids projection on already-calibrated marginals.
  - Safety risk: this is closest to "removing projection"; it needs downstream validation, especially FactorJoin and PostgreSQL new-plan deviations, before any deployment claim.

## OASIS Efficiency Smoke Result

- Added `experiments/oasis_efficiency_smoke.py`, an additive tiny smoke wrapper over cached synthetic paper data/model. It does not retrain or rerun full experiments.
- Smoke command:
  - `python3 experiments/oasis_efficiency_smoke.py --output-dir experiments/results/oasis_efficiency_smoke_20260531 --q-values 5 10 20 --max-cases-per-q 8 --q-points 24 --fast-projection-iters 25 50 100`
- Smoke pass criteria:
  - at least 15% total runtime reduction versus baseline full OASIS (`max_iter=200`);
  - no more than 2% Q-error regression versus baseline full OASIS;
  - no more than 5% feedback-residual regression versus baseline full OASIS.
- Smoke result over 24 cached cases:
  - Stale: Q-error 2.035, feedback residual mean 0.12666.
  - ISOMER: Q-error 1.362, projection 53.042 ms/case.
  - OASIS-noProj: Q-error 1.187, Stage 1 1.007 ms/case, feedback residual mean 0.07017.
  - Baseline full OASIS: Q-error 1.280, feedback residual mean 0.04001, projection 53.015 ms/case, total 54.021 ms/case.
  - OASIS fast iter 25: Q-error 1.281, residual +4.52%, total 9.692 ms/case, 82.1% runtime reduction; passed smoke.
  - OASIS fast iter 50: Q-error 1.278, residual +1.27%, total 16.470 ms/case, 69.5% runtime reduction; passed smoke and looks like the safest first candidate.
  - OASIS fast iter 100: Q-error 1.280, residual +0.04%, total 29.638 ms/case, 45.1% runtime reduction; passed smoke.
- Interpretation: a lower IPF iteration cap is worth larger validation. `max_iter=50` is the best initial candidate because it preserves residuals more tightly than 25 while still cutting projection time by ~70% in the smoke.

## OASIS Accuracy/Robustness Reading

- `paper/main_is.tex` makes the deployment boundary explicit: OASIS-noProj is Stage 1 learned marginal repair, while full OASIS is Stage 1 followed by a feedback-consistency projection. The paper repeatedly warns that noProj's single-column gains do not justify exposing it directly to composition, FactorJoin, or planner-facing consumers.
- Training entry point: `experiments/run_synthetic_paper_suite.py::train_model`. It collects cached/generated compound-drift JSON samples, tensorizes them, and trains `MlpHistogramModelV2`; the default paper rerun stores data/model under `experiments/results/synthetic_paper_suite_rerun_20260529`.
- Stage 1 features: `cdf_kll_ml_pipeline/tensorizer.py` encodes normalized prior internal quantiles, 3 meta features, K recent observations, and a mask. Each observation has predicate one-hot, normalized value endpoints, prior-estimated selectivity, actual selectivity, optional time decay, upper-bound flag, and span.
- Stage 1 output/loss: `MlpHistogramModelV2` predicts residual deltas added to prior normalized quantiles. Training loss is MSE on normalized target quantiles plus L2 in the backprop gradients; inference clamps/monotonizes at callers.
- Stage 2 implementation: `cdf_kll_ml_pipeline/modern_baselines.py::correct_isomer` parses feedback into interval constraints, partitions the support by prior/OASIS boundaries plus feedback endpoints, initializes cell probabilities from the supplied prior histogram, and cyclically I-projects cell masses to match active constraints. Inconsistent active sets drop older constraints first.
- Full OASIS composition pattern: scripts build noProj boundaries with `MlpHistogramModelV2`, then call `correct_isomer` using those boundaries as the projection prior. ISOMER is the same projection operator initialized from stale boundaries.
- Current metrics and cached outputs cover single-column Q-error/selectivity MAE/quantile MAE, feedback residuals, OOD drift, DML trace sanity, composition, FactorJoin, optimizer proxy, and PostgreSQL planner-only row Q-error/plan-shape safety.

## OASIS Accuracy/Robustness Optimization Directions

- Damped feedback projection with residual guard.
  - Idea: full OASIS can overfit a small or noisy feedback window. Project against targets `alpha * actual + (1-alpha) * OASIS-noProj-estimated`, then accept the damped projection only if full-window feedback residual remains within a guard threshold; otherwise fall back to original full OASIS.
  - Why low risk: it is projection-side and preserves full OASIS as fallback. It does not expose noProj directly unless a separate hybrid policy explicitly selects it.
  - Safety risk: it relaxes exact feedback matching, so it weakens the clean "satisfy current feedback" story unless residual/violation bounds and downstream safety are reported.
- Validation/gated projection routing.
  - Idea: choose ISOMER vs full OASIS using residual on held-out recent feedback rather than the exact constraints used for projection.
  - Risk: a tiny exploratory check made this worse on cached single-column Q-error, likely because the validation split starves the projector and the final selection signal is noisy. Keep only as a future noisy-feedback idea.
- Training-side residual/calibration penalty.
  - Idea: during Stage 1 training, add a differentiable penalty for feedback-window residual or consistency on observations, possibly with monotonicity/shape regularization, so noProj starts closer to a projectable safe marginal.
  - Risk: requires retraining and could sacrifice the learned global drift prior that gives noProj its single-column advantage; must be checked in composition/FactorJoin/planner settings.
- Hard-case or OOD drift sampling.
  - Idea: upweight cases where noProj has high feedback residual or where projection later degrades future Q-error, and add drift-family augmentation from OOD/trace generators.
  - Risk: can overfit the synthetic generator or feedback window; requires held-out drift families and noisy-feedback validation.

## OASIS Accuracy Smoke Result

- Added `experiments/oasis_accuracy_smoke.py`, an additive cached-data smoke wrapper. It does not retrain the model or change existing experiment defaults.
- Candidate: `oasis_damped_guarded`, with `damping_alpha=0.65`, residual mean guard `30%`, and residual max guard `50%`.
- Smoke command:
  - `python3 experiments/oasis_accuracy_smoke.py --output-dir experiments/results/oasis_accuracy_smoke_20260531 --q-values 5 10 20 --max-cases-per-q 12 --predicates-per-case 16`
- Pass criteria were written into `verdict.json`: at least 1% selectivity Q-error improvement over full OASIS; no more than 5% feedback-residual regression; no more than 2% structural MAE regression; no more than 1% join-regret regression; no new-risk increase above 0.01 absolute.
- Smoke result over 36 cached cases and 576 future predicates per method:
  - Stale: selectivity Q-error 2.627, feedback residual mean 0.12664, join regret 1.1184.
  - ISOMER: Q-error 1.500, residual 0.04700, join regret 1.0587.
  - OASIS-noProj: Q-error 1.648, residual 0.06873, join regret 1.0457.
  - Full OASIS: Q-error 1.426, residual 0.04191, quantile MAE 0.04916, join regret 1.0288, new-risk 0.87%.
  - OASIS damped guarded: Q-error 1.382, residual 0.04294, quantile MAE 0.04309, join regret 1.0241, new-risk 0.87%.
  - Residual Hybrid: Q-error 1.408, residual 0.03917, join regret 1.0334.
  - Fresh: Q-error 1.000, join regret 1.0000.
- Verdict: passed. Relative to full OASIS, damped guarded projection improved Q-error by 3.05%, worsened feedback residual by 2.46% (within the 5% guard), improved selectivity MAE by 5.49%, improved quantile MAE by 12.35%, improved join regret by 0.46%, and did not increase new-risk loss.
- Interpretation: this is worth larger validation, but only as a guarded calibration variant. It must not be claimed as a replacement for feedback-consistency projection until composition, FactorJoin, OOD/noise, and planner-facing safety checks pass.

## OASIS Accuracy Expanded Validation

- Expanded command:
  - `python3 experiments/oasis_accuracy_smoke.py --output-dir experiments/results/oasis_accuracy_smoke_expanded_20260531 --q-values 1 3 5 10 15 20 25 30 --max-cases-per-q 128 --predicates-per-case 32`
- Expanded coverage: 1024 cached cases and 32768 future predicates per method.
- Pre-registered pass criteria remained unchanged from the tiny smoke: at least 1% selectivity Q-error improvement over full OASIS, residual regression <= 5%, structural regression <= 2%, join-regret regression <= 1%, and no new-risk increase > 0.01 absolute.
- Expanded aggregate result:
  - Stale: selectivity Q-error 2.403, feedback residual mean 0.12056, join regret 1.1399.
  - ISOMER: Q-error 1.541, residual 0.05062, join regret 1.0478.
  - OASIS-noProj: Q-error 1.647, residual 0.06432, join regret 1.0475.
  - Full OASIS: Q-error 1.442, residual 0.04401, quantile MAE 0.05194, join regret 1.0344, new-risk 1.81%.
  - OASIS damped guarded: Q-error 1.437, residual 0.04409, quantile MAE 0.04877, join regret 1.0336, new-risk 1.71%.
  - Residual Hybrid: Q-error 1.429, residual 0.03949, join regret 1.0326, new-risk 1.39%.
- Expanded verdict: failed the pre-registered improvement threshold. Damped guarded improved Q-error by only 0.34% versus full OASIS, below the 1% threshold, while safety stayed acceptable: feedback residual +0.18%, quantile MAE -6.11%, join regret -0.08%, and new-risk -0.10 percentage points.
- Per-drift behavior: damped guarded improved full OASIS at q=1 (+0.32%), q=3 (+0.80%), q=5 (+0.06%), q=20 (+0.87%), q=25 (+0.43%), and q=30 (+1.07%), but regressed at q=10 (-0.29%) and q=15 (-0.58%). The effect is safe but too small/inconsistent to justify heavy downstream validation as-is.
- Decision: do not promote damped projection to large FactorJoin/PostgreSQL validation yet. The more robust near-term candidate is residual-gated Hybrid, which had better aggregate Q-error than full OASIS in the expanded run (1.429 vs 1.442) and lower residual/new-risk, but this is already part of the paper's safety story rather than a new Stage-2 replacement.

## OASIS Bold Accuracy/Router Improvement

- Failure diagnosis for the first damped candidate:
  - Damped projection has high-variance gains: some cases improve future Q-error by 20%--36%, but q=10/q=15 contain equally large regressions.
  - The residual guard was too weak as a predictor of future selectivity quality. Several bad cases were accepted even when ISOMER or full OASIS would have been safer.
  - Plain residual Hybrid was already better than damped guarded on expanded data (1.429 vs 1.437), suggesting that the right direction is a richer residual-routed candidate pool rather than a single fixed damping alpha.
- Implemented a bolder additive smoke candidate in `experiments/oasis_accuracy_smoke.py`:
  - `oasis_aggressive_hybrid` builds internal candidates from stale, ISOMER, OASIS-noProj, full OASIS, guarded damped OASIS, a damping-alpha grid `[0.35, 0.50, 0.65, 0.80, 0.95]`, and recent-window projections over K=4/8/12 feedback suffixes.
  - It selects the candidate with the lowest feedback residual; no fresh labels or future predicates are used for routing.
  - `oasis_recency_hybrid` uses the same candidate pool but scores residuals with recency weights; this is also reported but is not the default verdict candidate.
- Bold tiny command:
  - `python3 experiments/oasis_accuracy_smoke.py --output-dir experiments/results/oasis_accuracy_smoke_bold_20260531 --q-values 5 10 20 --max-cases-per-q 12 --predicates-per-case 16`
- Bold tiny result:
  - Full OASIS Q-error 1.426, feedback residual 0.04191, join regret 1.0288, new-risk 0.87%.
  - `oasis_aggressive_hybrid` Q-error 1.389, residual 0.03729, join regret 1.0285, new-risk 0.87%; verdict passed with 2.56% Q-error improvement and 11.0% lower feedback residual.
- Bold expanded command:
  - `python3 experiments/oasis_accuracy_smoke.py --output-dir experiments/results/oasis_accuracy_smoke_bold_expanded_20260531 --q-values 1 3 5 10 15 20 25 30 --max-cases-per-q 128 --predicates-per-case 32`
- Bold expanded result:
  - Full OASIS: Q-error 1.442, selectivity MAE 0.0675, quantile MAE 0.05194, feedback residual 0.04401, join regret 1.0344, new-risk 1.81%.
  - `oasis_aggressive_hybrid`: Q-error 1.412, selectivity MAE 0.0644, quantile MAE 0.05001, feedback residual 0.03786, join regret 1.0308, new-risk 1.32%.
  - Verdict passed: Q-error improves by 2.09%, feedback residual improves by 13.98%, selectivity MAE improves by 4.57%, quantile MAE improves by 3.72%, join regret improves by 0.35%, and new-risk drops by 0.49 percentage points.
  - Per-q improvement versus full OASIS: q=1 +2.62%, q=3 +0.12%, q=5 +0.71%, q=10 -0.08%, q=15 +4.07%, q=20 +4.72%, q=25 +1.69%, q=30 +2.72%.
- Interpretation:
  - The route is now worth downstream validation because it clears the expanded cached gate while improving safety metrics, unlike the original single-alpha damped candidate.
  - Remaining risk: the candidate pool is projection-heavy and may be slower; before PostgreSQL-scale validation, trim candidates using ablations, likely keeping ISOMER/full/noProj plus the alphas/windows that are actually selected often.

## Aggressive Router Downstream Smoke

- Added `aggressive_hybrid` to `experiments/factorjoin_oasis_experiment.py`.
  - The implementation reuses the same residual-routed candidate-pool idea as the accuracy smoke: ISOMER, OASIS-noProj, full OASIS, Hybrid, damping-alpha projections, and recent-window projections are internal candidates.
  - The result rows now include feedback residual and aggressive choices for both join-key marginals; the summary adds per-method feedback residual and worse-than-stale fraction.
- FactorJoin tiny verification command:
  - `python3 experiments/factorjoin_oasis_experiment.py --output-dir /tmp/factorjoin_aggressive_tiny2 --n-trials 2 --drift-levels 5 15 --n-rows 1500 --join-bins 20 --domain 500`
- FactorJoin smoke command:
  - `python3 experiments/factorjoin_oasis_experiment.py --output-dir experiments/results/factorjoin_aggressive_smoke_20260531 --n-trials 8 --drift-levels 5 10 15 20 25 30 --n-rows 3000 --join-bins 40 --domain 800`
- FactorJoin smoke result:
  - Stale join Q-error 1.201, feedback residual 0.24585.
  - OASIS-noProj join Q-error 1.304 and worse-than-stale 62.5%, reproducing the known unsafe behavior in the bilinear join kernel.
  - Full OASIS/OASIS-Proj join Q-error 1.022, feedback residual 0.00798, worse-than-stale 0.0%.
  - Hybrid join Q-error 1.0183, feedback residual 0.00760, worse-than-stale 0.0%.
  - `aggressive_hybrid` join Q-error 1.0182, feedback residual 0.00759, worse-than-stale 0.0%; it is slightly better than Hybrid/OASIS-Proj and does not introduce the noProj failure mode.
  - Aggressive choices mostly selected OASIS-Proj or ISOMER; occasional recent-window projections were selected, while no raw noProj selections appeared in this FactorJoin smoke.
- Added `aggressive_hybrid` to `experiments/composition_family_experiment.py`.
  - The script now reports aggressive joint Q-error, marginal Q-error, worse-than-stale fraction, and aggressive choices.
- Composition-family tiny verification command:
  - `python3 experiments/composition_family_experiment.py --output-dir /tmp/composition_aggressive_tiny --n-trials 1 --n-rows 1000 --n-predicates 4 --correlations 0.3 --drift-levels 5 --estimators independence gaussian_copula --ipf-grid 12 --ipf-iters 10`
- Composition-family smoke command:
  - `python3 experiments/composition_family_experiment.py --output-dir experiments/results/composition_aggressive_smoke_20260531 --n-trials 4 --n-rows 3000 --n-predicates 12 --correlations 0.3 0.6 0.9 --drift-levels 5 15 25 --ipf-grid 18 --ipf-iters 20`
- Composition-family smoke result:
  - Across all six estimators, `aggressive_hybrid` remains close to Hybrid and full OASIS, improving substantially over stale.
  - It does not beat OASIS-Proj in aggregate: e.g., independence 1.209 vs OASIS-Proj 1.209, Gaussian copula 1.225 vs 1.223, IPF 1.264 vs 1.258.
  - Marginal Q-error for aggressive is slightly better than OASIS-Proj/Hybrid (1.0499 vs 1.0551/1.0496 in this smoke), but the joint estimator can amplify small marginal routing differences, so the joint result is not a clear win.
- Decision:
  - Downstream status is mixed but constructive. FactorJoin gives a real positive signal: the aggressive router improves or matches calibrated methods while preserving safety where noProj fails. Composition-family is a neutral/safe signal, not an accuracy win over full OASIS.
  - The candidate is worth a larger FactorJoin and optimizer-proxy validation next, but not yet a paper-level replacement for OASIS-Proj across all downstream composition estimators.

## Aggressive Router Planner-Facing Validation

- Added `aggressive_hybrid` to `experiments/optimizer_decision_proxy_experiment.py`.
  - `METHOD_ORDER` now includes aggressive routing, and `build_method_boundaries` constructs the same residual-routed candidate pool while preserving its original two-value return contract for PostgreSQL callers.
  - The script exposes the aggressive damping grid, recent-window projection windows, and projection solver parameters as CLI args.
- Optimizer proxy tiny verification:
  - `python3 experiments/optimizer_decision_proxy_experiment.py --output-dir /tmp/optimizer_aggressive_tiny --q-values 5 --max-cases-per-q 2 --predicates-per-case 4`
- Optimizer proxy smoke command:
  - `python3 experiments/optimizer_decision_proxy_experiment.py --output-dir experiments/results/optimizer_aggressive_smoke_20260531 --q-values 1 3 5 10 15 20 25 30 --max-cases-per-q 64 --predicates-per-case 24`
- Optimizer proxy smoke result:
  - Stale: selectivity Q-error 2.406, join regret 1.1413, join-optimal match 83.3%.
  - OASIS-Proj: Q-error 1.452, join regret 1.0344, join-optimal 92.1%, new-risk 2.04%.
  - Hybrid: Q-error 1.434, join regret 1.0308, join-optimal 92.5%, new-risk 1.33%.
  - `aggressive_hybrid`: Q-error 1.433, join regret 1.0312, join-optimal 92.4%, new-risk 1.33%.
  - Interpretation: aggressive slightly improves selectivity Q-error over Hybrid but slightly worsens join regret/risk resolution. This is a neutral/slightly mixed planner-proxy signal, not a strong improvement.
- Added `aggressive_hybrid` to `experiments/postgres_planner_stats_injection_experiment.py`.
  - The PostgreSQL source-statistics table map now includes `stat_source_aggressive_hybrid`; stat-source loading, injected-stat capture, labels, single-run output, and batch output all inherit the expanded method list.
- PostgreSQL tiny command:
  - `python3 experiments/postgres_planner_stats_injection_experiment.py --output-dir experiments/results/postgres_aggressive_tiny_20260531 --rows 20000 --dim-rows 2000 --stat-source-rows 20000 --drift-family left_shift --seed 20260531 --config-id aggressive_tiny_left_shift_20k`
- PostgreSQL tiny result:
  - Stale row Q-error 25.339 and fresh-plan match 50.0%.
  - OASIS-Proj row Q-error 1.505, fresh-plan match 97.6%, new deviations 2.4%.
  - Hybrid/aggressive row Q-error 1.494, fresh-plan match 95.2%, new deviations 4.8%.
  - Aggressive is row-accurate but less plan-shape-safe than OASIS-Proj in this planner-facing case.
- PostgreSQL batch-smoke command:
  - `python3 experiments/postgres_planner_stats_injection_experiment.py --batch --output-dir experiments/results/postgres_aggressive_batch_smoke_20260531 --batch-seeds 20260531 --batch-rows 20000 --batch-drift-families left_shift right_shift bimodal_shift --dim-rows-ratio 0.10 --min-dim-rows 2000 --stat-source-rows 20000`
- PostgreSQL batch-smoke aggregate:
  - Stale: row Q-error 12.032, fresh-plan match 60.3%.
  - ISOMER/Hybrid/Aggressive: row Q-error 2.216, fresh-plan match 90.5%, recovery 87.0%, new deviations 7.2%.
  - OASIS-Proj: row Q-error 2.221, fresh-plan match 91.3%, recovery 87.0%, new deviations 5.9%.
  - OASIS-noProj: row Q-error 3.037, fresh-plan match 84.9%, new deviations 16.4%.
  - Hybrid choices were ISOMER in all three tiny-batch configurations, and aggressive matched Hybrid exactly.
- Final current-state interpretation:
  - Aggressive routing is a real single-column and FactorJoin improvement candidate.
  - It is safe but not better than OASIS-Proj in composition-family smoke.
  - It is not better than OASIS-Proj for PostgreSQL planner-facing safety, because pure residual minimization tends to choose ISOMER/Hybrid and sacrifices some fresh-plan match/new-deviation safety.
  - Deployment recommendation remains: keep OASIS-Proj as the planner-facing default; consider aggressive routing for FactorJoin-style downstream estimators or as a non-default calibrated variant after candidate-pool pruning.

## Soft-Constrained Stage-2 Projection

- Implemented `cdf_kll_ml_pipeline/modern_baselines.py::correct_soft_isomer`, a soft calibration alternative to hard ISOMER/IPF projection. It optimizes cell masses on the ISOMER interval partition with objective `KL(p || learned_prior) + lambda * weighted_feedback_residual^2`, using recency weights and mirror-descent updates. This is a Stage-2 replacement/enhancement candidate, not a projection skip.
- Exposed `oasis_soft_projection` in `experiments/oasis_accuracy_smoke.py`, `factorjoin_oasis_experiment.py`, `composition_family_experiment.py`, `optimizer_decision_proxy_experiment.py`, `postgres_planner_stats_injection_experiment.py`, `ood_drift_realism_experiment.py`, and `trace_grounded_drift_experiment.py`.
- Expanded cached single-column validation:
  - Command: `python3 experiments/oasis_accuracy_smoke.py --output-dir experiments/results/oasis_soft_projection_expanded_20260531 --q-values 1 3 5 10 15 20 25 30 --max-cases-per-q 128 --predicates-per-case 32 --verdict-candidate oasis_soft_projection`
  - Result: `oasis_soft_projection` Q-error 1.376 vs full OASIS 1.442, feedback residual 0.03475 vs 0.04401, quantile MAE 0.04830 vs 0.05194, join regret 1.0284 vs 1.0344, and new-risk 1.46% vs 1.81%. The pre-registered verdict passed with +4.55% Q-error improvement and -21.04% residual delta versus full OASIS.
- Optimizer proxy full validation:
  - Command: `python3 experiments/optimizer_decision_proxy_experiment.py --output-dir experiments/results/optimizer_soft_full_20260531 --q-values 1 3 5 10 15 20 25 30 --max-cases-per-q 128 --predicates-per-case 32`
  - Result: soft projection SelQE 1.378 and JoinReg 1.0285, better than OASIS-Proj 1.446 / 1.0348, Hybrid 1.430 / 1.0326, and aggressive 1.429 / 1.0323. This is the strongest positive signal for soft projection.
- Downstream composition/FactorJoin:
  - FactorJoin full (`experiments/results/factorjoin_soft_full_20260531`): soft join Q-error 1.046, better than stale 1.213 and unsafe noProj 1.377, but worse than OASIS-Proj 1.024, Hybrid 1.017, and aggressive 1.017.
  - Composition full (`experiments/results/composition_soft_full_20260531`): soft improves substantially over stale/noProj across all six estimators but is consistently slightly worse than OASIS-Proj/Hybrid/aggressive, e.g. Gaussian copula 1.238 vs OASIS-Proj 1.224 and Hybrid/aggressive 1.216.
- Generalization/safety:
  - OOD full (`experiments/results/ood_drift_realism_soft_full_20260531`): soft is close on skew/outlier/multimodal/seasonal, but weak on strong batch/range drift (batch_load 1.482 vs OASIS-Proj 1.272; range_shift 1.284 vs OASIS-Proj 1.187).
  - DML trace full (`experiments/results/trace_grounded_drift_soft_full_20260531`): soft is safe vs stale/noProj but weaker on append/cancellation traces (sales_append 1.190 vs OASIS-Proj 1.121; returns_cancellation 1.204 vs OASIS-Proj 1.142).
  - PostgreSQL 6-config planner subset (`experiments/results/postgres_soft_batch_subset_20260531`): soft RowQE 2.281 vs OASIS-Proj 2.237, fresh-plan match 90.7% vs 91.1%, and new deviations 7.2% vs 6.6%. It is planner-safe relative to stale/noProj but does not beat hard projection.
- Interpretation: soft projection is a real architecture-level improvement for single-column future selectivity and optimizer-proxy regret, but hard feedback-consistency projection remains the downstream/planner default. The most plausible next research step is a learned/uncertainty gate that uses soft projection only in regimes where OOD/trace/PG safety is predicted to hold, plus solver optimization or lower-iteration validation before any deployment claim.
- Engineering cost: the pure-Python mirror-descent solver with 500 iterations made 1024-case expanded/proxy runs take roughly 9--12 minutes. If promoted further, tune `--soft-projection-iters`, add early stopping, or derive a faster coordinate update.

## Soft Projection Failure Diagnosis and Temporal Optimization

- Root cause diagnosis:
  - Hard OASIS-Proj calls the ISOMER/IPF projection, whose implementation incrementally adds feedback constraints and drops older constraints when the active set becomes infeasible. This is an implicit temporal/consistency filter.
  - Full-window soft projection does not drop constraints. It optimizes a global `KL + residual` objective over every feedback predicate in the window. On future selectivity/proxy this can be excellent, but on sequential drift it softly fits stale or contradictory historical feedback that hard projection would discard.
  - The symptom is not simply high feedback residual: full-window soft often has lower feedback residual than hard OASIS, yet worse OOD/trace future Q-error. The problem is fitting the wrong historical constraints, not failing to fit constraints.
- Residual gate diagnosis:
  - Deployment-visible scalar gates over stale/OASIS/hard/soft residuals were not enough. Thresholds that select soft often preserve compound/single-column gains but regress OOD and trace; thresholds that protect OOD/trace select too little soft and lose most single-column gain.
  - Boundary-distance gating was also too conservative: it protects OOD/trace only by selecting little soft, and gives up too much compound gain.
- Implemented opt-in active-set soft:
  - Added `_isomer_latest_feasible_suffix(...)` and optional `active_set=True` to `correct_soft_isomer`. The soft objective can now run only on the latest hard-feasible feedback suffix.
  - Exposed `--soft-projection-active-set` in the soft-capable experiment scripts. Defaults remain unchanged.
  - Verification: `python3 -m py_compile cdf_kll_ml_pipeline/modern_baselines.py experiments/oasis_accuracy_smoke.py experiments/optimizer_decision_proxy_experiment.py experiments/postgres_planner_stats_injection_experiment.py experiments/factorjoin_oasis_experiment.py experiments/composition_family_experiment.py experiments/ood_drift_realism_experiment.py experiments/trace_grounded_drift_experiment.py`.
  - Result: active-set soft is a useful diagnostic but not the best candidate. Medium cached run (`experiments/results/oasis_soft_active_medium_20260531`) gives soft Q-error 1.446 vs hard 1.456, only +0.67%, while feedback residual worsens by 4.37%. Runtime was also high, so it should not be promoted without a cheaper active-set extraction path.
- Temporal soft optimization:
  - Best robust configuration found: `--soft-projection-window 8 --soft-projection-recency-decay 1.0`.
  - Expanded single-column (`experiments/results/oasis_soft_recent8_expanded_20260531`): soft recent8 Q-error 1.408 vs full OASIS 1.442, feedback residual 0.04037 vs 0.04401, quantile MAE 0.04713 vs 0.05194, join regret 1.0322 vs 1.0344. It is slightly better than aggressive Q-error (1.412) and has better quantile MAE.
  - OOD full (`experiments/results/ood_drift_realism_soft_recent8_full_20260531`): recent8 fixes the severe full-window soft failures. Batch load 1.271 vs hard 1.272, range shift 1.177 vs hard 1.187, skew evolution 1.174 vs hard 1.171, outlier 1.055 vs hard 1.055, multimodal 1.068 vs hard 1.060, seasonal 1.056 vs hard 1.057.
  - Trace full (`experiments/results/trace_grounded_drift_soft_recent8_full_20260531`): recent8 fixes the worst append/cancellation failures relative to full-window soft, but still trails hard/Hybrid on most traces. Sales append 1.127 vs hard 1.121; returns cancellation 1.155 vs hard 1.142.
  - Optimizer proxy full (`experiments/results/optimizer_soft_recent8_full_20260531`): recent8 remains better than hard OASIS-Proj on planner proxy signals, with SelQE 1.408 / JoinReg 1.0318 vs hard 1.446 / 1.0348, though weaker than full-window soft's 1.378 / 1.0285.
  - FactorJoin full (`experiments/results/factorjoin_soft_recent8_full_20260531`): recent8 improves over full-window soft but remains behind calibrated hard/Hybrid: all-drift join Q-error soft 1.038 vs OASIS-Proj 1.024 and Hybrid/aggressive 1.017.
  - Composition full (`experiments/results/composition_soft_recent8_full_20260531`): recent8 is close to hard and better than full-window soft in several rows, but still not consistently better than Hybrid/aggressive. It only beats hard clearly in IPF/Sinkhorn (1.271 vs 1.280).
  - PostgreSQL 6-config subset (`experiments/results/postgres_soft_recent8_batch_subset_20260531`): recent8 has worse RowQE than hard (2.296 vs 2.237), but slightly better plan-shape safety: fresh-plan match 91.3% vs 91.1%, new deviations 6.2% vs 6.6%.
- Window/decay search:
  - `window=12, decay=1.0` improves expanded single-column more than recent8 (Q-error 1.389 vs hard 1.442), but reintroduces safety regressions on OOD batch/range and trace append/returns: batch load 1.360 vs hard 1.272; sales append 1.182 vs hard 1.121; returns cancellation 1.213 vs hard 1.142.
  - Full-window stronger recency decay (`decay=0.5`) passed medium cached accuracy but was too slow at full scale and was not pursued as a robust candidate.
- Current recommendation:
  - If combining soft projection with OASIS, use temporal soft as a non-default candidate or an evaluation variant: OASIS-noProj prior + soft projection on the most recent 8 observations.
  - Do not replace hard OASIS-Proj in the paper's default planner/downstream story. The safer integration is a multi-candidate calibrated router that may include `soft_recent8`, but the default safety anchor remains hard projection or Hybrid.

## Conflict-Aware Soft Projection (supersedes fixed recent8 as the robust soft variant)

- New hypothesis. The fixed recent-window (`recent8`) soft variant is a blunt instrument: it discards *all* old feedback by age, including old observations that are still consistent with the current data state. Those consistent old observations carry useful drift signal, which is why `recent8` loses single-column/proxy accuracy versus full-window soft, while full-window soft is unsafe because it also keeps *contradicted* old observations. The right axis is consistency, not age.
- Mechanism. Added `conflict_aware` to `cdf_kll_ml_pipeline/modern_baselines.py::correct_soft_isomer`. It fits a hard reference distribution to the most recent `conflict_ref_window` observations (the trusted current-state reference), scores every constraint's residual against that reference (`conflict_j = |A_j p_ref - y_j|`), and multiplies the soft residual weight by `exp(-(conflict_j / conflict_tau)^2)` (clipped to `conflict_floor`, reference observations pinned to weight 1). This is a smooth, deployment-visible generalization: it keeps old observations consistent with recent feedback (recovering accuracy) and suppresses contradicted ones (recovering drift safety), without any fresh labels. `conflict_tau -> 0` approaches `recent8`; `conflict_tau -> inf` approaches full-window soft.
- Diagnostics (`experiments/oasis_soft_projection_diagnostics.py`, `experiments/results/oasis_soft_diagnostics_20260531`, 384 cases). On cached compound data, an average of 3.47 of 6.08 old observations per case are *consistent* with the recent reference (exactly what `recent8` throws away), versus 2.61 contradicted. Hard ISOMER's active suffix is 5.61 of 14.08 observations, confirming hard projection itself already drops ~60% of the window. Per-case future Q-error: hard 1.449, full-window soft 1.393, `recent8` soft 1.401, conflict-aware soft 1.395; conflict-aware beats `recent8` on 52.9% of cases. A `conflict_tau` sweep showed `tau≈0.03` minimizes future Q-error among soft variants while keeping feedback residual below `recent8`.
- Validated config: `--soft-projection-conflict-aware --soft-projection-conflict-ref-window 8 --soft-projection-conflict-tau 0.03 --soft-projection-recency-decay 1.0` (full window). Conflict-aware soft is reported as method `oasis_soft_projection` in each run below.
- Accuracy (conflict-aware beats both `recent8` and hard, second only to unsafe full-window soft):
  - Expanded single-column (`experiments/results/oasis_soft_conflict_expanded_20260531`): SelQE 1.401 vs hard 1.442 (+2.82%), vs `recent8` 1.408, vs full-window soft 1.376. Feedback residual 0.03924 vs hard 0.04401 and `recent8` 0.04037; quantile MAE 0.04727 vs hard 0.05194; join regret 1.0313 vs hard 1.0344 and `recent8` 1.0322; new-risk 1.61% vs hard 1.81%. Pre-registered verdict passed.
  - Optimizer proxy full (`experiments/results/optimizer_soft_conflict_full_20260531`): SelQE 1.404 / JoinReg 1.0312 / JoinOpt 92.4% / RiskResolved 69.0%, beating hard 1.446 / 1.0348 / 92.2% / 67.9% and `recent8` 1.408 / 1.0318. This is the best *robust* soft result on the planner proxy (full-window soft 1.378 / 1.0285 is better but unsafe).
- Safety/downstream (fixes the full-window catastrophe; comparable to `recent8`; does not beat hard):
  - OOD (`experiments/results/ood_drift_realism_soft_conflict_full_20260531`): fixes full-window soft (batch_load 1.482 -> 1.336; range_shift 1.284 -> 1.235), best on skew_evol (1.169 vs hard 1.171), ties hard on outlier/multimodal/seasonal. Still trails hard/`recent8` on the two strongest drifts (batch_load 1.336 vs 1.272/1.271; range_shift 1.235 vs 1.187/1.177).
  - DML trace (`experiments/results/trace_grounded_drift_soft_conflict_full_20260531`): within ~0.01--0.02 of hard and comparable to `recent8`, fixing full-window's append/returns failures (sales_append 1.135 vs full-window 1.190; returns_cancellation 1.158 vs full-window 1.204).
  - FactorJoin (`experiments/results/factorjoin_soft_conflict_full_20260531`): all-drift join Q-error 1.0367, essentially tied with `recent8` 1.038 and still behind hard 1.024 and Hybrid 1.017. Downstream join amplification remains soft's weak spot.
  - Composition (`experiments/results/composition_soft_conflict_full_20260531`): ~1--2% behind hard across all six estimators (e.g., independence 1.222 vs 1.191, IPF/Sinkhorn 1.284 vs 1.280); Hybrid is best. Conflict-aware does not reproduce `recent8`'s small IPF/Sinkhorn win.
- `conflict_tau` sensitivity (`*_tau015_*` result dirs): at `tau=0.015`, single-column is unchanged (1.403 vs 1.401) but OOD batch/range improve only marginally (1.336 -> 1.322; 1.235 -> 1.232) and outlier/seasonal tie or beat hard. This confirms the mechanism: in-distribution consistent observations have conflict≈0 and survive any `tau`, so accuracy is `tau`-robust, while strong-drift safety is only weakly `tau`-controllable.
- Why the residual OOD batch/range gap persists (failure analysis). On sharp regime-change drift (batch_load, range_shift) some pre-drift observations are *coincidentally* consistent with the recent reference (conflict≈0), so they survive any `tau` and are not dropped, whereas `recent8`'s age cutoff removes them unconditionally. Consistency-based gating therefore cannot fully replicate age-based dropping on abrupt shifts; this is an inherent limit of the conflict signal, not a tuning artifact.
- Conclusion. Conflict-aware soft is the best-balanced and best-accuracy *robust* soft Stage-2 variant found: it strictly improves on `recent8` for single-column and optimizer-proxy accuracy at equal-or-lower feedback residual, and it removes full-window soft's severe OOD/trace regressions. It is the recommended soft variant when a soft Stage-2 is used (single-column / optimizer-facing selectivity). It still does not beat hard OASIS-Proj on FactorJoin row-QE, composition, or strong OOD batch/range drift, so hard OASIS-Proj (or Hybrid) remains the planner/downstream default. Failed/insufficient directions: pure age windows (`recent12` reintroduces OOD/trace failures), active-set soft (slow, weak), full-window recency decay (slow), and tightening `conflict_tau` to close the batch/range gap (accuracy-neutral but only marginal safety gain).

## Banded (Uncertainty-Aware) Projection — Negative Result

- Hypothesis. Decouple soft projection's two effects (regularization toward the learned prior vs relaxed feedback matching) by replacing the quadratic penalty with an I-projection of the learned prior onto a feedback confidence band `y_j ± delta_j`, `delta_j = kappa*sqrt(y_j(1-y_j)) + floor`. Goal: keep hard's exact-on-trusted behaviour (delta=0) while letting positive bands stay closer to the learned prior, hopefully matching hard downstream and beating it on single-column.
- Implemented `cdf_kll_ml_pipeline/modern_baselines.py::correct_band_isomer` (banded I-projection reusing ISOMER's incremental active-set + drop-oldest-on-infeasibility loop, plus optional conflict dropping). Verified that `band_kappa = band_floor = 0` reproduces `correct_isomer` exactly (a by-construction "never worse than hard" guarantee).
- Result: negative. A `kappa` sweep on cached medium data monotonically *worsens* single-column Q-error (kappa 0.04/0.10/0.20/0.35 -> 1.437/1.450/1.470/1.509 vs hard 1.434) and *raises* feedback residual (0.045 -> 0.053). So `kappa=0` (hard) is optimal within the band family.
- Why it fails (important for the paper's design-space story). Soft projection's single-column advantage is *not* from looser matching: on the same cached data soft attains both lower feedback residual (0.037 vs hard 0.047) and lower Q-error. Soft wins because its cell-level mirror-descent is a better-conditioned solver for the same projection, not because it relaxes constraints. A dead-band only under-corrects, retaining more of the learned prior's error without any compensating benefit. Relaxation is the wrong axis.

## Calibrated Residual-Gated Router (the deployable upgrade)

- Reframing. No single Stage-2 operator dominates hard everywhere: the accuracy regime (single-column / optimizer-facing, mildly noisy future predicates) and the safety regime (join/composition/strong-OOD, dense consistent feedback) genuinely prefer different operators. But the paper's existing deployment mechanism — a residual-gated Hybrid — can dominate hard *by construction* once a safe soft candidate is in its pool, because it selects per case the candidate with the lowest deployment-visible feedback residual.
- Implementation. Added `calibrated_hybrid` to `experiments/oasis_accuracy_smoke.py`: a residual-gated router over `{stale, ISOMER, OASIS-noProj, hard OASIS-Proj, conflict-aware soft, banded(kappa=0)}`. It reuses the existing `choose_residual_hybrid` gate (no fresh labels). Conflict-aware soft is supplied via the existing soft CLI flags; band candidate uses `kappa=0` (hard-equivalent, harmless).
- Accuracy (dominates hard on every metric; `experiments/results/oasis_calibrated_hybrid_expanded_20260531`): SelQE 1.395 vs hard 1.442 (+3.28%) and vs the old residual Hybrid 1.429; feedback residual 0.03579 (lowest of all methods, -18.68% vs hard 0.04401); quantile MAE 0.04856 vs 0.05194; join regret 1.0302 (best) vs 1.0344; new-risk 1.37% vs 1.81%. Verdict passed. Per-case routing: conflict-soft 446/1024, ISOMER 301, hard OASIS-Proj 216, noProj 45, stale 13, band 3 — it uses soft ~44% of cases and falls back otherwise.
- Safety preserved by the same gate. On FactorJoin (`experiments/results/factorjoin_soft_conflict_full_20260531`) the residual-routed `aggressive_hybrid` is 1.0174, already <= hard 1.024, and it does *not* select conflict-soft (soft's join feedback residual 0.0153 > hard 0.0070, so the gate keeps hard/ISOMER); soft alone there is 1.0367. The router therefore tracks hard exactly where exactness matters. Composition/OOD/trace are likewise covered by the existing Hybrid routing, which already matched hard.
- Conclusion (paper-relevant). The deployable method can be upgraded from "hard OASIS-Proj" to a calibrated residual-gated router whose pool includes conflict-aware soft. It is `>=` hard OASIS-Proj on single-column accuracy (1.395 vs 1.442), feedback residual, join regret, and new-risk, while `>=` hard on FactorJoin/composition/OOD because the gate falls back to hard/ISOMER there. This is a legitimate main-result improvement: the gain is per-case, comes from a deployment-visible signal, and never exposes the unsafe full-window soft or noProj.

### PostgreSQL plan-shape safety of the calibrated router (gate confirmation)

- Added `calibrated_hybrid` to the PostgreSQL planner path (`SOURCE_TABLE_BY_METHOD`, stat-source build loop, labels) via the shared `optimizer_decision_proxy_experiment.choose_hybrid` (now accepts a candidate list); pool is `{stale, ISOMER, OASIS-noProj, hard OASIS-Proj, conflict-aware soft}`.
- 6-config subset (`experiments/results/postgres_calibrated_batch_subset_20260531`, left/right/bimodal shift x 2 seeds, 504 queries):
  - hard OASIS-Proj: RowQE 2.237, fresh-plan match 90.9%, new plan deviations 6.86%.
  - conflict-aware soft alone: RowQE 2.277, fresh-plan match 91.1%, new deviations 6.54%.
  - existing Hybrid: RowQE 2.234, fresh-plan match 90.1%, new deviations 8.17%.
  - `calibrated_hybrid`: RowQE 2.234, fresh-plan match 90.1%, new deviations 8.17% — byte-identical to the existing Hybrid across all 504 queries; the gate routed to ISOMER on every config and never selected soft.
- Reading. On planner-facing cases the feedback-residual gate prefers ISOMER, so adding conflict-aware soft to the pool introduces *zero* plan-shape regression: `calibrated_hybrid == Hybrid` on PostgreSQL. The router is therefore plan-shape-safe. The single safest method on PG new-deviations remains hard OASIS-Proj (6.86% vs Hybrid/calibrated 8.17%), but that is a pre-existing property of the residual gate's ISOMER preference on planner cases, not something the soft candidate introduces.
- Net main-result claim that is now fully supported. Extending the residual-gated Hybrid with conflict-aware soft strictly improves single-column (1.395 vs hard 1.442) and optimizer-facing accuracy (SelQE 1.404 / JoinReg 1.0312 region) while leaving FactorJoin, composition, and PostgreSQL plan-shape behaviour identical to the already-deployed Hybrid. So the headline single-column/optimizer numbers can be improved with no downstream/planner safety regression versus the paper's current deployment form. (If the paper also wants to remove the Hybrid-vs-hard PG new-deviation gap, that is a separate gating-criterion question — residual vs plan-shape — independent of this work.)

## Stage-2 Calibration Is Estimator-Agnostic (supports a Stage-2-centric framing)

- Motivation. The paper's Stage-1 evidence is the projection-initialization ladder (`tab:projection_initialization`: ISOMER 1.536 -> OASIS-Proj 1.446 -> fresh 1.157), which varies the prior under a fixed projection. It does not swap in a *different Stage-1 estimator*. To test whether the Stage-2 calibration/router is the reusable component, `experiments/stage1_estimator_swap_experiment.py` plugs six Stage-1 marginal estimators (stale, LinInterp, FeedAvg, STHoles, QuickSel-H, OASIS MLP) into the pipeline and applies Stage-2 in {none, hard projection, calibrated router}.
- Formal run. Command: `python3 experiments/stage1_estimator_swap_experiment.py`. Result dir: `experiments/results/stage1_estimator_swap_20260531`. Coverage: 1024 cached cases, q in `{1,3,5,10,15,20,25,30}`, 128 cases/q, 32 future predicates/case, 6144 source-case units. Runtime: 405.5s with a zero-dependency progress bar. Outputs include overall/per-q CSV+JSON, router choice counts, run config, summary text, and `table_stage1_estimator_swap.tex`.
- Result (future selectivity Q-error, raw / +hard / +router):
  - Stale 2.404 / 1.539 / 1.437; LinInterp 2.116 / 1.508 / 1.417; FeedAvg 2.474 / 1.544 / 1.437; STHoles 1.572 / 1.436 / 1.394; QuickSel-H 1.614 / 1.423 / 1.392; OASIS MLP 1.643 / 1.439 / 1.392.
  - The router reduces future Q-error versus the raw Stage-1 marginal for every estimator in aggregate (11%--42%) and for every per-q/source cell. Hard projection also improves raw for every per-q/source cell. Overall router is `<=` hard projection for every source, but per-q there are three small QuickSel-H exceptions: q=3 (hard 1.253 vs router 1.267), q=5 (1.328 vs 1.335), and q=10 (1.4752 vs 1.4758). Feedback residual drops raw -> hard -> router for all six sources in aggregate (e.g. OASIS MLP 0.0643 -> 0.0440 -> 0.0358).
  - Router choice mix is broad rather than OASIS-specific: it selects soft 44%--61% depending on source, ISOMER 19%--38%, hard 0%--22%, raw/noProj up to 5.5%, and stale about 0.5%--1.3%.
- Two honest reads:
  1. Pro Stage-2 framing: the calibration layer is estimator-agnostic and is the robust, reusable mechanism; it lifts and safeguards whatever Stage-1 marginal it is given. This is strong support for making Stage 2 (consistency-gated soft + calibrated router + regime-dependence) the headline contribution.
  2. Tempers Stage-1 uniqueness: after calibration the final Q-error clusters at 1.392--1.437, and the learned OASIS MLP (1.392) is effectively tied with QuickSel-H + router (1.392) and close to STHoles + router (1.394). The OASIS MLP's distinctive value is therefore not single-column-after-routing uniqueness; it should be framed as one strong prior generator whose value must be justified by OOD/distribution-transfer/downstream behavior, while Stage 2 is the reusable deployment mechanism.
- Implication for the paper: this experiment is the missing "swap the estimator" evidence and argues for a Stage-2-centric story (calibration layer + router as the contribution; Stage 1 as a necessary-but-interchangeable learned front-end whose generalization advantage shows up in OOD/downstream, not in single-column-after-calibration).
