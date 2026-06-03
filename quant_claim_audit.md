# Quantitative Claim Audit

Working directory: `/Volumes/QUQ/Stat-optimizer`

Method: I checked `paper/main_is.tex`, the LaTeX table inputs, and the raw logs/CSVs/summaries listed in the prompt. For the sparse PostgreSQL sweep, numeric metrics use only the top aggregate block of each `k*.log`/`summary.csv`, not the family sub-blocks.

Key arithmetic:

- Sparse K=2 gap: `10.12268038233334 / 1.960288210398859 = 5.1648`, reported as `5.2x`.
- Sparse K=8 gap: `1.566 / 1.514 = 1.0343`; equivalently `(1.566 - 1.514) / 1.566 = 3.32%`. This is not within `3%`.
- TPC-H plan deviations: compared each method's `plan_signature` against fresh for each `(seed, query)`, 18 instances total.
- TPC-H tail ratios: computed `method median_ms / fresh median_ms` and `method median_ms / stale median_ms` per `(seed, query)`.

| claim_id | location | paper_text | paper_value | evidence_file | evidence_value | status | details |
|---|---|---|---|---|---|---|---|
| C01 | abstract | OASIS cuts future-predicate Q-error by 12.8% over ISOMER | 12.8% | `experiments/results/proj_v3/projection_pivot_summary.csv` | `(1.522544255 - 1.327798483) / 1.522544255 = 12.7908%` | rounding_ok | Rounds to 12.8%. |
| C02 | abstract / projection prose | beats STHoles and QuickSel-H | 6.1%, 5.7% | `experiments/results/proj_v3/classical_init_pivot_summary.csv` | STHoles `6.1425%`; QuickSel `5.6562%` | rounding_ok | Matches after rounding. |
| C03 | abstract | 5.2x advantage at sparse feedback | 5.2x | `experiments/results/exp2_sparse_v3_20260601/k2.log` | `10.122680382 / 1.960288210 = 5.1648x` | rounding_ok | Top aggregate block only. |
| C04 | abstract / conclusion | composition family by 20--36% | 20--36% | `experiments/results/comp_v3/composition_family_summary.csv` | OASIS range `19.6586%` to `36.0307%` | rounding_ok | Rounded min/max is 20--36%. |
| C05 | abstract / FactorJoin prose | uncalibrated prior is harmful | harmful | `experiments/results/fj_v3/factorjoin_summary.csv` | noProj `1.621361855` vs stale `1.213093376` | exact_match | noProj is worse than stale. |
| C06 | abstract / TPC-H prose | reproduces fresh-statistics plans and accuracy | fresh plans, accuracy | seed `tpch_runtime_rows.csv` files | OASIS/Router plan deviations `0/18`; Scan Q-err OASIS `2.02196`, Router `2.01404`, Fresh `2.00646` | exact_match | OASIS means `oasis_projected`; Router means `calibrated_hybrid`. |
| C07 | abstract | no runtime-superiority claim over stale | none | `paper/main_is.tex` | abstract says reproduces fresh behavior, not faster than stale | exact_match | Scope statement in intro also says no broad stale-runtime superiority. |
| C08 | rank table/prose | free DOF 6.68 at K=2; 0 by K>=12; 73% pinned at K=8 | 6.68, approx 0, 73% | `paper/main_is.tex` | embedded table has same numbers; no raw rank file found | missing_evidence | No independent raw rank evidence was in the listed evidence files or found by `rg rank`. |
| C09 | projection prose | ISOMER 1.523 to OASIS 1.328, 12.8%; Fresh-init 1.151 | 1.523, 1.328, 12.8%, 1.151 | `experiments/results/proj_v3/projection_pivot_summary.csv` | `1.522544255`, `1.327798483`, `12.7908%`, `1.151329499` | rounding_ok | Matches. |
| C10 | projection prose | STHoles 1.415 and QuickSel-H 1.407 | 1.415, 1.407 | `experiments/results/proj_v3/classical_init_pivot_summary.csv` | `1.414695649`, `1.407403716` | rounding_ok | Matches. |
| C11 | projection prose | gains at q=10,20,30 | 15.2%, 22.2%, 19.6% | `experiments/results/proj_v3/projection_pivot_summary.csv` | `15.1722%`, `22.2140%`, `19.6351%` | rounding_ok | Matches. |
| C12 | locality prose/figure | near gain | 9.1% | `experiments/results/proj_v3/locality_summary.csv` | `(1.423794182 - 1.294854851)/1.423794182 = 9.05899%` | rounding_ok | Matches. |
| C13 | locality prose/figure | mid gain | 14.5% | `experiments/results/proj_v3/locality_summary.csv` | `14.4779%` | rounding_ok | Matches. |
| C14 | locality prose/figure | far gain | 21.2% | `experiments/results/proj_v3/locality_summary.csv` | `21.2205%` | rounding_ok | Matches. |
| C15 | budget prose | K=2 gain | +4.3%, 1.994 vs 2.083 | `experiments/results/feedback_budget_sensitivity_v3/summary.csv` | OASIS-projected `1.993720375`; ISOMER `2.082727023`; gain `4.2727%` | rounding_ok | Matches. |
| C16 | budget prose | K=4 gain | +6.8% | `experiments/results/feedback_budget_sensitivity_v3/summary.csv` | gain `6.7747%` from `1.796528424` vs `1.674808429` | rounding_ok | Matches. |
| C17 | budget prose | K=8 gain | +12.1% | `experiments/results/feedback_budget_sensitivity_v3/summary.csv` | gain `12.1193%` from `1.678251192` vs `1.474828230` | rounding_ok | Matches. |
| C18 | budget prose | K=16 gain | +15.7%, 1.402 vs 1.664 | `experiments/results/feedback_budget_sensitivity_v3/summary.csv` | gain `15.7431%` from `1.664083683` vs `1.402128306` | rounding_ok | Matches. |
| C19 | budget prose | learned completion beats ISOMER at every budget | all reported K | `experiments/results/feedback_budget_sensitivity_v3/summary.csv` | K=2/4/8/16 OASIS-projected is lower than ISOMER | exact_match | True for every reported budget. |
| C20 | ablation table/prose | reconstruction-only fails at every budget | -1.2, -1.9, -7.8 | `paper/main_is.tex` | embedded table only; no raw ablation file found | missing_evidence | Table supports the values, but raw evidence is missing. |
| C21 | ablation prose | post-hoc projection matches full in-loop within 0.5% at every budget | within 0.5% | `paper/main_is.tex` | table differences: K=2 `9.4-8.7=0.7`; K=6 `9.0-7.9=1.1`; K=16 `0.0` | number_mismatch | The embedded table contradicts the prose threshold. |
| C22 | stage1-swap prose | projection and router improve every prior | every prior | `experiments/results/stage1_estimator_swap_v3/estimator_swap_overall.csv` | every row has `raw QE > hard QE > router QE` | exact_match | Matches. |
| C23 | stage1-swap prose | LQM raw and routed | raw 1.607; routed 1.425 | `experiments/results/stage1_estimator_swap_v3/estimator_swap_overall.csv` | `1.607046970`; `1.425446199` | rounding_ok | Matches. |
| C24 | stage1-swap prose | OASIS raw/router and classical raw values | 1.489, 1.328, 1.572, 1.614, 1.607 | `experiments/results/stage1_estimator_swap_v3/estimator_swap_overall.csv` | OASIS raw `1.488868128`; router `1.328259339`; STHoles `1.571883585`; QuickSel `1.613581858`; LQM `1.607046970` | rounding_ok | Matches. |
| C25 | stage1-swap prose | final values in 1.39--1.44 band | 1.39--1.44 | `experiments/results/stage1_estimator_swap_v3/estimator_swap_overall.csv` | non-OASIS routers are `1.392`--`1.437`; OASIS router is `1.328` | scope_overclaim | True only if OASIS is excluded; the sentence does not clearly exclude it. |
| C26 | OOD prose | batch-load and skew projected OASIS edge ISOMER | 1.251 vs 1.461; 1.171 vs 1.182 | `experiments/results/ood_drift_realism_v3/summary.csv` | batch `1.251366470` vs `1.460996298`; skew `1.171211102` vs `1.181610141` | rounding_ok | Matches. |
| C27 | OOD prose | raw prior over-extrapolates on batch-load and skew | 1.987, 1.521 | `experiments/results/ood_drift_realism_v3/summary.csv` | raw OASIS `1.987149240`; `1.520605095` | rounding_ok | Matches. |
| C28 | OOD prose/figure | projected and routed forms remain best non-fresh methods | best non-fresh | `experiments/results/ood_drift_realism_v3/table_ood_drift_realism.tex` | range shift: ISOMER `1.210` beats projected OASIS `1.212`; skew: Soft `1.163` beats projected OASIS `1.171` | scope_overclaim | Safety story is broadly supported, but "best" is too strong. |
| C29 | public trace prose | raw aggregate regression | 1.466 vs stale 1.241 | `experiments/results/public_trace_workload_v3/summary.csv` | raw OASIS `1.465508793`; stale `1.241469276` | rounding_ok | Matches. |
| C30 | public trace prose | time-of-day raw catastrophic regression | 2.66 vs stale 1.77 | `experiments/results/public_trace_workload_v3/summary.csv` | raw OASIS `2.660975205`; stale `1.767332329` | rounding_ok | Matches. |
| C31 | public trace prose | recovery to ISOMER level on time-of-day | about 1.25 | `experiments/results/public_trace_workload_v3/summary.csv` | ISOMER `1.253812749`; Hybrid `1.251210663`; Router `1.255020726` | rounding_ok | Matches as "about 1.25". |
| C32 | public trace prose | best deployable aggregate for Router | 1.115 for Router | `experiments/results/public_trace_workload_v3/summary.csv` | Router `1.115281324`; Hybrid `1.113785115` is lower | scope_overclaim | Router value is correct, but it is not the best aggregate in the table. |
| C33 | composition prose | OASIS improves every estimator by 19.7%--36.0% | 19.7--36.0% | `experiments/results/comp_v3/composition_family_summary.csv` | OASIS-projected improvement min `19.6586%`, max `36.0307%` | rounding_ok | Matches. |
| C34 | composition prose | OASIS-noProj weak or negative gains | weak/negative | `experiments/results/comp_v3/composition_family_summary.csv` | noProj `+0.578%` for Independence; negative for other five estimators | exact_match | Matches. |
| C35 | FactorJoin prose | noProj harmful | 1.621 vs stale 1.213 | `experiments/results/fj_v3/factorjoin_summary.csv` | noProj `1.621361855`; stale `1.213093376` | rounding_ok | Matches. |
| C36 | FactorJoin prose/table | OASIS gain | +15.3% | `experiments/results/fj_v3/factorjoin_summary.csv` | OASIS-projected improvement `15.3017%` | rounding_ok | Matches. |
| C37 | PostgreSQL batch prose | table reports mean-over-configs; paragraph uses pooled | mean vs pooled | `experiments/results/postgres_batch_v3/table_postgres_planner_stats_injection_batch.tex`; `batch_summary.csv` | table stale mean `28.453`; pooled stale `27.768050851` | exact_match | The prose explicitly explains this aggregation difference. |
| C38 | PostgreSQL batch prose | pooled stale to ISOMER/OASIS/noProj | 27.768 -> 2.374 / 2.362 / 3.703 | `experiments/results/postgres_batch_v3/batch_summary.csv` | stale `27.768050851`; ISOMER `2.373730113`; OASIS-projected `2.362247794`; noProj `3.702834349` | rounding_ok | Matches. |
| C39 | PostgreSQL batch prose | OASIS fresh-plan/recovery/new deviations | 95.9%, 92.1%, 1.1% | `experiments/results/postgres_batch_v3/batch_summary.csv` | `95.9325%`; `92.0993%`; `1.0619%` | rounding_ok | Matches. |
| C40 | PostgreSQL batch prose | noProj new deviations | 2.3% | `experiments/results/postgres_batch_v3/batch_summary.csv` | `2.3009%` | rounding_ok | Matches. |
| C41 | PostgreSQL batch prose | OASIS and ISOMER near-identical | 2.362 vs 2.374 | `experiments/results/postgres_batch_v3/batch_summary.csv` | `2.362247794` vs `2.373730113` | exact_match | Claim is qualitative but supported. |
| C42 | sparse prose/table | representative config | left-shift, 40K, 84 queries | `experiments/results/exp2_sparse_v3_20260601/k2/plan_rows.csv`; `k2.log` | `config_id=left_shift_rows40000_seed20260529`; `Queries per method: 84` | exact_match | Config is visible in `plan_rows.csv`; metrics still audited from top blocks. |
| C43 | sparse table | K sweep RowQE table | rows for K=2,4,6,8,16 | `experiments/results/exp2_sparse_v3_20260601/k*.log` | top blocks round to table values | rounding_ok | K=2 `27.9/10.12/1.96/1.95/1.17`; K=16 `28.0/1.51/1.53/1.52/1.17`, etc. |
| C44 | sparse prose | K=2 RowQE values | ISOMER 10.12; OASIS 1.96; Router 1.95 | `experiments/results/exp2_sparse_v3_20260601/k2.log` | `10.122680382`; `1.960288210`; `1.949694610` | rounding_ok | Matches. |
| C45 | sparse prose | K=2 fresh-plan matches | ISOMER about 63%; OASIS about 94% | `experiments/results/exp2_sparse_v3_20260601/k2.log` | ISOMER `63.095%`; OASIS-projected `94.048%`; Router `95.238%` | rounding_ok | The paper's 94% refers to OASIS-projected; Router is 95.2%. |
| C46 | sparse prose | K=2 new deviations | ISOMER 14.3%; OASIS 0% | `experiments/results/exp2_sparse_v3_20260601/k2.log` | `14.2857%`; OASIS-projected/Router `0%` | exact_match | Matches after printed rounding. |
| C47 | sparse prose | K=2 row-Q-error gap | 5.2x | `experiments/results/exp2_sparse_v3_20260601/k2.log` | `10.122680382 / 1.960288210 = 5.1648x` | rounding_ok | Matches. |
| C48 | sparse prose | by K=8 the two are within 3% | within 3% | `experiments/results/exp2_sparse_v3_20260601/k8.log` | ISOMER `1.566`; OASIS-projected `1.514`; improvement `3.32%`; ratio gap `3.43%` | number_mismatch | Even Router gives `3.13%` improvement vs ISOMER. |
| C49 | sparse prose | K=16 tie | 1.51 vs 1.53 | `experiments/results/exp2_sparse_v3_20260601/k16.log` | ISOMER `1.506`; OASIS-projected `1.527` | rounding_ok | Matches. |
| C50 | TPC-H table/prose | Scan Q-err table | Stale 5.23; OASIS 2.02; Router 2.01; Fresh 2.01 | seed `tpch_runtime_rows.csv` files | seed geomean means: stale `5.2263`; OASIS-projected `2.0220`; Router `2.0140`; Fresh `2.0065` | rounding_ok | Population SDs match the table rounding. |
| C51 | TPC-H table/prose | Time/Fresh and Time/Stale | OASIS 0.96/1.49; Router 1.01/1.57 | seed `tpch_runtime_rows.csv` files | OASIS-projected Time/Fresh `0.9644`, Time/Stale `1.4929`; Router `1.0094`, `1.5670` | rounding_ok | Matches. |
| C52 | TPC-H tail prose | plan deviations from fresh | 0/18 for OASIS and Router | seed `tpch_runtime_rows.csv` files | OASIS-projected `0/18`; Router `0/18`; ISOMER `0/18` | exact_match | Computed from `plan_signature`, not the CSV's stale/fresh flag. |
| C53 | TPC-H tail prose | OASIS runtime-to-fresh tail | median 0.96; worst 1.23 | seed `tpch_runtime_rows.csv` files | median `0.957406292`; max `1.231767929` | rounding_ok | Matches. |
| C54 | TPC-H tail prose | Router runtime-to-fresh tail | median 1.00; worst 1.40 | seed `tpch_runtime_rows.csv` files | median `0.995672782`; max `1.396156612` | rounding_ok | Matches. |
| C55 | TPC-H tail prose | Q14 about 13x vs stale; fresh same gap | about 13x | seed `tpch_runtime_rows.csv` files | OASIS/Router worst-vs-stale rows are Q14; seed2 Router/stale `13.4457`; fresh/stale seed2 `13.2240` | rounding_ok | Nuance: fresh/stale Q14 ratios across seeds are `12.214`, `13.224`, `6.956`; only the worst seed is about 13x. |
| C56 | optimizer proxy prose | ordering and key values | stale 2.867/1.187; ISOMER 1.664/1.063; OASIS 1.401/1.029; Router 1.375/1.029/92.9% | `experiments/results/optimizer_decision_proxy_v3/summary.txt` | all-row values: `2.867130`, `1.187104`; `1.664112`, `1.062533`; `1.401423`, `1.029237`; `1.375081`, `1.028487`, `92.9281%` | rounding_ok | Matches. |
| C57 | noise prose | 10% Router degradation and baselines | 1.375->1.418; 92.9->92.2; ISOMER 1.758; stale 2.867 | `experiments/results/feedback_noise_robustness_v3/summary.csv` | Router `1.375081277` to `1.417635788`; join opt `92.9281%` to `92.2255%`; ISOMER `1.758212498`; stale `2.867130374` | rounding_ok | Matches raw summary. |
| C58 | noise table/input structure | Router noise values are table-backed | Router values in table | `experiments/results/feedback_noise_robustness_v3/table_feedback_noise_robustness.tex`; `paper/main_is.tex` | table has no Router/calibrated_hybrid columns; `main_is.tex` does not input this table | missing_evidence | The Router noise prose is supported by `summary.csv/txt`, but not by the listed LaTeX table or an actual `main_is.tex` input. |

## Counts

- total claims checked: 58
- exact_match: 11
- rounding_ok: 39
- number_mismatch: 2
- aggregation_mismatch: 0
- config_mismatch: 0
- scope_overclaim: 3
- missing_evidence: 3

Overall verdict: WARN

Reason: the main headline numbers largely reconcile with the raw evidence, but the audit found two numeric mismatches (`K=8 within 3%`; ablation `within 0.5%`) plus several scope/evidence issues around embedded rank/ablation evidence, "best" wording, and the Router noise table visibility.
