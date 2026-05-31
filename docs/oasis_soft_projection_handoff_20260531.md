# OASIS Soft Projection Handoff

Date: 2026-05-31

This document summarizes the current soft-projection work for OASIS and gives
a handoff prompt for the next model. It is written as a continuation point, not
as final paper prose.

## TL;DR

Soft projection is a real Stage-2 architecture candidate, but it should not
replace hard OASIS-Proj yet.

The best current robust soft variant is:

```bash
--soft-projection-window 8 --soft-projection-recency-decay 1.0
```

This temporal soft projection uses OASIS-noProj as the learned prior and applies
soft calibration only to the 8 most recent feedback observations. It preserves
much of the single-column/proxy accuracy gain and fixes the severe OOD/trace
failures of full-window soft projection. It still trails hard OASIS-Proj or
Hybrid in several downstream/planner-facing settings, so it is a calibrated
variant, not the paper default.

## Paper Boundary

The current Information Systems paper positions OASIS as a two-stage method:

- Stage 1: learned marginal repair, often called `OASIS-noProj`.
- Stage 2: feedback-consistency projection, full deployed OASIS.

The paper safety story depends on Stage 2. `OASIS-noProj` may improve
single-column selectivity but is unsafe in composition, FactorJoin, and
planner-facing settings. Do not recommend dropping projection unless a
replacement calibration/safety mechanism is validated across accuracy and
downstream safety.

## Implemented Code

Main soft projection implementation:

- `cdf_kll_ml_pipeline/modern_baselines.py`
  - `correct_soft_isomer(...)`: soft Stage-2 projection.
  - Objective: `KL(p || prior) + lambda * sum_j w_j (A_j p - y_j)^2`.
  - Uses the same ISOMER interval partition as hard projection.
  - Uses mirror-descent style multiplicative updates over cell masses.
  - Supports `constraint_strength`, `recency_decay`, `target_blend`,
    `max_iter`, `learning_rate`, and `tol`.
  - Added optional `active_set=True`, which first keeps the latest hard-feasible
    feedback suffix before running soft projection.
  - Added `_isomer_latest_feasible_suffix(...)` as the active-set diagnostic
    helper.

Experiment integration:

- `experiments/oasis_accuracy_smoke.py`
- `experiments/optimizer_decision_proxy_experiment.py`
- `experiments/postgres_planner_stats_injection_experiment.py`
- `experiments/factorjoin_oasis_experiment.py`
- `experiments/composition_family_experiment.py`
- `experiments/ood_drift_realism_experiment.py`
- `experiments/trace_grounded_drift_experiment.py`

Soft projection CLI knobs exposed across the relevant scripts:

```bash
--soft-projection-strength
--soft-projection-recency-decay
--soft-projection-target-blend
--soft-projection-window
--soft-projection-iters
--soft-projection-lr
--soft-projection-tol
--soft-projection-active-set
```

The defaults preserve prior behavior except where explicitly passed on the
command line.

## Key Results

### Full-Window Soft

Full-window soft is strongest on single-column and optimizer-proxy accuracy.

- Expanded single-column:
  - Result dir: `experiments/results/oasis_soft_projection_expanded_20260531`
  - Soft Q-error 1.376 vs hard OASIS-Proj 1.442.
  - Feedback residual 0.03475 vs hard 0.04401.
  - Quantile MAE 0.04830 vs hard 0.05194.
  - Join regret proxy 1.0284 vs hard 1.0344.
- Optimizer proxy full:
  - Result dir: `experiments/results/optimizer_soft_full_20260531`
  - Soft SelQE/JoinReg 1.378/1.0285 vs hard 1.446/1.0348.

But full-window soft fails in sequential/OOD safety:

- OOD full:
  - Result dir: `experiments/results/ood_drift_realism_soft_full_20260531`
  - Batch load 1.482 vs hard 1.272.
  - Range shift 1.284 vs hard 1.187.
- Trace full:
  - Result dir: `experiments/results/trace_grounded_drift_soft_full_20260531`
  - Sales append 1.190 vs hard 1.121.
  - Returns/cancellation 1.204 vs hard 1.142.
- PostgreSQL 6-config subset:
  - Result dir: `experiments/results/postgres_soft_batch_subset_20260531`
  - Soft RowQE 2.281 vs hard 2.237.
  - Fresh-plan match 90.7% vs hard 91.1%.
  - New deviations 7.2% vs hard 6.6%.

Conclusion: full-window soft is not a replacement for OASIS-Proj.

### Temporal Soft, Window 8

Command shape:

```bash
--soft-projection-window 8 --soft-projection-recency-decay 1.0
```

This is the best robust soft variant found so far.

- Expanded single-column:
  - Result dir: `experiments/results/oasis_soft_recent8_expanded_20260531`
  - Soft Q-error 1.408 vs hard 1.442.
  - Feedback residual 0.04037 vs hard 0.04401.
  - Quantile MAE 0.04713 vs hard 0.05194.
  - Join regret proxy 1.0322 vs hard 1.0344.
- OOD full:
  - Result dir: `experiments/results/ood_drift_realism_soft_recent8_full_20260531`
  - Batch load 1.271 vs hard 1.272.
  - Range shift 1.177 vs hard 1.187.
  - Skew evolution 1.174 vs hard 1.171.
  - Outlier 1.055 vs hard 1.055.
  - Multimodal 1.068 vs hard 1.060.
  - Seasonal 1.056 vs hard 1.057.
- Trace full:
  - Result dir: `experiments/results/trace_grounded_drift_soft_recent8_full_20260531`
  - Sales append 1.127 vs hard 1.121.
  - Returns/cancellation 1.155 vs hard 1.142.
  - It fixes the severe full-window failures but still trails hard/Hybrid on
    most traces.
- Optimizer proxy full:
  - Result dir: `experiments/results/optimizer_soft_recent8_full_20260531`
  - Soft SelQE/JoinReg 1.408/1.0318 vs hard 1.446/1.0348.
- FactorJoin full:
  - Result dir: `experiments/results/factorjoin_soft_recent8_full_20260531`
  - Soft all-drift join Q-error 1.038 vs hard 1.024 and Hybrid/aggressive
    about 1.017.
- Composition full:
  - Result dir: `experiments/results/composition_soft_recent8_full_20260531`
  - Close to hard projection, and better than hard in IPF/Sinkhorn
    1.271 vs 1.280.
  - Still not consistently better than Hybrid/aggressive across estimators.
- PostgreSQL 6-config subset:
  - Result dir: `experiments/results/postgres_soft_recent8_batch_subset_20260531`
  - Soft RowQE 2.296 vs hard 2.237.
  - Fresh-plan match 91.3% vs hard 91.1%.
  - New deviations 6.2% vs hard 6.6%.

Conclusion: temporal soft is useful, especially for single-column/proxy and
OOD robustness, but still not the planner/default downstream winner.

### Temporal Soft, Window 12

Window 12 looks better on single-column accuracy but reintroduces OOD/trace
failures.

- Expanded single-column:
  - Result dir: `experiments/results/oasis_soft_recent12_expanded_20260531`
  - Soft Q-error 1.389 vs hard 1.442.
- OOD full:
  - Result dir: `experiments/results/ood_drift_realism_soft_recent12_full_20260531`
  - Batch load 1.360 vs hard 1.272.
  - Range shift 1.224 vs hard 1.187.
- Trace full:
  - Result dir: `experiments/results/trace_grounded_drift_soft_recent12_full_20260531`
  - Sales append 1.182 vs hard 1.121.
  - Returns/cancellation 1.213 vs hard 1.142.

Conclusion: window 12 is too long for the current drift settings.

### Active-Set Soft

Active-set soft was added as an opt-in diagnostic:

```bash
--soft-projection-active-set
```

It tries to mimic hard ISOMER's stale-constraint discard behavior, then runs
soft projection on the latest hard-feasible suffix.

Result:

- Medium cached run:
  - Result dir: `experiments/results/oasis_soft_active_medium_20260531`
  - Soft Q-error 1.446 vs hard 1.456.
  - Feedback residual worsened by 4.37%.
  - Runtime was high because the feasibility suffix search repeatedly rebuilds
    partitions and runs IPF checks.

Conclusion: conceptually helpful, not yet a useful implementation.

### Full-Window Strong Recency Decay

`--soft-projection-recency-decay 0.5` passed medium cached accuracy, but full
runs consumed about 5 CPU minutes per task before normal progress output.
Those runs were stopped. This path may be revisited after solver optimization,
but it is not a practical default right now.

## Current Failure Hypothesis

The soft model's core issue is temporal constraint selection.

Hard OASIS-Proj uses `correct_isomer`, which incrementally adds observations and
drops older constraints if the active set becomes infeasible. That behavior is
an implicit drift filter.

Full-window soft projection lacks this filter. It optimizes a low residual
against all feedback observations, including old or contradictory ones. This
explains why full-window soft can have lower feedback residual than hard
OASIS-Proj but worse future/OOD/trace Q-error.

Residual-only gating is insufficient. Gates based on stale residual, OASIS
residual, hard residual, soft residual, or hard-soft boundary distance either:

- select soft often enough to preserve single-column gains but regress
  OOD/trace; or
- protect OOD/trace by selecting almost no soft, losing the point of the soft
  model.

Downstream amplifies the issue. In FactorJoin and composition estimators, small
marginal errors can be amplified by joins or dependence assumptions. In
PostgreSQL, RowQE and plan-shape safety are not identical: recent8 soft has
worse RowQE than hard but slightly better fresh-plan match/new-deviation rates
on the 6-config subset.

## Recommended Next Research Directions

1. Build a soft failure diagnostic script.

   Suggested output per case:

   - full-window soft vs recent8 vs hard boundaries;
   - per-observation residual by age;
   - signed residual vector, not only absolute mean;
   - hard active suffix length;
   - fraction of old observations contradicted by recent observations;
   - KL movement from OASIS-noProj prior;
   - soft-hard boundary distance and per-bin mass movement;
   - future Q-error deltas and downstream deltas.

   Goal: predict when soft should use full window, recent8, hard, or Hybrid.

2. Make active-set soft efficient or metadata-aware.

   Current active-set soft recomputes feasibility in a slow way. Better options:

   - modify `correct_isomer` to optionally return active intervals/metadata;
   - reuse the active suffix from hard projection instead of recomputing;
   - cache partitions for suffixes;
   - approximate conflict score without full IPF convergence.

3. Replace fixed window with noise/conflict-aware weights.

   Candidate weight:

   ```text
   weight_j = recency_j * consistency_j * uncertainty_j * predicate_reliability_j
   ```

   Where:

   - `recency_j` is age decay;
   - `consistency_j` downweights observations contradicted by newer feedback;
   - `uncertainty_j` comes from disagreement among OASIS-noProj, hard, soft,
     ISOMER, or bootstrap perturbations;
   - `predicate_reliability_j` accounts for predicate width/type/noise.

4. Add a calibrated soft/hard/router policy.

   The simple scalar residual gates failed. Try a small interpretable model or
   rule set over deployment-visible features:

   - hard active suffix length;
   - recent/full residual disagreement;
   - old-vs-recent signed residual conflict;
   - soft-hard KL or boundary distance;
   - OASIS-noProj vs hard residual gap;
   - drift magnitude from stale residual.

   Validate the router on single-column, OOD, trace, FactorJoin, composition,
   optimizer proxy, and PostgreSQL subset.

5. Planner-aware objective.

   Soft currently optimizes marginal residual, not downstream safety. Add a
   proxy objective or selection criterion based on:

   - optimizer proxy join regret;
   - fresh-plan match proxy where available;
   - new-risk loss;
   - FactorJoin/composition amplification risk.

6. Solver optimization.

   If soft remains useful, optimize before larger sweeps:

   - reduce `max_iter` with measured accuracy loss;
   - add stronger early stopping;
   - derive coordinate updates for interval constraints;
   - vectorize repeated partition operations.

## Reproduction Commands

Syntax check:

```bash
python3 -m py_compile \
  cdf_kll_ml_pipeline/modern_baselines.py \
  experiments/oasis_accuracy_smoke.py \
  experiments/optimizer_decision_proxy_experiment.py \
  experiments/postgres_planner_stats_injection_experiment.py \
  experiments/factorjoin_oasis_experiment.py \
  experiments/composition_family_experiment.py \
  experiments/ood_drift_realism_experiment.py \
  experiments/trace_grounded_drift_experiment.py
```

Temporal soft recent8 single-column:

```bash
python3 experiments/oasis_accuracy_smoke.py \
  --output-dir experiments/results/oasis_soft_recent8_expanded_20260531 \
  --q-values 1 3 5 10 15 20 25 30 \
  --max-cases-per-q 128 \
  --predicates-per-case 32 \
  --verdict-candidate oasis_soft_projection \
  --soft-projection-window 8 \
  --soft-projection-recency-decay 1.0
```

Temporal soft recent8 OOD:

```bash
python3 experiments/ood_drift_realism_experiment.py \
  --output-dir experiments/results/ood_drift_realism_soft_recent8_full_20260531 \
  --soft-projection-window 8 \
  --soft-projection-recency-decay 1.0
```

Temporal soft recent8 trace:

```bash
python3 experiments/trace_grounded_drift_experiment.py \
  --output-dir experiments/results/trace_grounded_drift_soft_recent8_full_20260531 \
  --soft-projection-window 8 \
  --soft-projection-recency-decay 1.0
```

Temporal soft recent8 optimizer proxy:

```bash
python3 experiments/optimizer_decision_proxy_experiment.py \
  --output-dir experiments/results/optimizer_soft_recent8_full_20260531 \
  --q-values 1 3 5 10 15 20 25 30 \
  --max-cases-per-q 128 \
  --predicates-per-case 32 \
  --soft-projection-window 8 \
  --soft-projection-recency-decay 1.0
```

PostgreSQL 6-config subset:

```bash
python3 experiments/postgres_planner_stats_injection_experiment.py \
  --batch \
  --output-dir experiments/results/postgres_soft_recent8_batch_subset_20260531 \
  --batch-seeds 20260531 20260532 \
  --batch-rows 20000 \
  --batch-drift-families left_shift right_shift bimodal_shift \
  --dim-rows-ratio 0.10 \
  --min-dim-rows 2000 \
  --stat-source-rows 20000 \
  --soft-projection-window 8 \
  --soft-projection-recency-decay 1.0
```

## Update: Conflict-Aware Soft Projection (supersedes recent8)

The fixed recent-window soft variant was superseded by a conflict-aware soft
variant. The key insight: `recent8` cuts feedback by *age*, which also discards
old observations that are still consistent with the current data state. The
right axis is *consistency*, not age.

Implemented in `cdf_kll_ml_pipeline/modern_baselines.py::correct_soft_isomer`
via `conflict_aware`, `conflict_ref_window`, `conflict_tau`, `conflict_floor`
(default off). It fits a hard reference distribution to the most recent
`conflict_ref_window` observations, scores each constraint's residual against
that reference (`conflict_j = |A_j p_ref - y_j|`), and multiplies the soft
residual weight by `exp(-(conflict_j / conflict_tau)^2)`. `tau -> 0` approaches
`recent8`; `tau -> inf` approaches full-window soft. CLI flags
`--soft-projection-conflict-aware/-ref-window/-tau/-floor` are exposed across all
soft-capable scripts.

Diagnostics: `experiments/oasis_soft_projection_diagnostics.py`
(`experiments/results/oasis_soft_diagnostics_20260531`, 384 cases). Per case,
3.47 of 6.08 old observations are consistent with the recent reference; hard
ISOMER active suffix is 5.61 of 14.08. Conflict-aware beats `recent8` future
Q-error on 52.9% of cases.

Validated config: `--soft-projection-conflict-aware --soft-projection-conflict-ref-window 8 --soft-projection-conflict-tau 0.03 --soft-projection-recency-decay 1.0`.

- Single-column expanded (`oasis_soft_conflict_expanded_20260531`): SelQE 1.401
  vs hard 1.442 (+2.82%), recent8 1.408, residual 0.03924 vs recent8 0.04037;
  verdict passed.
- Optimizer proxy (`optimizer_soft_conflict_full_20260531`): SelQE 1.404 /
  JoinReg 1.0312 vs hard 1.446 / 1.0348 and recent8 1.408 / 1.0318.
- OOD (`ood_drift_realism_soft_conflict_full_20260531`): fixes full-window
  (batch 1.482->1.336, range 1.284->1.235), best on skew_evol, ties on milder
  drift; still trails hard/recent8 on batch_load (1.336 vs 1.272/1.271) and
  range_shift (1.235 vs 1.187/1.177).
- Trace (`trace_grounded_drift_soft_conflict_full_20260531`): within ~0.01--0.02
  of hard, fixes full-window append/returns failures.
- FactorJoin (`factorjoin_soft_conflict_full_20260531`): all-drift 1.0367 ~
  recent8 1.038, behind hard 1.024.
- Composition (`composition_soft_conflict_full_20260531`): ~1--2% behind hard
  across all six estimators.
- `tau=0.015` sensitivity (`*_tau015_*`): single-column unchanged (1.403),
  OOD batch/range improve only marginally (1.322 / 1.232).

Failure analysis for the residual OOD batch/range gap: on sharp regime-change
drift, some pre-drift observations are coincidentally consistent with the recent
reference (conflict≈0) and survive any `tau`, whereas `recent8`'s age cutoff
removes them unconditionally. Consistency-based gating cannot fully replicate
age-based dropping on abrupt shifts.

Recommendation: conflict-aware soft is the best robust soft Stage-2 variant
(better single-column/proxy accuracy than recent8 at equal-or-lower residual,
fixes full-window OOD/trace failures). Hard OASIS-Proj/Hybrid remains the
planner/downstream default because soft still trails on FactorJoin, composition,
and strong OOD batch/range drift. PostgreSQL subset was not run because the
candidate did not beat hard projection on the cheaper downstream proxies.

Open directions for the next model: (1) a consistency+age hybrid weight that adds
an explicit age floor to catch coincidentally-consistent pre-drift observations
on abrupt shifts; (2) downstream/planner-aware conflict scoring (FactorJoin or
join-regret proxy instead of marginal residual); (3) a calibrated router that
selects conflict-aware soft only in single-column/optimizer-facing regimes and
hard projection otherwise.

## Update 2: Banded Projection (negative) + Calibrated Router (the main-result upgrade)

- Banded projection `correct_band_isomer` (I-project learned prior onto feedback
  confidence bands `y +/- kappa*sqrt(y(1-y))`; `kappa=0` == hard exactly):
  negative result. Increasing `kappa` monotonically worsens single-column
  (1.437 -> 1.509) and raises residual. Soft's single-column gain is from a
  better cell-level solver (lower residual AND lower Q-error), not from relaxed
  matching, so relaxation is the wrong axis.
- Calibrated residual-gated router `calibrated_hybrid` (in `oasis_accuracy_smoke.py`,
  and in `optimizer_decision_proxy_experiment.py` / PostgreSQL via the shared
  `choose_hybrid` with a candidate list): routes per case by feedback residual
  over `{stale, ISOMER, OASIS-noProj, hard OASIS-Proj, conflict-aware soft}`.
  - Expanded single-column (`oasis_calibrated_hybrid_expanded_20260531`): SelQE
    1.395 vs hard 1.442, lowest feedback residual (0.0358), lowest join regret
    (1.0302), lowest new-risk (1.37%). Routes to soft ~44% of cases.
  - FactorJoin: residual router stays at hard level (1.0174 <= 1.024); soft never
    selected (higher join residual).
  - PostgreSQL 6-config subset (`postgres_calibrated_batch_subset_20260531`,
    504 queries): `calibrated_hybrid` byte-identical to existing Hybrid (RowQE
    2.234, fresh-plan 90.1%, new-dev 8.17%); gate routes to ISOMER, never soft.
    Zero plan-shape regression from adding soft.
- Net: the deployable method can be upgraded from hard OASIS-Proj to this
  calibrated router — strictly better single-column/optimizer accuracy, identical
  FactorJoin/composition/PostgreSQL behaviour to the deployed Hybrid. The
  Hybrid-vs-hard PG new-deviation gap (8.17% vs 6.86%) is a pre-existing
  residual-gate property (prefers ISOMER on planner cases), not caused by soft.
  Open: a plan-shape-aware gate (instead of pure residual) could also close that
  gap and let the router route to hard/soft over ISOMER on planner cases.

## Handoff Prompt For The Next Model

Copy the following prompt for the next model:

```text
你现在在 `/Volumes/QUQ/Stat-optimizer` 这个 repo 里继续研究 OASIS 的 soft projection 方向。先读：

- `paper/main_is.tex`
- `task_plan.md`
- `findings.md`
- `progress.md`
- `docs/oasis_soft_projection_handoff_20260531.md`

背景：
- 论文里的 OASIS 是两阶段：Stage 1 learned marginal repair (`OASIS-noProj`)，Stage 2 hard feedback-consistency projection (`OASIS-Proj` / full OASIS)。
- 不能为了单列 Q-error 直接去掉 projection。`OASIS-noProj` 在 composition / FactorJoin / planner-facing 场景不安全。
- 已实现 soft Stage-2 projection：`cdf_kll_ml_pipeline/modern_baselines.py::correct_soft_isomer`，目标是 `KL(p || learned_prior) + lambda * weighted_feedback_residual^2`。
- 已把 `oasis_soft_projection` 接入 single-column smoke、OOD、trace、FactorJoin、composition、optimizer proxy、PostgreSQL planner batch。
- 当前最稳 soft 配置是 `--soft-projection-window 8 --soft-projection-recency-decay 1.0`。它修复 full-window soft 的 OOD/trace 大失败，但还不能替代 hard OASIS-Proj。

当前关键结果：
- Full-window soft：
  - single-column expanded Q-error 1.376 vs hard OASIS-Proj 1.442；
  - optimizer proxy SelQE/JoinReg 1.378/1.0285 vs hard 1.446/1.0348；
  - 但 OOD batch/range 失败：1.482/1.284 vs hard 1.272/1.187；
  - trace sales/returns 失败：1.190/1.204 vs hard 1.121/1.142；
  - PostgreSQL subset RowQE/fresh-plan/new-dev 2.281/90.7%/7.2% vs hard 2.237/91.1%/6.6%。
- Recent8 soft：
  - single-column expanded 1.408 vs hard 1.442；
  - OOD batch/range 1.271/1.177 vs hard 1.272/1.187；
  - optimizer proxy 1.408/1.0318 vs hard 1.446/1.0348；
  - FactorJoin 1.038 vs hard 1.024 and Hybrid/aggressive 1.017；
  - PostgreSQL subset RowQE worse than hard, 2.296 vs 2.237, but fresh-plan/new-dev slightly better, 91.3%/6.2% vs 91.1%/6.6%。
- Recent12 soft single-column 更好，1.389 vs hard 1.442，但 OOD/trace 又坏：batch 1.360 vs hard 1.272，sales append 1.182 vs hard 1.121，returns 1.213 vs hard 1.142。
- Active-set soft 已实现 `--soft-projection-active-set`，但当前实现慢且 medium 结果弱：Q-error 1.446 vs hard 1.456，feedback residual 反而 +4.37%。不要直接推广，除非先优化 active-set 提取。

你的任务：
1. 不要只复述已有结果。继续研究 soft 为什么仍然在 FactorJoin/composition/PostgreSQL RowQE 上不稳。
2. 优先新增一个诊断脚本，输出每个 case 的：
   - hard active suffix length 或可行后缀 proxy；
   - old-vs-recent residual conflict；
   - signed residual vector by feedback age；
   - full soft / recent8 / hard 的 KL、boundary distance、cell mass movement；
   - future Q-error delta；
   - downstream delta（至少 optimizer proxy，最好再覆盖 FactorJoin/composition/PG subset）。
3. 基于诊断提出并实现一个比 fixed recent8 更聪明的 soft weighting/gating 机制。候选方向：
   - conflict-aware feedback weights；
   - noise-aware/uncertainty-aware weights；
   - soft/hard/router policy using deployment-visible features；
   - efficient active-set metadata reuse from hard projection；
   - planner-aware or downstream-risk-aware selection objective。
4. 验证不能只看 single-column Q-error。至少覆盖：
   - expanded cached single-column；
   - OOD drift realism；
   - trace-grounded drift；
   - optimizer proxy；
   - FactorJoin；
   - composition；
   - PostgreSQL planner 6-config subset，如果候选有希望。
5. 保持实验可复现，不删除旧结果目录，不重置 git，不改无关文件。先用 `ps` 检查是否有实验在跑。
6. 更新 `task_plan.md`、`findings.md`、`progress.md`，必要时也更新 `docs/oasis_soft_projection_handoff_20260531.md`。

推荐第一步：
- 先写一个 `experiments/oasis_soft_projection_diagnostics.py`，复用 `oasis_accuracy_smoke.py` 的 cached data path，比较 hard/full-soft/recent8/recent12/active-set soft，并输出 case-level CSV。
- 用诊断结果训练或手写一个只用部署期特征的 soft gate，然后再推进 full downstream validation。

最终回答用户时必须包含：
- 改了哪些文件；
- 新假设；
- 训练/验证命令；
- accuracy + safety/downstream 结论；
- 哪些方向失败了，为什么失败。
```

