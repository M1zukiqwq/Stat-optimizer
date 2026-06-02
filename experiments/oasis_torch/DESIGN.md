# OASIS Stage-1 Rework: Downstream-Objective Training (DESIGN)

## Thesis
The current Stage-1 MLP is trained on single-column quantile **reconstruction** (MAE to a
teacher), then judged on **optimizer-facing** metrics. Exp 3/2 (2026-06-01) show the
consequence: as a single-column prior it does not beat ISOMER/STHoles/QuickSel-H (aggregate
post-projection Q-error OASIS-init 1.446 vs STHoles 1.430 / QuickSel 1.412); its value only
shows under sparse feedback, and the residual gate fails to harvest it (K=2: picks ISOMER 9.97
when OASIS-proj 6.03 was available).

**New objective (user's framing, formalized):** train the prior to minimize the *system /
deployment* error, with single-column accuracy and feedback-consistency kept as
**regularizing constraints** (not discarded — the histogram is consumed by many downstream
estimators and unseen queries, so it must stay valid and safe).

## Gate (definition of success — what must hold to re-run all paper experiments)
Using the **real** projection (`correct_isomer`) on the model's prior, on held-out drift:
1. **Primary:** post-projection held-out future-predicate Q-error `< ISOMER` **and**
   `<= min(STHoles-init, QuickSel-init)` in aggregate. (Beats Exp-3 baselines.)
2. **Sparse regime:** strictly better than ISOMER at small feedback K (e.g. K in {2,4,6}).
3. **No safety regression:** feedback residual not worse than ISOMER; valid monotone CDF;
   join-regret (FactorJoin surrogate) <= ISOMER.
If after iteration these cannot all be met, conclude the learned prior cannot pass and report.

## Representation (matches existing pipeline exactly)
- Marginal = equi-depth inner quantiles `q in (0,1)^{B-1}`, B=10, monotone; boundaries
  `[0]+q+[1]` in normalized `[0,1]`. Selectivity via piecewise-linear CDF (differentiable).
- Q-error `max(e/t, t/e)`; trained in log-space `|log e - log t|` (smooth, symmetric).
- Data: 7024 cached JSON samples (`compound_data/{train,test}_q*`), each with stale prior
  quantiles, feedback observations (type/value/est_sel/act_sel), and **fresh** corrected
  quantiles = ground truth for sampling future predicates and computing true selectivity.

## Model (PyTorch — new, bold; replaces numpy attention-MLP)
Permutation-invariant **set encoder** over feedback + stale-prior tokens:
- Obs token (K<=16): [6-hot predicate type, value, value_upper, est_sel, act_sel,
  residual=est-act, has_upper] -> linear -> d=64.
- Prior tokens (B+1): [boundary value, level p, is_boundary] -> linear -> d.
- Learnable CLS; 3 Transformer encoder layers, 4 heads, GELU; CLS -> MLP(d->128->B) ->
  **width logits -> softmax -> cumsum** => valid monotone inner quantiles by construction.
- Residual mode: predict deltas to the stale cumulative widths in logit space, so at
  zero output the model returns the stale prior (graceful with no/uninformative feedback).

## Differentiable projection in the loop (method 1)
Differentiable IPF on a fixed grid (G=40 cells over [0,1]):
- prior cell masses `m = softmax(widths)`; each obs interval -> soft coverage `A_i in [0,1]^G`,
  target mass `t_i = act_sel_i`. Cyclic I-projection (T=8 unrolled steps): scale covered
  cells by `t_i / (A_i·m)` and renormalize. Fully differentiable.
- Loss is computed on the **projected** marginal `m_proj` (so the prior is optimized for what
  it becomes after calibration — directly closing the Exp-3 raw-vs-projected gap).
- Eval uses the **real** `correct_isomer` (apples-to-apples with ISOMER), not the grid IPF.

## Loss (system error primary; single-column as constraint)
`L = w_fut * QErr_log(proj, future_preds)        # held-out future predicates drawn from FRESH`
`  + w_join * JoinRegret(proj)                    # bilinear FactorJoin surrogate, two drifted keys`
`  + w_cons * FeedbackResidual(prior)             # stay feedback-consistent (safety)`
`  + w_reg  * QuantileMAE(prior, fresh)           # light reconstruction reg (valid/safe hist)`
Future predicates are sampled from the fresh CDF and held out from the feedback set, so the
model must **generalize beyond the feedback window** — exactly where a learned prior can beat
projection-from-stale. Start weights: w_fut=1.0, w_join=0.3, w_cons=0.3, w_reg=0.1 (tunable).

## Curriculum (method 5)
Per batch sample feedback count `K ~ {2,4,6,8,12,16}` (subsample observations) so the model is
strong across feedback densities, with extra weight on small K where learning pays.

## Generalization (method A, cheap add-on)
Mix OOD drift families into training (reuse `extended_drift_generators.py`) if compound-only
training overfits the drift generator. Deferred until base gate behavior is seen.

## Eval harness
- Add `proj_newmodel` initializer to `projection_locality_experiment.py`'s classical-init
  table (alongside ISOMER/STHoles/QuickSel/OASIS) -> direct gate read.
- Re-run Exp-2 sparse PG sweep with the new checkpoint (needs an adapter: export inner
  quantiles in a format the numpy `oasis_boundaries` path can consume, or a thin torch
  inference shim).

## Iteration plan
1. Train base (compound, curriculum, full loss) -> gate eval.
2. If primary fails: add diff-IPF emphasis / raise w_fut; inspect where it loses (which K, which q).
3. If sparse fails: upweight small-K curriculum; add learned gate (method 6).
4. If join fails: raise w_join.
5. If OOD-fragile: add method A.
Stop when gate passes (re-run all paper experiments) or 4-5 iterations show the projection
ceiling dominates (conclude learned prior cannot pass; report honestly).
