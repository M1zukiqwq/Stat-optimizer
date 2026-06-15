# OASIS re-centering — autonomous revision log (2026-06-14)

You authorized: validate the cross-column idea, and if it holds, drive straight through to a
finished, recompiled paper without checking in. It held. Summary below.

## What was validated (the pivot)

**Cross-column feedback repair beats the independence assumption (AVI) on real data.**
`experiments/crosscol_feedback_experiment.py` → `results/crosscol_feedback_v1/`.
- 8 real correlated column pairs (Wine, Bike, Forest, Census, Power).
- AVI = outer product of TRUE marginals (best case for independence → isolates correlation).
- Feedback estimate = 2D max-entropy (IPF) projection onto K=16 observed conjunctive masses.
- Result: **+13.6% aggregate, up to +31.2%** (bike casual/cnt), and the **gain grows with
  |correlation|** (census r=0.07 → +3.9%; forest hill9/hill3 r=−0.78 → +19.1%). Deterministic
  (fixed seed; verified two identical runs).

This is the positive core. It contrasts cleanly with the single-column negative finding:
- Single-column (real data, `results/real_singlecol_v1/`): feedback projection (ISOMER) cuts
  stale error a lot, but a learned prior / STHoles / QuickSel all tie ISOMER within 1–2%
  (dense & sparse). Max-entropy is near-optimal → ML dispensable for 1D.

→ **The honest dichotomy that is now the paper's spine: single columns are easy
(max-entropy optimal); the independence assumption is the bottleneck, and feedback fixes it.**

## New paper thesis (title/abstract/intro/contributions all rewritten)

- Title: *"OASIS: Repairing Stale and Correlation-Blind Optimizer Statistics from Query
  Feedback"* (dropped "Feedback Nullspace" — it caused the DKE mis-read and was tied to the
  refuted learned-prior claim).
- Contributions: (1) identifiability theory (Proposition: when feedback pins the statistic);
  (2) the dichotomy validated on real data; (3) feedback-driven independence repair (the win);
  (4) OASIS as a no-rescan maintenance middleware, validated into real PostgreSQL.
- The learned single-column prior is **demoted to an honest "in-distribution only" result**
  (synthetic 12.8% noted, shown not to transfer). NOT framed as a negative-results paper.

## Files changed
- `paper/main_is.tex` — major re-center (frontmatter, intro, theory→identifiability, new
  §"Repairing the Independence Assumption from Feedback" = `sec:crosscol`, single-column
  reframe + real-data table, System Overview, related work, limitations, conclusion). Backup:
  `paper/main_is.tex.bak_prerewrite_20260614`. Compiles clean (0 undefined, 0 errors, 41 pp).
- `experiments/crosscol_feedback_experiment.py` (new) + `make_real_dataset_cases.py`,
  `make_real_singlecol_table.py`, `gen_train_pool.sh`, `run_real_local.sh` (new).
- `experiments/data/real/` real datasets (UCI: adult, covtype, power, wine, bike).
- `experiments/results/{crosscol_feedback_v1, real_singlecol_v1, real_*_{dense,sparse}{,_v1}}`.
- New table inputs the paper uses: `crosscol_feedback_v1/table_crosscol_feedback.tex`,
  `real_singlecol_v1/table_real_singlecol.tex`.

## Decisions made autonomously (review these)
1. Kept (demoted, clearly labeled "in-distribution / synthetic") the old single-column
   synthetic tables/figures rather than deleting them, to avoid dangling refs. They can be
   trimmed to the supplement for length (paper is 41 pp).
2. `\journal{}` left as a neutral placeholder (venue undecided). `cover_letter_dke.md` is now
   stale — rewrite when the venue is chosen (TODS / VLDBJ / TKDE / EDBT per earlier discussion).
3. Cross-column is an **estimate-quality** result (real correlated columns), not yet a
   PostgreSQL multi-column injection — stated as a limitation. A natural next step is injecting
   repaired joint selectivities / extended-statistics into PG for a plan-level result.

## Update (2026-06-15) — learned prior removed; paper cut 41 → 30 pp
Per request, removed the learned prior as a method/contribution and trimmed length:
- Deleted: the Stage-1 section; the synthetic single-column diagnostics (projection-init,
  locality/budget figures, objective ablation); the OOD/DML-trace/NASA subsection; the
  sparse-PostgreSQL paragraph+table; and the optimizer-decision-proxy subsection. (~356 lines.)
- The learned prior now survives ONLY as a baseline column in the real single-column table and
  as the "unprojected control" (OASIS-noProj) in the composition/FactorJoin figures — i.e. a
  negative control showing the *projection*, not the prior, is what matters. Not a contribution.
- Reframed System Overview (L2 = training-free projection), Stage-2→"Repair Layer", methodology,
  overhead (no forward pass), related work, data availability (UCI, not NASA). Compiles clean
  (0 undefined / 0 errors / 0 bad boxes, 30 pp). Old synthetic result dirs (`proj_v3`,
  `ablation_objective_v3`, `ood_drift_realism_v3`, `optimizer_decision_proxy_v3`, etc.) are now
  unused by the paper but kept on disk.

## Suggested next steps (not done)
- Trim the demoted synthetic single-column diagnostics to the supplement (length).
- Optional: PG plan-level experiment for the cross-column repair (strengthens contribution 3).
- Re-run `citation-audit` / `paper-claim-audit` before submission.
