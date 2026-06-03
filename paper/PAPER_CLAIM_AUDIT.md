# Paper Claim Audit Report

**Date**: 2026-06-02
**Auditor**: gpt-5.5 xhigh — fresh zero-context thread (run manually by the author; full log in `quant_claim_audit.md`)
**Paper**: OASIS: Repairing Stale Optimizer Statistics in the Feedback Nullspace (`paper/main_is.tex`)

## Overall Verdict: PASS (post-fix) — was WARN

The independent zero-context audit checked **58 claims** and reconciled the headline
numbers (12.8%, 5.2× sparse, composition 20–36%, PostgreSQL pooled batch, TPC-H zero
plan deviations, TPC-H tail ratios). It flagged 2 number mismatches and 3 scope
overclaims, all now fixed, plus 3 benign missing-evidence notes.

## Claims verified: 58
- exact_match: 11
- rounding_ok: 39
- number_mismatch: 2 → **fixed**
- scope_overclaim: 3 → **fixed**
- missing_evidence: 3 → advisory (not paper errors)
- aggregation_mismatch / config_mismatch: 0

## Issues found and fixes

### [FIXED] C21 — ablation "within 0.5%"
- **Was**: post-hoc vs in-loop "within $0.5\%$ at every budget".
- **Evidence**: `tab:mr_ablation` differs by 0.7 (K=2) and 1.1 (K=6) points.
- **Fix**: changed to "within roughly one percentage point at every budget (Table~\ref{tab:mr_ablation})".

### [FIXED] C48 — sparse "by K=8 within 3%"
- **Was**: "by $K{=}8$ the two are within $3\%$".
- **Evidence**: `exp2_sparse_v3/k8.log` ISOMER 1.566 vs OASIS 1.514 = 3.32% (3.43% ratio gap).
- **Fix**: "by $K{=}8$ the gap has nearly closed (ISOMER 1.57, OASIS 1.51, within $3.5\%$)".

### [FIXED] C25 — stage1-swap "1.39–1.44 band"
- **Was**: band "$1.39$–$1.44$" which excludes OASIS's routed 1.328.
- **Fix**: band widened to "$1.33$–$1.44$" (covers OASIS 1.328); the following sentence already states OASIS routes lowest.

### [FIXED] C28 — OOD "best non-fresh methods"
- **Was**: projected/routed forms "remain the best non-fresh methods".
- **Evidence**: on skew, Soft 1.163 < projected 1.171; on range shift, ISOMER 1.210 < projected 1.212.
- **Fix**: "remain among the strongest non-fresh methods".

### [FIXED] C32 — public trace "best deployable aggregate (Router)"
- **Was**: "best deployable aggregate ($1.115$ for the Router)".
- **Evidence**: Hybrid 1.1138 < Router 1.1153.
- **Fix**: "best deployable aggregates (Hybrid 1.114, Router 1.115)".

## Missing-evidence items — now resolved
- **C08 (rank table)**: raw evidence exported to `experiments/results/mechanism_rank_v3/rank_table.csv` (and `rank_stdout.txt`), produced by `experiments/oasis_torch/mechanism_analysis.py` on the held-out compound data. Values match `tab:mr_rank` exactly (e.g. K=8 free-DOF 0.29, 72.8% pinned → "73%"; K≥12 98.2% → "98%").
- **C20 (ablation table)**: raw evidence exported to `experiments/results/ablation_objective_v3/ablation_gate.csv` (and the gate-eval log `abl_gate_out.log`), from the six objective-ablation checkpoints (`ckpt_abl_*`, `ckpt_v3_it3`). The `impr_vs_stholes_pct` column matches `tab:mr_ablation` exactly (recon −1.2/−1.9/−7.8; post-hoc +9.4/+9.0/+7.8; full +8.7/+7.9/+7.8).
- **C58 (noise)**: the main paper cites the noise numbers as prose only (backed by `feedback_noise_robustness_v3/summary.csv`, which contains the `calibrated_hybrid` rows); it does not `\input` the noise table, so there is no table/prose mismatch in the main paper.

## Result
All actionable discrepancies fixed; main_is.tex recompiles clean (36 pp, 0 undefined refs).
