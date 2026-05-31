# Paper Improvement Log — OASIS

**Date**: 2025-04-25
**Venue target**: PKDD (LNCS, 16-page limit)
**Reviewer model**: GPT-5.4 via Codex MCP

## Round 1 — Score: 6/10 (Borderline Reject)

### Issues Identified
1. CRITICAL: ANALYZE described inaccurately as "full-table scan"
2. CRITICAL: Technical correctness of correction pipeline unclear (validity projection, monotonicity, zero selectivities)
3. MAJOR: Portability claim overstated (only PostgreSQL validated)
4. MAJOR: "Causal link" claim too strong for limited EXPLAIN experiment
5. MAJOR: Scope limitations not surfaced prominently
6. MAJOR: Novelty vs self-tuning histograms insufficiently clarified
7. MAJOR: "Non-invasive" claim misleading (still requires engine integration)
8. MINOR: Abstract overstates evidence strength

### Changes Made
- Fixed ANALYZE description to "typically full or sampled table scan" (abstract, intro, background)
- Added explicit validity projection details: clamping to [0,1], monotone sorting, duplicate nudging (min bucket 1e-6), near-zero masking (s* < 1e-8)
- Moderated portability claim: added "full cross-DBMS validation remains future work"
- Softened "causal link" to "suggestive evidence" / "initial but not conclusive evidence"
- Added prominent scope limitation paragraph in experimental setup
- Clarified "non-invasive": no CBO/planner changes, but ~200-400 lines per-engine glue code
- Sharpened novelty: OASIS learns drift-correction function vs per-table adjustments
- Trimmed text to maintain 16-page limit

## Round 2 — Score: 7/10 (Borderline Reject → improved)

### Remaining Issues (structural, not fixable without new experiments)
1. CRITICAL: End-to-end evidence remains counterfactual, no live DBMS runtime experiments
2. CRITICAL: External validity limited to synthetic drift families
3. MAJOR: Baseline fairness could be more convincing
4. MAJOR: Scope is narrow (single-column only)

### Additional Fixes
- Quantified projection activation rate (<2% of test predictions)
- Quantified near-zero masking impact (<0.5% of test predicates)

### Assessment
The reviewer notes the paper is "meaningfully better and closer to publishable quality." The remaining critical issues require new experiments (live DBMS runtime study) that cannot be addressed by text edits alone. For PKDD submission, the paper's synthetic evaluation is a known limitation that should be honestly stated (as it now is).

## Key Trade-offs Made
- Honesty over selling: Claims are now carefully calibrated to match evidence
- Page budget: Some useful details moved to supplementary to stay at 16 pages
- Structural limitations acknowledged: Single-column scope, synthetic-only evaluation, counterfactual E2E
