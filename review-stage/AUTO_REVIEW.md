# Auto Review Loop — OASIS Paper

## Paper
**OASIS: Feedback-Driven Statistics Correction for Cost-Based Query Optimizers**

Started: 2026-04-24
Difficulty: medium
Max rounds: 4

---

## Round 1 (2026-04-24)

### Assessment (Summary)
- Score: 6/10
- Verdict: not ready
- Key criticisms:
  1. Zero-shot generalization claim not fully credible (only synthetic training data)
  2. End-to-end evidence too weak (no per-query breakdown)
  3. Scope narrower than framing (single-column only, not broader optimizer correction)
  4. Not clear learned model is necessary vs simple heuristics
  5. Robustness coverage thin (only K-sensitivity, no stress tests)
  6. Empirical rigor: no error bars, seed sensitivity, or confidence intervals

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

Score: 6/10 — weak reject / borderline.

Critical Weaknesses:
1. Zero-shot generalization not fully credible — training on synthetic, needs real-data/trace-driven drift study
2. End-to-end evidence too weak — PostgreSQL only 6.8% improvement, needs per-query evidence
3. Scope narrower than framing — single-column only, needs explicit quantification
4. Not clear learned model is necessary — need simpler baselines with same feedback budget
5. Robustness coverage thin — need sparse feedback, bursty drift, out-of-range predicates
6. Empirical rigor — no error bars, seed sensitivity, repeated runs, concurrency overhead

Verdict: No. Fix #1, #2, #4 first.

</details>

### Actions Taken
1. **Simple baselines added (#4)**: Implemented `correct_linear_interp` (LinInterp) and `correct_feedback_avg` (FeedAvg) in `baselines.py`. Added to experiment suite METHODS list.
2. **Multi-seed experiments launched (#6)**: Running 3 seeds (42, 123, 456) on remote server via tmux. Will compute 95% CIs.
3. **TPC-DS per-query analysis script created (#2)**: `tpcds_experiment/analyze_per_query.py` — classifies queries by improvement category, computes per-query timing and Q-error deltas.
4. **Paper scope improvements planned (#3)**: Will narrow claims and add explicit quantification.

### Status
- continuing to round 2
- Difficulty: medium
- Waiting for: 3 seed experiments to complete on remote server

---

## Round 2 (2026-04-24)

### Assessment (Summary)
- Score: 7/10
- Verdict: almost
- Key criticisms:
  1. Zero-shot claim still over-claimed relative to evidence (synthetic training only)
  2. End-to-end story not causal enough (need per-query evidence, not just aggregate)
  3. Scope still broader than proven (need error-budget attribution)
  4. Need attention ablation (mean/max pooling) to prove attention specifically is needed
  5. Safe deployment under mild drift (abstention policy)

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

Score: 7/10 — Almost.

Moved from weak reject to borderline/weak accept. #6 (empirical rigor) mostly addressed, #4 (simple baselines) substantially addressed. But #2 (per-query E2E) still only partially addressed (framework exists but no actual evidence), #3 (scope) still not in paper. Biggest remaining blocker: external validity of zero-shot claim.

Remaining Weaknesses:
1. Zero-shot over-claimed — add real-data eval or narrow claim
2. E2E not causal — need actual per-query results, plan diffs, explain random-offset case
3. Scope not quantified — need error-budget attribution in paper
4. Attention ablation needed — replace with mean/max pooling
5. Abstention/trigger policy for mild drift

</details>

### Actions Taken
1. **Paper revised**: Updated abstract, contributions, baselines section, main results, overhead, and conclusion
2. **Simple baselines table added**: New Table showing LinInterp and FeedAvg vs OASIS
3. **Scope paragraph expanded**: Added error-budget positioning paragraph
4. **Seed stability results added**: Mentioned 3-seed CIs in results text
5. **Attention justification added**: Explained why attention is needed over mean/max pooling
6. **Lightweight E2E experiment proposal**: Documented in `review-stage/lightweight_e2e_proposal.md`

### Planned Next Steps (user request)
- Design lightweight E2E experiment (pg_stats injection approach)
- Add Copula + OASIS composition experiment
- These address weaknesses #1 and #2 most directly

### Status
- continuing to round 3
- Difficulty: medium
- Pending: lightweight E2E experiment design, Copula experiment

## Round 3 (2026-04-24)

### Assessment (Summary)
- Score: 7.5/10
- Verdict: almost (but not ready)
- Key criticisms:
  1. End-to-end causality still not demonstrated (scripts ≠ results)
  2. Attention-specific claim still not proven (need ablation)
  3. Safe deployment under mild drift under-specified
  4. Copula experiment is positioning, not downstream validation
  5. Synthetic-to-real generalization gap reduced but not closed

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

Score: 7.5/10 — Almost, but not ready yet.

Meaningfully better than Round 2. Claim narrowing helps, seed-stability is solid, heuristic baselines make "learned correction matters" credible. But main blocker unchanged: paper lacks actual per-query end-to-end evidence.

Remaining Weaknesses:
1. E2E causality not demonstrated — pg_stats is infrastructure, not results. Need per-query stale/OASIS/ANALYZE cardinality error, plan changes, runtime. Include win/neutral/loss breakdown.
2. Attention-specific claim not proven — LinInterp/FeedAvg show learned correction is necessary, not that attention specifically is needed. Add mean/max pooling ablation.
3. Safe deployment under mild drift — at low drift OASIS not clearly best. Add trigger/abstention policy.
4. Copula helps positioning but not substitute for multi-column evidence — frame as limitation analysis, move to appendix if tight.
5. Synthetic-to-real gap — tie claim to tested drift families or add one real/trace-driven case.

Bottom Line: #1 (E2E results) and #2 (attention ablation) are the critical fixes.

</details>

### Actions Planned
1. **Attention ablation**: Add mean/max pooling ablation experiment (addresses #2)
2. **Abstention policy**: Add simple trigger for mild drift (addresses #3)
3. **Frame copula as limitation**: Move to appendix, frame as positioning (addresses #4)
4. **E2E results**: Need PostgreSQL with TPC-DS — pending database availability (addresses #1)

### Status
- continuing to round 4
- Difficulty: medium
- Critical: need attention ablation and E2E results

