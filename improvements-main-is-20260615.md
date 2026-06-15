# Paper Improvement Plan

**Paper:** OASIS: Repairing Stale and Correlation-Blind Query-Optimizer Statistics from Query Feedback  
**Generated:** 2026-06-15

---

## Critical Issues

- [ ] **Add end-to-end cross-column optimizer evidence**
  - **Location:** Section 5.2, Section 5.3
  - **Problem:** The strongest novel part is cross-column feedback repair, but it is evaluated mainly as conjunctive-predicate Q-error, not as real optimizer plan/runtime impact.
  - **Suggested Fix:** Inject repaired joint/selectivity estimates into PostgreSQL or a planner harness on JOB/TPC-DS multi-predicate queries; report row Q-error, plan match, runtime or cost-regret.
  - **Estimated Effort:** High

- [ ] **Clarify novelty against ISOMER/STHoles/multidimensional feedback histograms**
  - **Location:** Introduction, Related Work, Method
  - **Problem:** Core hard projection is acknowledged as ISOMER/IPF; reviewers may see OASIS as repackaging known feedback histograms.
  - **Suggested Fix:** Add a precise delta table: prior system, feedback type, statistic output, single vs joint, optimizer integration, scan requirement, safety/router, theoretical characterization.
  - **Estimated Effort:** Medium

- [ ] **Soften safety claims**
  - **Location:** Abstract, Contributions, Section 3.3, Conclusion
  - **Problem:** Router minimizes in-window feedback residual, which does not guarantee future-predicate or plan-shape safety.
  - **Suggested Fix:** Replace guarantee-like wording with empirical wording; state the exact invariant: the router can lower in-window residual over its candidate pool, but cannot guarantee plan-shape non-regression.
  - **Estimated Effort:** Low

## Major Improvements

- [ ] **Add sensitivity studies**
  - **Location:** Experiments
  - **Suggested Enhancement:** Sweep K, B, G, feedback/future mismatch, drift intensity, noisy feedback, and stale-constraint conflict policy.
  - **Expected Impact:** Reduces concern that results depend on K=16, B=10, G=12 and friendly predicate coverage.
  - **Estimated Effort:** Medium

- [ ] **Report variability and tails**
  - **Location:** Tables 3-6 and figures
  - **Suggested Enhancement:** Add confidence intervals or seed-level distributions; include worst-case Q-error and failure examples.
  - **Expected Impact:** Makes geometric means more trustworthy.
  - **Estimated Effort:** Medium

- [ ] **Make OASIS/ISOMER/Router definitions unambiguous**
  - **Location:** Section 3 and experiment captions
  - **Suggested Enhancement:** Add a compact algorithm table listing Stale, ISOMER, OASIS, Soft, Router, Unprojected prior, Learned/LQM.
  - **Expected Impact:** Prevents confusion when OASIS and ISOMER coincide.
  - **Estimated Effort:** Low

- [ ] **Strengthen reproducibility section**
  - **Location:** Declarations or appendix/artifact section
  - **Suggested Enhancement:** Provide exact commands, hardware, PostgreSQL version/configuration, runtime budget, seed list, and mapping from each paper table to script.
  - **Expected Impact:** Improves artifact credibility.
  - **Estimated Effort:** Medium

## Minor Suggestions

- [ ] Replace `\journal{a database systems journal}` before submission.
- [ ] Add mutual information or Spearman correlation for cross-column pairs; Pearson r misses nonlinear dependence.
- [ ] Explain how executor feedback is obtained in real DBMSs without full `COUNT(*)` queries.
- [ ] Add one negative case where feedback constraints are misleading or sparse and show router behavior.
- [ ] Move some dense methodological caveats from Section 5 into a shorter setup table.

## Priority Roadmap

### Phase 1: Before Resubmission

1. Add cross-column planner/end-to-end experiment or explicitly narrow the claim.
2. Rewrite safety claims.
3. Add novelty comparison table.
4. Clarify algorithm naming.

### Phase 2: Strengthen Evidence

1. Add sensitivity studies.
2. Add confidence intervals/tail results.
3. Add noisy/conflicting feedback experiment.

### Phase 3: Polish

1. Tighten abstract.
2. Clean venue metadata.
3. Improve reproducibility instructions.

## Estimated Impact

- **Current score:** 6.4/10
- **Potential score after critical fixes:** 7.5-8.0/10
- **Expected recommendation change:** Borderline / Weak Reject -> Weak Accept or Accept, depending on cross-column end-to-end evidence

---

**End of Improvement Plan**
