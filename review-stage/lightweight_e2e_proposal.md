# Lightweight End-to-End Experiment Proposal

## User's Request
端到端实验是短板，但修改数据库源代码太麻烦。需要一个轻量级方案。

同时考虑：加上依赖单直方图的多列依赖方法（如 copula）加上 OASIS 之后的表现实验。

## Proposed Approach: Trace-Driven Simulation

### Key Insight
Instead of modifying a real DBMS, we can simulate the optimizer's statistics consumption path using the query planner's selectivity estimation with controlled histogram inputs.

### Experiment 1: Selectivity Trace Replay (Lightweight E2E)

**Setup**: Extract real predicate workloads from TPC-DS query plans, then replay selectivity estimation with different histogram inputs.

1. Collect actual query plans from PostgreSQL/MySQL using `EXPLAIN (ANALYZE, FORMAT JSON)` for TPC-DS queries
2. Extract per-predicate selectivity estimates and actual selectivities from execution stats
3. For each drifted column:
   - Replace the stale histogram with OASIS-corrected histogram
   - Re-estimate selectivities for all predicates referencing that column
   - Compare estimated vs actual selectivities (Q-Error) under: Stale / OASIS / Full ANALYZE
4. Compute downstream impact: join order changes, scan method changes

**No source code modification needed** — only need `EXPLAIN ANALYZE` output and PostgreSQL's `pg_stats` view manipulation.

### Experiment 2: Statistics Injection via pg_stats (PostgreSQL-specific)

PostgreSQL allows direct manipulation of statistics via:
```sql
-- Read current stats
SELECT * FROM pg_stats WHERE tablename = 'item';

-- Update statistics directly (requires superuser)
UPDATE pg_statistic SET ...
```

Or use `ALTER TABLE ... SET STATISTICS` + custom ANALYZE scripts.

**Protocol**:
1. Load TPC-DS, run ANALYZE → capture baseline stats
2. Inject drift (DML), capture stale stats
3. Run OASIS correction on stale stats → produce corrected histogram values
4. Write corrected stats back into pg_statistic
5. Run workload with corrected stats, measure per-query time

This avoids modifying PostgreSQL source but still gives real optimizer behavior.

### Experiment 3: Copula + OASIS Composition

**Setup**: Test whether OASIS improves the marginals used by copula-based multi-column estimators.

1. Implement a simple Gaussian copula model:
   - Input: per-column marginals (histograms) + pairwise correlations
   - Output: joint selectivity estimates for multi-column predicates
2. Compare:
   - Copula(stale marginals) vs Copula(OASIS marginals) vs Copula(fresh marginals)
3. Metrics: Q-Error on multi-column range predicates, join cardinality estimation

**This doesn't require real DBMS** — it's a pure Python evaluation using:
- Column data from TPC-DS tables
- OASIS-corrected marginals from the same framework
- Standard copula theory for joint estimation

### Recommended Priority

1. **Experiment 2** (pg_stats injection) — easiest to implement, most convincing
2. **Experiment 3** (Copula + OASIS) — novel contribution, addresses "complementary to multi-column methods" claim
3. **Experiment 1** (trace replay) — lightweight but less convincing than real DBMS

### Implementation Plan

For Experiment 2 (pg_stats injection):
1. Write a Python script that:
   - Connects to PostgreSQL with TPC-DS loaded
   - Reads stale stats for drifted columns
   - Runs OASIS correction using collected observations
   - Writes corrected stats back via SQL UPDATE on pg_statistic
2. Run TPC-DS queries with each stats configuration (stale, OASIS, full ANALYZE)
3. Collect per-query timing and Q-Error data

For Experiment 3 (Copula composition):
1. Implement Gaussian copula selectivity estimator (~100 lines Python)
2. Use TPC-DS column pairs with known correlations
3. Feed OASIS-corrected marginals vs stale marginals
4. Show improvement in joint selectivity estimation
