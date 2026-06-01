# Plan: Runtime sanity check + standard (TPC-H DML) workload

Addresses two reviewer requests, minimizing risk and preserving the paper's honesty boundary.

- **R1 — Runtime.** A real PostgreSQL `EXPLAIN ANALYZE` micro-experiment on a curated
  workload: does the plan-shape improvement translate into runtime, or at least *not*
  introduce regressions?
- **R2 — Standard workload.** TPC-H-derived DML instead of the NASA HTTP append-only proxy.

## Locked decisions (from user)

1. **Host:** remote 4090 box `10.181.8.145` (ssh key login). Build a **fresh PostgreSQL
   16.x** there (mirror the Mac build), dedicated data dir + non-default port. Long runs
   under `screen` per `CLAUDE.md`.
2. **TPC-H scale:** **SF10**, generated on the remote box via the existing
   `tpch-kit`/`dbgen` at `/home/tianqc/tpch-kit`.
3. **Runtime claim:** **relax** the paper's "no query runtime is measured" boundary to a
   bounded **"no-regression sanity check"** (keep "no broad runtime-superiority claim").
4. **NASA trace:** **keep as secondary**; TPC-H becomes the primary standard-workload
   evidence.

## 0. Framing and the claim we will defend (read first)

The relaxed boundary is a *no-regression / directional-translation* claim, **not** "we're
faster":

> In a controlled, warm-cache runtime check, injecting calibrated statistics (OASIS / Hybrid /
> Router) **introduces no runtime regressions** relative to stale or fresh statistics, and on the
> queries where the plan shape changes toward the fresh-statistics plan, **median execution time
> moves toward the fresh-plan runtime.** Runtime depends on caching, physical design, and system
> load, which we hold fixed; we make no broad runtime-superiority claim.

Pre-registering this (Section 5) means a null or mildly negative result is still a clean,
publishable contribution and does not contradict the rest of the paper.

## R0. Remote setup (one-time, under `screen`)

1. **Build PG 16.x on remote**: download/build (or copy the Mac build recipe), `initdb` a
   dedicated cluster, e.g. data dir `/home/tianqc/pg/data`, port `55432`; set
   `shared_buffers`, `work_mem`, `effective_cache_size`, `random_page_cost`,
   `max_parallel_workers_per_gather=0`, `jit=off`, `autovacuum=off` in `postgresql.conf`.
2. **Generate TPC-H SF10**: `cd /home/tianqc/tpch-kit && ./dbgen -s 10 -f` → `.tbl` files.
3. **Load into PG**: create TPC-H schema with PK/FK, `COPY` each `.tbl`, add B-tree indexes
   on the drifting columns (`lineitem.l_shipdate`, `orders.o_orderdate`, and the join keys).
4. **Sanity**: `ANALYZE`; run the curated queries once to confirm they execute in a
   measurable range (>100ms, ideally >500ms) so timing is well above noise.

All heavy steps run detached: `screen -dmS pgsetup bash -c '...'`.

## 1. Two experiments (both run against the remote PG)

### Experiment A — Runtime sanity check on the existing injection table (low risk, do first)

Purpose: answer **R1** on already-validated infrastructure
(`postgres_planner_stats_injection_experiment.py`), and validate the **timing methodology**
before any TPC-H dependency.

- **Reuse**: same drift families, typed `pg_statistic` injection, method set
  {stale, ISOMER, OASIS, OASIS-noProj, Hybrid, Router, fresh}.
- **Extension**: after injecting a method's stats, additionally run
  `EXPLAIN (ANALYZE, BUFFERS, TIMING ON, FORMAT JSON)` and read top-level `Execution Time`.
  Keep the existing `EXPLAIN (FORMAT JSON)` + `COUNT(*)` capture.
- **Make plan choice matter**: add a B-tree on `fact.x`; scale `fact` to 1–5M rows so the
  seq-scan vs index-scan / nested-loop vs hash-join decision has measurable runtime.
- **Determinism** (mostly already in `planner_prefix`): parallelism off, `jit=off`, fixed
  `work_mem`/`shared_buffers`/`effective_cache_size`/`random_page_cost`;
  `ALTER TABLE ... SET (autovacuum_enabled=false)`; **never** `ANALYZE` after injection.
- **Timing protocol**: 1 warm-up + **N=7** timed runs per (config, query, method); drop
  min & max; report **median** of remaining 5 + bootstrap 95% CI. Optional one cold-cache pass.
- **Curated split**: *plan-change* queries (stale≠fresh plan; test translation) vs
  *plan-stable* queries (regression guard — runtime must be statistically indistinguishable).
- **Metrics**: per-query median exec time per method; ratio vs stale and fresh;
  **regression flag** = OASIS median > stale × 1.15 with disjoint CIs; paired Wilcoxon
  (OASIS vs stale; OASIS vs fresh); aggregate workload time.

### Experiment B — TPC-H SF10 DML-drift workload (answers R2; its `EXPLAIN ANALYZE` also answers R1 on a standard schema)

**B.1 Data & schema.** TPC-H SF10 in the remote PG (R0), PK/FK + B-trees on drifting columns.

**B.2 DML drift = "TPC-H-derived DML."** Use the **official refresh functions** as the drift:
- **RF1**: insert a batch of new orders + lineitems (recent `o_orderdate`/`l_shipdate`).
- **RF2**: delete a batch of the oldest orders + lineitems.
- Run several RF1/RF2 batches to materially shift the date distributions. Keep two variants:
  - *official-RF* (defensible baseline drift),
  - *amplified* (extra RF1 concentrated in a new date band) — regime where correction matters.
- **Drifting columns we correct**: `lineitem.l_shipdate`, `orders.o_orderdate` (primary);
  optionally `l_quantity`, `l_discount`.

**B.3 Stats states.** Same method set, injected as single-column `pg_statistic` rows for the
drifting columns only. `stale` = ANALYZE before refresh; `fresh` = ANALYZE after (oracle);
calibrated states from stale + a K=16 feedback window from probe predicates on post-drift
data. **OASIS uses the existing pretrained checkpoint — no retraining** (tests cross-schema
generalization; consistent with "no per-table retuning").

**B.4 Query workload (curated ~8–10 of 22).** Date/stats-sensitive: **Q1, Q3, Q4, Q5, Q6,
Q7/Q8, Q10, Q12, Q14, Q20**. TPC-H constants are parameterized → set to probe the drifted
date band (still standard templates). Include a small **control set** (predicates not on
drifting columns) as the regression guard.

**B.5 Measurement.** Same rigor as A: warm median exec time over N runs; plan shape +
estimated rows from `EXPLAIN (FORMAT JSON)`; truth via `COUNT(*)`; regression flags; paired
tests; aggregate workload time. Multiple dbgen + refresh seeds for variance.

**B.6 Defensible headline (template).**
> On TPC-H SF10 driven by official refresh-stream DML, injecting calibrated single-column
> statistics recovers X% of stale→fresh plan-shape disagreements and reduces estimated-row
> Q-error from A to B. In a controlled warm-cache runtime check, 0/N queries regress beyond
> 15% vs stale; on the K plan-change queries, median execution time moves from T_stale toward
> the fresh-plan T_fresh.

## 2. Metrics summary

| Metric | Source | Purpose |
|---|---|---|
| Plan-shape match to fresh | EXPLAIN JSON | existing metric, keep |
| Estimated-row Q-error | EXPLAIN JSON vs COUNT(*) | existing metric, keep |
| **Median execution time (warm)** | EXPLAIN ANALYZE JSON `Execution Time` | **new — R1** |
| Runtime ratio vs stale / vs fresh | derived | translation / no-regression |
| #regressions (>15%, CI-disjoint) | derived | **primary no-regression test** |
| Wilcoxon signed-rank (paired) | derived | significance across queries |
| Aggregate workload time per method | sum of medians | overall picture |

## 3. Risk minimization

- **Additive only**: new scripts (`postgres_runtime_tpch_experiment.py`, or extend the
  existing harness behind `--analyze`) and new result dirs. **No existing number, table, or
  experiment modified.**
- **Injected-stats integrity**: read back `pg_stats`, assert histogram == intended boundaries;
  assert estimated rows differ across methods (planner actually consumed injected stats);
  confirm no `ANALYZE`/autovacuum between injection and measurement.
- **Timing noise**: N≥7 warm runs, drop extremes, median + bootstrap CI; queries scaled so
  runtime ≫ noise; record host, PG version, config, cache state in `run_config.json`.
- **Quiescent host**: schedule on the remote box with no competing GPU/CPU jobs; document it.
- **Scope guard**: SF10 + ~10 queries + few seeds → full sweep within a few hours.

## 4. Fallbacks

1. **Remote PG build issues** → use a container image, or fall back to the Mac PG for Exp A only.
2. **Runtime too noisy** → add runs, schedule a quiescent window, or raise SF.
3. **OASIS ≈ ISOMER on TPC-H** → report as "both fix stale, no regression." This *reinforces*
   the paper's thesis (Stage 2 carries safety; prior helps mainly in sparse extrapolation —
   cf. the existing PostgreSQL OASIS≈ISOMER explanation).
4. **A plan-change query regresses** → investigate (cost-model mismatch), report honestly;
   this is exactly what R1 asks.

## 5. Pre-registration (commit before running)

- **Primary (no-regression)**: success = no calibrated-stats warm median exceeds stale warm
  median by >15% with disjoint 95% CIs.
- **Secondary (translation)**: on plan-change queries, OASIS/Hybrid median runtime ≤ stale
  and ≥ fresh within CI.
- **Reporting rule**: all queries/methods/seeds reported; no query dropped post-hoc;
  null/negative stated plainly.

## 6. Paper integration (after results, separate edit)

- New §6 subsection: "TPC-H DML-Drift Workload and Runtime Sanity Check."
- **Relax** the 5 "no runtime measured" statements (abstract, intro, methodology, PostgreSQL
  section, limitations) → "a controlled runtime sanity check confirms no regressions on
  TPC-H; no broad runtime-superiority claim."
- **NASA stays secondary**; TPC-H is the primary standard-workload evidence.

## 7. Effort / order

1. **R0 remote setup** (build PG, dbgen SF10, load + index) — ~1 day, dominated by load.
2. **Experiment A** (extend harness with `--analyze`, index + row scaling) — ~½ day,
   validates timing methodology.
3. **Experiment B** sweep + tables/figure — ~1 day.
