# Cross-column repair on the Join Order Benchmark (real PostgreSQL)

Reproduces Table `tab:crosscol_job` in the paper: cross-column (independence) repair on the
**Join Order Benchmark (JOB)** — the standard real-data cardinality-estimation benchmark —
measured against a live **PostgreSQL** planner.

## What it shows
On real IMDB column pairs, for conjunctive predicates (range on numeric columns, equality on
categorical), it compares the geometric-mean cardinality **Q-error** of four estimators:

| estimator | meaning |
|---|---|
| **PG-default** | PostgreSQL's own estimate (per-column stats + independence) |
| **PG-extended** | PostgreSQL `CREATE STATISTICS` multi-column stats, built from a full scan |
| **AVI** | independence using the *true* marginals (isolates the cross-column error) |
| **OASIS** | the feedback-repaired joint (K=16 observed conjunctive masses, **no scan**) |

The honest cross-column effect is the **AVI→OASIS** reduction (same true marginals, only the
joint differs): **56–66%** on pairs where independence is the real error
(`title`/`aka_title` `production_year`×`kind_id`), and ≈0 on a near-independent control
(`person_info`). PostgreSQL's own estimates are off up to ~78× and its extended statistics do
not help; see the paper for the full discussion and the per-column caveat.

## Prerequisites
- PostgreSQL >= 10, with the `psql` client.
- Python 3 + `numpy`.
- The IMDB dataset for JOB as delimited `<table>.dat` files (21 tables). Schema + queries:
  <https://github.com/gregrahn/join-order-benchmark>. The data files used in the paper are
  pipe-delimited with empty=NULL; the canonical comma-CSV dump also works (see `load_imdb.sh`).

## Steps
```bash
# 1. load IMDB into a 'imdb' database (set your psql + data dir)
PSQL_BIN=/path/to/psql PGHOST=/tmp PGPORT=5432 \
IMDB_DAT_DIR=/path/to/imdb_dat_files ./load_imdb.sh

# 2. run the experiment (connection via libpq env vars or --host/--port/--user)
PSQL_BIN=/path/to/psql PGHOST=/tmp PGPORT=5432 \
python3 crosscol_pg_inject_experiment.py        # -> results_pg_inject/summary.json
```

## Files
- `schema_load.sql` — 21-table JOB schema (constraints stripped for fast/robust load; the
  always-null `imdb_id` columns are `text`).
- `load_imdb.sh` — create db + schema + `COPY` all tables + indexes + `ANALYZE`.
- `crosscol_pg_inject_experiment.py` — the headline experiment (PG-default / PG-extended /
  AVI / OASIS), reproduces `tab:crosscol_job`.
- `crosscol_job_experiment.py` — a lighter variant (idealized AVI vs feedback over more pairs;
  no PG planner comparison).
- `results/pg_inject_summary.json` — the paper's numbers.

The trained OASIS prior is **not** needed here: single-/cross-column repair is the
training-free max-entropy / IPF projection.
