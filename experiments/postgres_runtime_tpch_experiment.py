#!/usr/bin/env python3
"""Experiment B: TPC-H SF10 DML-drift runtime + plan-shape study.

Reuses the validated Stage-2 boundary builder and feedback-sample structure from
the planner-injection harness, generalized to TPC-H *date* columns:

  * dates are normalized to [0,1] via the global epoch-day domain, so the
    pretrained OASIS checkpoint is applied unchanged (a cross-schema
    generalization test);
  * per-method corrected boundaries are mapped back to dates, materialized in a
    date-typed source table, ANALYZEd, and their ``pg_statistic`` row is copied
    onto the real ``lineitem.l_shipdate`` / ``orders.o_orderdate`` columns;
  * for curated TPC-H queries we record plan shape, single-column scan row
    Q-error, and warm-cache ``EXPLAIN ANALYZE`` execution time.

No runtime-superiority claim: this is a no-regression / directional-translation
runtime sanity check on a standard workload. Run after ``tpch_setup_drift.sh``.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_PIPELINE_DIR = _REPO_DIR / "cdf_kll_ml_pipeline"
for p in (_SCRIPT_DIR, _PIPELINE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from histogram_types import DEFAULT_QUANTILE_LEVELS, FeedbackObservation, KllFeedbackSample, KllPrior  # noqa: E402
from histogram_math import evaluate_piecewise_cdf  # noqa: E402
from mlp_histogram_model_v2 import MlpHistogramModelV2  # noqa: E402
from optimizer_decision_proxy_experiment import METHOD_ORDER, build_method_boundaries, cdf_levels  # noqa: E402
import postgres_planner_stats_injection_experiment as pg  # noqa: E402

EPOCH = date(1970, 1, 1)
TARGETS = {  # logical name -> (table, column)
    "l_shipdate": ("lineitem", "l_shipdate"),
    "o_orderdate": ("orders", "o_orderdate"),
}
# methods we inject for the TPC-H study (subset of METHOD_ORDER, in display order)
B_METHODS = ["stale", "isomer", "oasis", "oasis_projected", "calibrated_hybrid", "fresh"]


@dataclass
class BRow:
    query: str
    method: str
    target_col: str
    scan_true_rows: int
    scan_est_rows: float
    scan_row_qerr: float
    plan_signature: str
    plan_changed_vs_stale_fresh: bool
    median_ms: float
    min_ms: float
    iqr_ms: float
    n_runs: int


# ---- psql helpers (TCP-less, via unix socket; one psql process per call) ----

def psql(args, sql: str, scalar: bool = False) -> str:
    cmd = [str(args.pg_bin / "psql"), "-X", "-v", "ON_ERROR_STOP=1",
           "-h", str(args.socket_dir), "-p", str(args.pg_port), "-U", args.pg_user,
           "-d", args.dbname, "-q", "-A", "-t", "-c", sql]
    env = {"LD_LIBRARY_PATH": str(args.pg_bin.parent / "lib")}
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**_os_environ(), **env})
    if proc.returncode != 0:
        raise RuntimeError(f"psql failed: {sql[:120]}\n{proc.stderr}")
    return proc.stdout.strip()


def _os_environ() -> Dict[str, str]:
    import os
    return dict(os.environ)


def planner_prefix(args) -> str:
    return (f"SET max_parallel_workers_per_gather={args.parallel_workers}; SET jit=off; "
            f"SET random_page_cost={args.random_page_cost}; SET effective_cache_size='{args.effective_cache_size}'; ")


# ---- date / normalization helpers ----

def to_epoch_days(d: date) -> int:
    return (d - EPOCH).days


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


def norm(d: date, d0: int, d1: int) -> float:
    return (to_epoch_days(d) - d0) / max(d1 - d0, 1)


def denorm_to_date(u: float, d0: int, d1: int) -> date:
    return EPOCH + timedelta(days=int(round(d0 + u * (d1 - d0))))


# ---- build the feedback sample for one date column (normalized [0,1]) ----

def stale_quantile_dates(args, col: str) -> List[date]:
    rows = psql(args, f"SELECT v FROM stale_quantiles WHERE col='{col}' ORDER BY q")
    return [parse_date(x) for x in rows.splitlines() if x]


def fresh_quantile_dates(args, table: str, column: str) -> List[date]:
    levels = "ARRAY[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]"
    out = psql(args, f"SELECT percentile_disc({levels}) WITHIN GROUP (ORDER BY {column}) FROM {table}")
    # psql prints a postgres array literal like {1992-01-02,...}
    inner = out.strip().strip("{}")
    return [parse_date(x) for x in inner.split(",") if x]


def column_total(args, table: str) -> int:
    return int(psql(args, f"SELECT count(*) FROM {table}"))


def actual_le_selectivity(args, table: str, column: str, d: date, total: int) -> float:
    c = int(psql(args, f"SELECT count(*) FROM {table} WHERE {column} <= DATE '{d.isoformat()}'"))
    return c / max(total, 1)


def build_sample_for_column(args, logical: str) -> Tuple[KllFeedbackSample, int, int, List[float], List[float]]:
    table, column = TARGETS[logical]
    stale_q = stale_quantile_dates(args, f"{table}.{column}")
    fresh_q = fresh_quantile_dates(args, table, column)
    total = column_total(args, table)

    d0 = min(to_epoch_days(stale_q[0]), to_epoch_days(fresh_q[0]))
    d1 = max(to_epoch_days(stale_q[-1]), to_epoch_days(fresh_q[-1]))

    stale_norm = [norm(d, d0, d1) for d in stale_q]
    fresh_norm = [norm(d, d0, d1) for d in fresh_q]

    # feedback predicates at probability levels of the fresh distribution
    probs = [0.05, 0.15, 0.30, 0.45, 0.60, 0.72, 0.82, 0.90, 0.95]
    levels = "ARRAY[" + ",".join(str(p) for p in probs) + "]"
    fresh_at = psql(args, f"SELECT percentile_disc({levels}) WITHIN GROUP (ORDER BY {column}) FROM {table}")
    fresh_dates = [parse_date(x) for x in fresh_at.strip().strip("{}").split(",") if x]

    stale_internal = stale_norm[1:-1]
    stale_full_levels = cdf_levels(stale_norm)  # for estimated selectivity via piecewise CDF
    observations: List[FeedbackObservation] = []
    from datetime import datetime, timezone, timedelta as _td
    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    for i, (p, dt) in enumerate(zip(probs, fresh_dates)):
        u = norm(dt, d0, d1)
        actual = actual_le_selectivity(args, table, column, dt, total)
        est = float(evaluate_piecewise_cdf(stale_norm, stale_full_levels, u))
        observations.append(FeedbackObservation(
            predicate_type="<=", value=u, value_upper=None,
            actual_selectivity=actual, estimated_selectivity=est,
            timestamp=base + _td(minutes=i),
        ))

    sample = KllFeedbackSample(
        prior=KllPrior(min_value=0.0, max_value=1.0, null_fraction=0.0,
                       quantile_levels=list(DEFAULT_QUANTILE_LEVELS),
                       quantile_values=stale_internal, value_type="double"),
        observations=observations,
        corrected_quantile_values=fresh_norm[1:-1],
    )
    return sample, d0, d1, stale_norm, fresh_norm


# ---- inject a corrected date histogram for one method/column ----

def inject_method_column(args, logical: str, method: str, boundaries_norm: List[float], d0: int, d1: int) -> None:
    table, column = TARGETS[logical]
    src = f"bsrc_{method}_{logical}"
    stable_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(f"{method}:{logical}")) % 100000
    samples = pg.sample_values_from_boundaries(boundaries_norm, count=args.stat_source_rows,
                                               seed=args.seed + stable_offset)
    dates = [denorm_to_date(u, d0, d1) for u in samples]
    psql(args, f"DROP TABLE IF EXISTS {src}; CREATE TABLE {src} (d date);")
    # bulk insert via COPY
    cmd = [str(args.pg_bin / "psql"), "-X", "-h", str(args.socket_dir), "-p", str(args.pg_port),
           "-U", args.pg_user, "-d", args.dbname, "-c", f"COPY {src}(d) FROM STDIN"]
    env = {**_os_environ(), "LD_LIBRARY_PATH": str(args.pg_bin.parent / "lib")}
    payload = "\n".join(d.isoformat() for d in dates) + "\n"
    proc = subprocess.run(cmd, input=payload, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"COPY into {src} failed: {proc.stderr}")
    psql(args, f"ALTER TABLE {src} ALTER COLUMN d SET STATISTICS 100; ANALYZE {src};")
    # copy the typed pg_statistic row onto the real column
    psql(args, _copy_stat_sql(table, column, src, "d"))


def _copy_stat_sql(tgt_table: str, tgt_col: str, src_table: str, src_col: str) -> str:
    sets = ",\n".join(
        [f"sta{k} = src.sta{k}" for k in
         ["nullfrac", "width", "distinct"]] +
        [f"sta{p}{i} = src.sta{p}{i}" for p in ["kind", "op", "coll", "numbers", "values"] for i in range(1, 6)]
    )
    return f"""
UPDATE pg_statistic AS tgt SET
{sets}
FROM pg_statistic AS src
WHERE tgt.starelid='{tgt_table}'::regclass
  AND tgt.staattnum=(SELECT attnum FROM pg_attribute WHERE attrelid='{tgt_table}'::regclass AND attname='{tgt_col}')
  AND tgt.stainherit=false
  AND src.starelid='{src_table}'::regclass
  AND src.staattnum=(SELECT attnum FROM pg_attribute WHERE attrelid='{src_table}'::regclass AND attname='{src_col}')
  AND src.stainherit=false;
"""


# ---- curated TPC-H queries (date-predicate driven) ----

def curated_queries() -> List[dict]:
    return [
        {"name": "Q6_lineitem_revenue", "target": "l_shipdate",
         "scan": "SELECT count(*) FROM lineitem WHERE l_shipdate >= DATE '1999-06-01' AND l_shipdate < DATE '2000-06-01'",
         "sql": "SELECT sum(l_extendedprice*l_discount) FROM lineitem WHERE l_shipdate >= DATE '1999-06-01' AND l_shipdate < DATE '2000-06-01' AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24"},
        {"name": "Q14_promo_part_join", "target": "l_shipdate",
         "scan": "SELECT count(*) FROM lineitem WHERE l_shipdate >= DATE '1999-06-01' AND l_shipdate < DATE '1999-07-01'",
         "sql": "SELECT 100.0*sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice*(1-l_discount) ELSE 0 END)/NULLIF(sum(l_extendedprice*(1-l_discount)),0) FROM lineitem, part WHERE l_partkey=p_partkey AND l_shipdate >= DATE '1999-06-01' AND l_shipdate < DATE '1999-07-01'"},
        {"name": "Q12_shipmode_orders_join", "target": "l_shipdate",
         "scan": "SELECT count(*) FROM lineitem WHERE l_shipdate >= DATE '1999-06-01' AND l_shipdate < DATE '2000-06-01'",
         "sql": "SELECT l_shipmode, count(*) FROM orders, lineitem WHERE o_orderkey=l_orderkey AND l_shipdate >= DATE '1999-06-01' AND l_shipdate < DATE '2000-06-01' GROUP BY l_shipmode"},
        {"name": "Q4_order_priority", "target": "o_orderdate",
         "scan": "SELECT count(*) FROM orders WHERE o_orderdate >= DATE '1999-06-01' AND o_orderdate < DATE '1999-09-01'",
         "sql": "SELECT o_orderpriority, count(*) FROM orders WHERE o_orderdate >= DATE '1999-06-01' AND o_orderdate < DATE '1999-09-01' AND EXISTS (SELECT 1 FROM lineitem WHERE l_orderkey=o_orderkey AND l_commitdate < l_receiptdate) GROUP BY o_orderpriority"},
        {"name": "Q3_3way_join", "target": "o_orderdate",
         "scan": "SELECT count(*) FROM orders WHERE o_orderdate < DATE '2000-01-01'",
         "sql": "SELECT l.l_orderkey, sum(l.l_extendedprice*(1-l.l_discount)) AS rev FROM customer c, orders o, lineitem l WHERE c.c_custkey=o.o_custkey AND l.l_orderkey=o.o_orderkey AND o.o_orderdate < DATE '2000-01-01' AND l.l_shipdate > DATE '1999-06-01' GROUP BY l.l_orderkey ORDER BY rev DESC LIMIT 20"},
        {"name": "Q1_pricing_scan", "target": "l_shipdate",
         "scan": "SELECT count(*) FROM lineitem WHERE l_shipdate <= DATE '2000-09-01'",
         "sql": "SELECT l_returnflag, l_linestatus, count(*), sum(l_quantity) FROM lineitem WHERE l_shipdate <= DATE '2000-09-01' GROUP BY l_returnflag, l_linestatus"},
    ]


def explain_plan(args, sql: str) -> dict:
    out = psql(args, planner_prefix(args) + f"EXPLAIN (FORMAT JSON) {sql}")
    return json.loads(out)[0]["Plan"]


def explain_analyze_ms(args, sql: str) -> Optional[float]:
    try:
        out = psql(args, planner_prefix(args) + f"EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON) {sql}")
        return float(json.loads(out)[0]["Execution Time"])
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[warn] analyze failed: {exc}\n")
        return None


def scan_estimate(args, scan_count_sql: str, table: str) -> float:
    # EXPLAIN the count query; find the scan node on `table` and read its Plan Rows
    plan = explain_plan(args, scan_count_sql)
    best = 0.0
    for node in pg.plan_nodes(plan):
        if node.get("Relation Name") == table:
            best = max(best, float(node.get("Plan Rows", 0.0)))
    return best


def timed_median(args, sql: str) -> Dict[str, float]:
    for _ in range(args.warmups):
        explain_analyze_ms(args, sql)
    samples = [s for s in (explain_analyze_ms(args, sql) for _ in range(args.timing_runs)) if s is not None]
    if not samples:
        return {"median": float("nan"), "min": float("nan"), "iqr": float("nan"), "n": 0}
    ordered = sorted(samples)
    trimmed = ordered[1:-1] if len(ordered) >= 5 else ordered
    q1 = statistics.quantiles(ordered, n=4)[0] if len(ordered) >= 4 else ordered[0]
    q3 = statistics.quantiles(ordered, n=4)[2] if len(ordered) >= 4 else ordered[-1]
    return {"median": statistics.median(trimmed), "min": ordered[0], "iqr": q3 - q1, "n": len(samples)}


def run(args) -> None:
    model = MlpHistogramModelV2.load(str(args.model_path))

    # build per-column corrected boundaries + domain
    col_boundaries: Dict[str, Dict[str, List[float]]] = {}
    col_domain: Dict[str, Tuple[int, int]] = {}
    for logical in TARGETS:
        sample, d0, d1, stale_norm, fresh_norm = build_sample_for_column(args, logical)
        boundaries, _ = build_method_boundaries(sample, model, args.num_buckets, args.max_observations)
        # Faithful stale/fresh use the REAL normalized quantiles (not the [0,1]-forced
        # reconstruction), so the stale histogram truncates at its true pre-drift max
        # instead of being stretched across the drifted domain.
        boundaries["stale"] = list(stale_norm)
        boundaries["fresh"] = list(fresh_norm)
        col_boundaries[logical] = boundaries
        col_domain[logical] = (d0, d1)
        print(f"[info] built boundaries for {logical} (domain {d0}..{d1} epoch-days, "
              f"stale_max_norm={stale_norm[-1]:.3f})")

    queries = curated_queries()
    # ground-truth scan counts (independent of injected stats)
    scan_truth = {q["name"]: int(psql(args, q["scan"])) for q in queries}
    scan_table = {q["name"]: TARGETS[q["target"]][0] for q in queries}

    # capture plan signatures per method (cheap EXPLAIN)
    plan_sig: Dict[str, Dict[str, str]] = {m: {} for m in B_METHODS}
    scan_est: Dict[str, Dict[str, float]] = {m: {} for m in B_METHODS}
    for method in B_METHODS:
        for logical in TARGETS:
            inject_method_column(args, logical, method, col_boundaries[logical][method], *col_domain[logical])
        for q in queries:
            plan = explain_plan(args, q["sql"])
            plan_sig[method][q["name"]] = pg.plan_signature(plan)
            scan_est[method][q["name"]] = scan_estimate(args, q["scan"], scan_table[q["name"]])
        print(f"[info] captured plans for method={method}")

    changed = {q["name"]: plan_sig["stale"][q["name"]] != plan_sig["fresh"][q["name"]] for q in queries}
    print(f"[info] {sum(changed.values())}/{len(queries)} queries change plan stale->fresh")

    # timing pass — dedup by plan signature: a query's warm-cache runtime depends
    # ONLY on the plan the injected stats induce, so methods that produce an
    # identical plan have identical runtime by construction. We therefore time
    # each DISTINCT (query, plan) once at full timing_runs and assign it to every
    # method sharing that plan. This is methodologically equivalent (same 7-run
    # median per distinct plan) but avoids re-timing identical plans, and reduces
    # timing noise. Progress is logged per (query, plan).
    rows: List[BRow] = []
    plan_time: Dict[Tuple[str, str], Dict[str, float]] = {}
    total_plans = sum(len({plan_sig[m][q["name"]] for m in B_METHODS}) for q in queries)
    done_plans = 0
    for q in queries:
        qname = q["name"]
        rep_by_sig: Dict[str, str] = {}
        for method in B_METHODS:
            rep_by_sig.setdefault(plan_sig[method][qname], method)  # first method per distinct plan
        for sig, rep in rep_by_sig.items():
            for logical in TARGETS:
                inject_method_column(args, logical, rep, col_boundaries[logical][rep], *col_domain[logical])
            t0 = time.time()
            plan_time[(qname, sig)] = timed_median(args, q["sql"])
            done_plans += 1
            print(f"[time {done_plans}/{total_plans}] {qname} plan={sig[:10]} rep={rep} "
                  f"median={plan_time[(qname, sig)]['median']:.0f}ms "
                  f"({len(rep_by_sig)} distinct plans for {qname}, {time.time()-t0:.0f}s)",
                  flush=True)
        for method in B_METHODS:
            t = plan_time[(qname, plan_sig[method][qname])]
            est = scan_est[method][qname]
            truth = scan_truth[qname]
            rows.append(BRow(
                query=qname, method=method, target_col=q["target"],
                scan_true_rows=truth, scan_est_rows=est,
                scan_row_qerr=pg.card_qerr(est, truth),
                plan_signature=plan_sig[method][qname],
                plan_changed_vs_stale_fresh=changed[qname],
                median_ms=t["median"], min_ms=t["min"], iqr_ms=t["iqr"], n_runs=int(t["n"]),
            ))
    print(f"[info] timed {total_plans} distinct (query,plan) pairs "
          f"(vs {len(B_METHODS)*len(queries)} method-query cells)")

    summary = summarize(rows, changed, queries)
    write_outputs(args, rows, summary)
    print_summary(summary)


def summarize(rows: Sequence[BRow], changed: Dict[str, bool], queries) -> List[dict]:
    by_q: Dict[str, Dict[str, BRow]] = {}
    for r in rows:
        by_q.setdefault(r.query, {})[r.method] = r
    out = []
    for subset in ("all", "plan_change", "plan_stable"):
        qids = [q["name"] for q in queries
                if subset == "all" or (subset == "plan_change") == changed[q["name"]]]
        for method in B_METHODS:
            ratios_stale, ratios_fresh, med, qerrs = [], [], [], []
            regress = 0
            for qid in qids:
                m, s, fr = by_q[qid][method], by_q[qid]["stale"], by_q[qid]["fresh"]
                if m.median_ms == m.median_ms:
                    med.append(m.median_ms)
                    if s.median_ms > 0:
                        ratios_stale.append(m.median_ms / s.median_ms)
                        if m.median_ms > s.median_ms * 1.15:
                            regress += 1
                    if fr.median_ms > 0:
                        ratios_fresh.append(m.median_ms / fr.median_ms)
                qerrs.append(m.scan_row_qerr)
            out.append({
                "subset": subset, "method": method, "n_queries": len(qids),
                "scan_qerr_gm": round(float(np.exp(np.mean(np.log(np.maximum(qerrs, 1.0))))), 4) if qerrs else None,
                "total_median_ms": round(sum(med), 2),
                "geomean_ratio_vs_stale": round(float(np.exp(np.mean(np.log(ratios_stale)))), 4) if ratios_stale else None,
                "geomean_ratio_vs_fresh": round(float(np.exp(np.mean(np.log(ratios_fresh)))), 4) if ratios_fresh else None,
                "n_regressions_gt15pct_vs_stale": regress,
            })
    return out


def write_outputs(args, rows, summary) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "tpch_runtime_rows.json", "w") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)
    with open(args.output_dir / "tpch_runtime_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys())); w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    with open(args.output_dir / "tpch_runtime_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(args.output_dir / "tpch_runtime_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)


def print_summary(summary) -> None:
    print("\n=== TPC-H runtime summary (warm median Execution Time) ===")
    for subset in ("all", "plan_change", "plan_stable"):
        print(f"\n[{subset}]")
        for s in summary:
            if s["subset"] != subset:
                continue
            print(f"  {s['method']:<18} scanQErr={s['scan_qerr_gm']:<8} totalms={s['total_median_ms']:<10} "
                  f"vs_stale={s['geomean_ratio_vs_stale']} vs_fresh={s['geomean_ratio_vs_fresh']} "
                  f"regress>15%={s['n_regressions_gt15pct_vs_stale']}/{s['n_queries']}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pg-bin", type=Path, default=Path("/home/tianqc/deps/install/bin"))
    p.add_argument("--socket-dir", type=Path, default=Path("/home/tianqc/oasis_pg/run"))
    p.add_argument("--pg-port", type=int, default=55432)
    p.add_argument("--pg-user", type=str, default="tianqc")
    p.add_argument("--dbname", type=str, default="tpch")
    p.add_argument("--model-path", type=Path,
                   default=_REPO_DIR / "experiments" / "results" / "synthetic_paper_suite_rerun_20260529" / "models" / "oasis_k16.json")
    p.add_argument("--num-buckets", type=int, default=10)
    p.add_argument("--max-observations", type=int, default=16)
    p.add_argument("--stat-source-rows", type=int, default=200000)
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--parallel-workers", type=int, default=0)
    p.add_argument("--random-page-cost", type=float, default=1.1)
    p.add_argument("--effective-cache-size", type=str, default="12GB")
    p.add_argument("--timing-runs", type=int, default=7)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--output-dir", type=Path, default=_REPO_DIR / "experiments" / "results" / "postgres_runtime_tpch")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
