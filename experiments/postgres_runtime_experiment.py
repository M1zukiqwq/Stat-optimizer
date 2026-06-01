#!/usr/bin/env python3
"""Runtime sanity check for OASIS statistics injection (Experiment A).

This is an ADDITIVE companion to ``postgres_planner_stats_injection_experiment.py``.
It reuses that harness's data generation, feedback construction, Stage-2 boundary
building, and typed ``pg_statistic`` injection, and adds a controlled warm-cache
``EXPLAIN ANALYZE`` runtime measurement so we can answer the reviewer question:

  Does the plan-shape improvement translate into runtime, or at least *not*
  introduce regressions?

The claim this supports is a *no-regression / directional-translation* claim, not a
runtime-superiority claim. No existing experiment, number, or table is modified.

Method labels (matching the paper): oasis = OASIS-noProj (raw learned prior),
oasis_projected = OASIS (learned prior + hard projection), calibrated_hybrid = Router.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_PIPELINE_DIR = _REPO_DIR / "cdf_kll_ml_pipeline"
for p in (_SCRIPT_DIR, _PIPELINE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import postgres_planner_stats_injection_experiment as pg  # noqa: E402
from optimizer_decision_proxy_experiment import (  # noqa: E402
    METHOD_ORDER,
    build_method_boundaries,
)
from mlp_histogram_model_v2 import MlpHistogramModelV2  # noqa: E402


@dataclass
class RuntimeRow:
    config_id: str
    drift_family: str
    rows: int
    query_id: str
    family: str
    method: str
    true_rows: int
    plan_rows: float
    row_qerr: float
    plan_signature: str
    total_cost: float
    median_ms: float
    min_ms: float
    iqr_ms: float
    n_runs: int
    plan_changed_vs_stale_fresh: bool


def explain_analyze_ms(args: argparse.Namespace, sql: str) -> Optional[float]:
    """Return server-side Execution Time (ms) from EXPLAIN ANALYZE, or None."""
    stmt = pg.planner_prefix(args) + f"EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON) {sql}"
    try:
        out = pg.psql_scalar(args, args.dbname, stmt)
        payload = json.loads(out)
        return float(payload[0]["Execution Time"])
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[warn] EXPLAIN ANALYZE failed: {exc}\n")
        return None


def timed_median(args: argparse.Namespace, sql: str, n_runs: int, warmups: int) -> Dict[str, float]:
    for _ in range(warmups):
        explain_analyze_ms(args, sql)
    samples = [explain_analyze_ms(args, sql) for _ in range(n_runs)]
    samples = [s for s in samples if s is not None]
    if not samples:
        return {"median": float("nan"), "min": float("nan"), "iqr": float("nan"), "n": 0}
    ordered = sorted(samples)
    # drop one min and one max when we have enough samples, then take the median
    trimmed = ordered[1:-1] if len(ordered) >= 5 else ordered
    q1 = statistics.quantiles(ordered, n=4)[0] if len(ordered) >= 4 else ordered[0]
    q3 = statistics.quantiles(ordered, n=4)[2] if len(ordered) >= 4 else ordered[-1]
    return {
        "median": statistics.median(trimmed),
        "min": ordered[0],
        "iqr": q3 - q1,
        "n": len(samples),
    }


def run(args: argparse.Namespace) -> None:
    pg.ensure_postgres(args)
    pg.setup_schema(args)

    initial, fresh, dim_ids = pg.generate_data(args.rows, args.dim_rows, args.seed, args.drift_family)
    pg.load_base_tables(args, fresh, dim_ids)

    # Give the planner a real scan/join choice so plan shape can move runtime.
    pg.psql_exec(args, f"""
CREATE INDEX IF NOT EXISTS fact_x_idx ON {pg.SCHEMA}.fact(x);
CREATE INDEX IF NOT EXISTS fact_dim_idx ON {pg.SCHEMA}.fact(dim_id);
ANALYZE {pg.SCHEMA}.fact;
ANALYZE {pg.SCHEMA}.dim;
""")

    model = MlpHistogramModelV2.load(str(args.model_path))
    sample = pg.make_feedback_sample(initial, fresh, args)
    method_boundaries, hybrid_choice = build_method_boundaries(
        sample, model, args.num_buckets, args.max_observations
    )
    pg.load_all_stat_sources(args, initial, fresh, method_boundaries)

    predicates = pg.generate_predicates(method_boundaries["fresh"])
    queries = pg.build_queries(predicates)

    # Ground-truth row counts (independent of injected stats).
    true_rows = {q.query_id: pg.count_rows(args, q.count_sql) for q in queries}

    # First pass: capture plan signatures per method/query (cheap EXPLAIN, no ANALYZE)
    plan_sig: Dict[str, Dict[str, str]] = {}
    plan_rows_est: Dict[str, Dict[str, float]] = {}
    plan_cost: Dict[str, Dict[str, float]] = {}
    for method in METHOD_ORDER:
        pg.copy_pg_statistic_from_source(args, method)
        plan_sig[method] = {}
        plan_rows_est[method] = {}
        plan_cost[method] = {}
        for q in queries:
            plan = pg.explain_json(args, q.select_sql)
            plan_sig[method][q.query_id] = pg.plan_signature(plan)
            plan_rows_est[method][q.query_id] = float(plan.get("Plan Rows", 0.0))
            plan_cost[method][q.query_id] = float(plan.get("Total Cost", 0.0))

    changed = {
        q.query_id: plan_sig["stale"][q.query_id] != plan_sig["fresh"][q.query_id]
        for q in queries
    }
    n_changed = sum(changed.values())
    print(f"[info] {n_changed}/{len(queries)} queries change plan shape stale->fresh")

    # Second pass: warm-cache EXPLAIN ANALYZE timing per method/query.
    rows: List[RuntimeRow] = []
    for method in METHOD_ORDER:
        pg.copy_pg_statistic_from_source(args, method)
        for q in queries:
            t = timed_median(args, q.select_sql, args.timing_runs, args.warmups)
            rows.append(RuntimeRow(
                config_id=args.config_id,
                drift_family=args.drift_family,
                rows=args.rows,
                query_id=q.query_id,
                family=q.family,
                method=method,
                true_rows=true_rows[q.query_id],
                plan_rows=plan_rows_est[method][q.query_id],
                row_qerr=pg.card_qerr(plan_rows_est[method][q.query_id], true_rows[q.query_id]),
                plan_signature=plan_sig[method][q.query_id],
                total_cost=plan_cost[method][q.query_id],
                median_ms=t["median"],
                min_ms=t["min"],
                iqr_ms=t["iqr"],
                n_runs=int(t["n"]),
                plan_changed_vs_stale_fresh=changed[q.query_id],
            ))
        done = METHOD_ORDER.index(method) + 1
        print(f"[info] timed method {method} ({done}/{len(METHOD_ORDER)})")

    summary = summarize(rows, changed)
    write_outputs(args, rows, summary, hybrid_choice, n_changed, len(queries))
    print_summary(summary)


def summarize(rows: Sequence[RuntimeRow], changed: Dict[str, bool]) -> List[dict]:
    by_q: Dict[str, Dict[str, RuntimeRow]] = {}
    for r in rows:
        by_q.setdefault(r.query_id, {})[r.method] = r

    out: List[dict] = []
    for subset in ("all", "plan_change", "plan_stable"):
        qids = [
            qid for qid in by_q
            if subset == "all"
            or (subset == "plan_change" and changed[qid])
            or (subset == "plan_stable" and not changed[qid])
        ]
        for method in METHOD_ORDER:
            ratios_stale, ratios_fresh, med_times = [], [], []
            regressions = 0
            for qid in qids:
                m = by_q[qid][method]
                s = by_q[qid]["stale"]
                fr = by_q[qid]["fresh"]
                if not (m.median_ms == m.median_ms):  # NaN guard
                    continue
                med_times.append(m.median_ms)
                if s.median_ms > 0:
                    ratios_stale.append(m.median_ms / s.median_ms)
                    if m.median_ms > s.median_ms * 1.15:
                        regressions += 1
                if fr.median_ms > 0:
                    ratios_fresh.append(m.median_ms / fr.median_ms)
            out.append({
                "subset": subset,
                "method": method,
                "n_queries": len(qids),
                "total_median_ms": round(sum(med_times), 3),
                "geomean_ratio_vs_stale": round(float(np.exp(np.mean(np.log(ratios_stale)))), 4) if ratios_stale else None,
                "geomean_ratio_vs_fresh": round(float(np.exp(np.mean(np.log(ratios_fresh)))), 4) if ratios_fresh else None,
                "n_regressions_gt15pct_vs_stale": regressions,
            })
    return out


def write_outputs(args, rows, summary, hybrid_choice, n_changed, n_queries) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "runtime_rows.json", "w") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)
    with open(args.output_dir / "runtime_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    with open(args.output_dir / "runtime_summary.json", "w") as f:
        json.dump({
            "config_id": args.config_id,
            "drift_family": args.drift_family,
            "rows": args.rows,
            "timing_runs": args.timing_runs,
            "hybrid_choice": hybrid_choice,
            "n_plan_change_queries": n_changed,
            "n_queries": n_queries,
            "summary": summary,
        }, f, indent=2)
    with open(args.output_dir / "runtime_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)


def print_summary(summary: Sequence[dict]) -> None:
    print("\n=== runtime summary (warm median Execution Time) ===")
    for subset in ("all", "plan_change", "plan_stable"):
        print(f"\n[{subset}]")
        for s in summary:
            if s["subset"] != subset:
                continue
            print(f"  {s['method']:<18} totalms={s['total_median_ms']:<10} "
                  f"vs_stale={s['geomean_ratio_vs_stale']} vs_fresh={s['geomean_ratio_vs_fresh']} "
                  f"regress>15%={s['n_regressions_gt15pct_vs_stale']}/{s['n_queries']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pg-bin", type=Path, default=Path("/home/tianqc/deps/install/bin"))
    p.add_argument("--data-dir", type=Path, default=Path("/home/tianqc/oasis_pg/data"))
    p.add_argument("--socket-dir", type=Path, default=Path("/home/tianqc/oasis_pg/run"))
    p.add_argument("--log-dir", type=Path, default=Path("/home/tianqc/oasis_pg/logs"))
    p.add_argument("--pg-port", type=int, default=55432)
    p.add_argument("--pg-user", type=str, default="tianqc")
    p.add_argument("--dbname", type=str, default="oasis_eval")
    p.add_argument("--model-path", type=Path,
                   default=_REPO_DIR / "experiments" / "results" / "synthetic_paper_suite_rerun_20260529" / "models" / "oasis_k16.json")
    p.add_argument("--rows", type=int, default=2_000_000)
    p.add_argument("--dim-rows", type=int, default=50_000)
    p.add_argument("--stat-source-rows", type=int, default=200_000)
    p.add_argument("--drift-family", type=str, default="left_shift")
    p.add_argument("--config-id", type=str, default="runtimeA")
    p.add_argument("--num-buckets", type=int, default=10)
    p.add_argument("--max-observations", type=int, default=16)
    p.add_argument("--pg-statistics-target", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--random-page-cost", type=float, default=1.1)
    p.add_argument("--effective-cache-size", type=str, default="12GB")
    p.add_argument("--timing-runs", type=int, default=7)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=_REPO_DIR / "experiments" / "results" / "postgres_runtime_A")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
