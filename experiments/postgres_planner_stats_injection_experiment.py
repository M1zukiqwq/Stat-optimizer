#!/usr/bin/env python3
"""
PostgreSQL planner-only evidence for OASIS-style marginal correction.

This experiment deliberately avoids end-to-end runtime claims. It asks whether
statistics corrected outside the DBMS can change PostgreSQL's optimizer-facing
signals when injected as single-column statistics:

  * true cardinalities come from COUNT(*);
  * estimated cardinalities and plan shapes come from EXPLAIN (FORMAT JSON);
  * no query wall-clock time is measured or reported.

The script creates a local synthetic table whose current data follows the
"fresh" distribution, then injects alternative pg_statistic rows for fact.x:
stale, ISOMER, OASIS-noProj, OASIS, Soft, Hybrid, Aggressive, Router, and fresh. PostgreSQL therefore uses
its real planner and cost model, while the experiment remains narrowly about
planner evidence rather than execution latency.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_PIPELINE_DIR = _REPO_DIR / "cdf_kll_ml_pipeline"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from histogram_math import evaluate_piecewise_cdf, inverse_piecewise_cdf, project_monotonic
from histogram_types import DEFAULT_QUANTILE_LEVELS, FeedbackObservation, KllFeedbackSample, KllPrior
from mlp_histogram_model_v2 import MlpHistogramModelV2
from optimizer_decision_proxy_experiment import (
    METHOD_ORDER,
    build_method_boundaries,
    cdf_levels,
    estimate_selectivity,
    geomean,
    pct_improvement,
)


SCHEMA = "oasis_pg_evidence"
SOURCE_TABLE_BY_METHOD = {
    "stale": "stat_source_stale",
    "isomer": "stat_source_isomer",
    "oasis": "stat_source_oasis",
    "oasis_projected": "stat_source_oasis_projected",
    "oasis_soft_projection": "stat_source_oasis_soft_projection",
    "hybrid": "stat_source_hybrid",
    "aggressive_hybrid": "stat_source_aggressive_hybrid",
    "calibrated_hybrid": "stat_source_calibrated_hybrid",
    "fresh": "stat_source_fresh",
}


@dataclass
class QuerySpec:
    query_id: str
    family: str
    predicate_id: str
    predicate_type: str
    value: float
    value_upper: Optional[float]
    select_sql: str
    count_sql: str


@dataclass
class PlanRow:
    config_id: str
    seed: int
    row_count: int
    dim_rows: int
    drift_family: str
    query_id: str
    family: str
    predicate_id: str
    predicate_type: str
    value: float
    value_upper: Optional[float]
    true_rows: int
    method: str
    plan_rows: float
    row_qerr: float
    total_cost: float
    startup_cost: float
    root_node: str
    scan_nodes: str
    join_nodes: str
    plan_signature: str


def run_cmd(args: Sequence[Path | str], *, input_text: Optional[str] = None, quiet: bool = False) -> str:
    proc = subprocess.run(
        [str(arg) for arg in args],
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        cmd = " ".join(str(arg) for arg in args)
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    if proc.stderr and not quiet:
        sys.stderr.write(proc.stderr)
    return proc.stdout


def pg_bin(args: argparse.Namespace, name: str) -> Path:
    return args.pg_bin / name


def ensure_postgres(args: argparse.Namespace) -> None:
    args.socket_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    if not (args.data_dir / "PG_VERSION").exists():
        args.data_dir.parent.mkdir(parents=True, exist_ok=True)
        run_cmd([
            pg_bin(args, "initdb"),
            "-D",
            args.data_dir,
            "-U",
            args.pg_user,
            "--no-locale",
            "-E",
            "UTF8",
        ])

    ready_cmd = [
        pg_bin(args, "pg_isready"),
        "-h",
        args.socket_dir,
        "-p",
        str(args.pg_port),
        "-U",
        args.pg_user,
    ]
    ready = subprocess.run([str(arg) for arg in ready_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if ready.returncode != 0:
        run_cmd([
            pg_bin(args, "pg_ctl"),
            "-D",
            args.data_dir,
            "-l",
            args.log_dir / "postgres.log",
            "-o",
            f"-p {args.pg_port} -k {args.socket_dir}",
            "start",
        ])
        for _ in range(30):
            ready = subprocess.run([str(arg) for arg in ready_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if ready.returncode == 0:
                break
            time.sleep(0.25)
        if ready.returncode != 0:
            raise RuntimeError(f"PostgreSQL did not become ready:\n{ready.stdout}\n{ready.stderr}")

    exists = psql_scalar(args, "postgres", f"SELECT 1 FROM pg_database WHERE datname = '{args.dbname}'")
    if exists.strip() != "1":
        run_cmd([
            pg_bin(args, "createdb"),
            "-h",
            args.socket_dir,
            "-p",
            str(args.pg_port),
            "-U",
            args.pg_user,
            args.dbname,
        ])


def psql_base(args: argparse.Namespace, dbname: Optional[str] = None) -> List[str]:
    return [
        str(pg_bin(args, "psql")),
        "-X",
        "-v",
        "ON_ERROR_STOP=1",
        "-h",
        str(args.socket_dir),
        "-p",
        str(args.pg_port),
        "-U",
        args.pg_user,
        "-d",
        dbname or args.dbname,
    ]


def psql_exec(args: argparse.Namespace, sql: str, *, dbname: Optional[str] = None) -> str:
    return run_cmd(psql_base(args, dbname) + ["-q", "-c", sql], quiet=True)


def psql_scalar(args: argparse.Namespace, dbname: str, sql: str) -> str:
    return run_cmd(psql_base(args, dbname) + ["-q", "-A", "-t", "-c", sql], quiet=True).strip()


def psql_copy(args: argparse.Namespace, table: str, columns: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    csv_lines = []
    for row in rows:
        csv_lines.append(",".join(format_copy_value(value) for value in row))
    csv_text = "\n".join(csv_lines) + ("\n" if csv_lines else "")
    columns_sql = ", ".join(columns)
    sql = f"COPY {table} ({columns_sql}) FROM STDIN WITH (FORMAT csv)"
    run_cmd(psql_base(args) + ["-q", "-c", sql], input_text=csv_text, quiet=True)


def format_copy_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.17g}"
    return str(value)


def setup_schema(args: argparse.Namespace) -> None:
    sql = f"""
DROP SCHEMA IF EXISTS {SCHEMA} CASCADE;
CREATE SCHEMA {SCHEMA};
CREATE TABLE {SCHEMA}.dim (
    dim_id integer PRIMARY KEY,
    payload integer NOT NULL
);
CREATE TABLE {SCHEMA}.fact (
    id integer PRIMARY KEY,
    x double precision NOT NULL,
    dim_id integer NOT NULL
);
ALTER TABLE {SCHEMA}.fact SET (autovacuum_enabled = false);
CREATE INDEX dim_payload_idx ON {SCHEMA}.dim(payload);
"""
    for table in SOURCE_TABLE_BY_METHOD.values():
        sql += f"CREATE TABLE {SCHEMA}.{table} (x double precision NOT NULL);\n"
    psql_exec(args, sql)


def generate_data(rows: int, dim_rows: int, seed: int, drift_family: str = "left_shift") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    if drift_family == "left_shift":
        initial = rng.beta(5.2, 1.7, rows)
        initial_mix = rng.random(rows) < 0.22
        initial[initial_mix] = rng.normal(0.58, 0.055, int(initial_mix.sum()))

        fresh = rng.beta(1.35, 5.4, rows)
        fresh_mix = rng.random(rows) < 0.16
        fresh[fresh_mix] = rng.normal(0.68, 0.08, int(fresh_mix.sum()))
    elif drift_family == "right_shift":
        initial = rng.beta(1.35, 5.4, rows)
        initial_mix = rng.random(rows) < 0.16
        initial[initial_mix] = rng.normal(0.68, 0.08, int(initial_mix.sum()))

        fresh = rng.beta(5.2, 1.7, rows)
        fresh_mix = rng.random(rows) < 0.22
        fresh[fresh_mix] = rng.normal(0.58, 0.055, int(fresh_mix.sum()))
    elif drift_family == "bimodal_shift":
        initial = np.empty(rows, dtype=float)
        left_mask = rng.random(rows) < 0.52
        initial[left_mask] = rng.normal(0.24, 0.055, int(left_mask.sum()))
        initial[~left_mask] = rng.normal(0.78, 0.065, int((~left_mask).sum()))

        fresh = np.empty(rows, dtype=float)
        center_mask = rng.random(rows) < 0.68
        fresh[center_mask] = rng.normal(0.42, 0.085, int(center_mask.sum()))
        fresh[~center_mask] = rng.beta(6.0, 2.4, int((~center_mask).sum()))
    else:
        raise ValueError(f"unknown drift family: {drift_family}")

    initial = np.clip(initial, 1e-6, 1.0 - 1e-6)
    fresh = np.clip(fresh, 1e-6, 1.0 - 1e-6)
    dim_ids = rng.integers(1, dim_rows + 1, size=rows, dtype=np.int64)
    return initial.astype(float), fresh.astype(float), dim_ids


def load_base_tables(args: argparse.Namespace, fresh: np.ndarray, dim_ids: np.ndarray) -> None:
    psql_copy(
        args,
        f"{SCHEMA}.dim",
        ["dim_id", "payload"],
        ((idx, idx % 100) for idx in range(1, args.dim_rows + 1)),
    )
    psql_copy(
        args,
        f"{SCHEMA}.fact",
        ["id", "x", "dim_id"],
        ((idx + 1, float(fresh[idx]), int(dim_ids[idx])) for idx in range(len(fresh))),
    )
    psql_exec(
        args,
        f"""
CREATE INDEX fact_x_idx ON {SCHEMA}.fact(x);
CREATE INDEX fact_dim_idx ON {SCHEMA}.fact(dim_id);
ALTER TABLE {SCHEMA}.fact ALTER COLUMN x SET STATISTICS {args.pg_statistics_target};
ALTER TABLE {SCHEMA}.dim ALTER COLUMN payload SET STATISTICS {args.pg_statistics_target};
ANALYZE {SCHEMA}.dim;
ANALYZE {SCHEMA}.fact;
""",
    )


def quantile_boundaries(values: np.ndarray, num_buckets: int) -> List[float]:
    probs = np.linspace(0.0, 1.0, num_buckets + 1)
    boundaries = [float(v) for v in np.quantile(values, probs)]
    boundaries[0] = 0.0
    boundaries[-1] = 1.0
    return [float(v) for v in project_monotonic(boundaries)]


def sorted_selectivity(sorted_values: np.ndarray, predicate: dict) -> float:
    n = len(sorted_values)
    pred_type = predicate["predicate_type"]
    value = float(predicate["value"])
    if pred_type in {"<", "<="}:
        return float(np.searchsorted(sorted_values, value, side="right") / n)
    if pred_type in {">", ">="}:
        return float((n - np.searchsorted(sorted_values, value, side="left")) / n)
    if pred_type == "BETWEEN":
        upper = float(predicate["value_upper"])
        lo, hi = sorted((value, upper))
        left = np.searchsorted(sorted_values, lo, side="left")
        right = np.searchsorted(sorted_values, hi, side="right")
        return float((right - left) / n)
    left = np.searchsorted(sorted_values, value - 1e-4, side="left")
    right = np.searchsorted(sorted_values, value + 1e-4, side="right")
    return float((right - left) / n)


def make_feedback_sample(initial: np.ndarray, fresh: np.ndarray, args: argparse.Namespace) -> KllFeedbackSample:
    stale_boundaries = quantile_boundaries(initial, args.num_buckets)
    fresh_boundaries = quantile_boundaries(fresh, args.num_buckets)
    sorted_fresh = np.sort(fresh)
    levels = cdf_levels(fresh_boundaries)
    feedback_specs = [
        ("<=", 0.04, None),
        ("<=", 0.12, None),
        ("<=", 0.28, None),
        ("<=", 0.48, None),
        ("<=", 0.68, None),
        ("<=", 0.86, None),
        (">=", 0.08, None),
        (">=", 0.20, None),
        (">=", 0.38, None),
        (">=", 0.58, None),
        (">=", 0.78, None),
        (">=", 0.92, None),
        ("BETWEEN", 0.06, 0.18),
        ("BETWEEN", 0.24, 0.36),
        ("BETWEEN", 0.44, 0.60),
        ("BETWEEN", 0.70, 0.90),
    ]

    # Sparse-feedback sweep (Exp 2): keep an evenly spaced subset of the 16
    # feedback predicates so the projection sees genuinely fewer constraints
    # (not just a smaller MLP window). num_feedback>=len keeps all of them.
    num_feedback = getattr(args, "num_feedback", 0) or len(feedback_specs)
    if 0 < num_feedback < len(feedback_specs):
        if num_feedback > 1:
            idx_keep = sorted({round(i * (len(feedback_specs) - 1) / (num_feedback - 1))
                               for i in range(num_feedback)})
        else:
            idx_keep = [len(feedback_specs) // 2]
        feedback_specs = [feedback_specs[i] for i in idx_keep]

    observations: List[FeedbackObservation] = []
    base_time = datetime(2026, 5, 29, tzinfo=timezone.utc)
    for idx, (pred_type, prob, upper_prob) in enumerate(feedback_specs):
        if pred_type == "BETWEEN":
            lo = inverse_piecewise_cdf(fresh_boundaries, levels, float(prob))
            hi = inverse_piecewise_cdf(fresh_boundaries, levels, float(upper_prob))
            predicate = {"predicate_type": "BETWEEN", "value": lo, "value_upper": hi}
        elif pred_type == "<=":
            value = inverse_piecewise_cdf(fresh_boundaries, levels, float(prob))
            predicate = {"predicate_type": "<=", "value": value, "value_upper": None}
        else:
            value = inverse_piecewise_cdf(fresh_boundaries, levels, 1.0 - float(prob))
            predicate = {"predicate_type": ">=", "value": value, "value_upper": None}

        observations.append(FeedbackObservation(
            predicate_type=predicate["predicate_type"],
            value=float(predicate["value"]),
            value_upper=predicate.get("value_upper"),
            actual_selectivity=sorted_selectivity(sorted_fresh, predicate),
            estimated_selectivity=estimate_selectivity(stale_boundaries, predicate),
            timestamp=base_time + timedelta(minutes=idx),
        ))

    return KllFeedbackSample(
        prior=KllPrior(
            min_value=0.0,
            max_value=1.0,
            null_fraction=0.0,
            quantile_levels=list(DEFAULT_QUANTILE_LEVELS),
            quantile_values=stale_boundaries[1:-1],
            value_type="double",
        ),
        observations=observations,
        corrected_quantile_values=fresh_boundaries[1:-1],
    )


def sample_values_from_boundaries(boundaries: Sequence[float], count: int, seed: int) -> List[float]:
    levels = cdf_levels(boundaries)
    values = [
        inverse_piecewise_cdf(boundaries, levels, (idx + 0.5) / count)
        for idx in range(count)
    ]
    rng = random.Random(seed)
    rng.shuffle(values)
    return [min(max(float(value), 1e-9), 1.0 - 1e-9) for value in values]


def load_stat_source(args: argparse.Namespace, method: str, values: Sequence[float]) -> None:
    table = f"{SCHEMA}.{SOURCE_TABLE_BY_METHOD[method]}"
    psql_copy(args, table, ["x"], ((float(value),) for value in values))
    psql_exec(args, f"ALTER TABLE {table} ALTER COLUMN x SET STATISTICS {args.pg_statistics_target}; ANALYZE {table};")


def load_all_stat_sources(
    args: argparse.Namespace,
    initial: np.ndarray,
    fresh: np.ndarray,
    method_boundaries: Dict[str, List[float]],
) -> None:
    load_stat_source(args, "stale", [float(value) for value in initial])
    load_stat_source(args, "fresh", [float(value) for value in fresh])
    for method in ["isomer", "oasis", "oasis_projected", "oasis_soft_projection", "hybrid", "aggressive_hybrid", "calibrated_hybrid"]:
        values = sample_values_from_boundaries(
            method_boundaries[method],
            count=args.stat_source_rows or len(fresh),
            seed=args.seed + METHOD_ORDER.index(method) * 7919,
        )
        load_stat_source(args, method, values)


def copy_pg_statistic_from_source(args: argparse.Namespace, method: str) -> None:
    source = f"{SCHEMA}.{SOURCE_TABLE_BY_METHOD[method]}"
    target = f"{SCHEMA}.fact"
    sql = f"""
UPDATE pg_statistic AS tgt
SET
    stanullfrac = src.stanullfrac,
    stawidth = src.stawidth,
    stadistinct = src.stadistinct,
    stakind1 = src.stakind1,
    stakind2 = src.stakind2,
    stakind3 = src.stakind3,
    stakind4 = src.stakind4,
    stakind5 = src.stakind5,
    staop1 = src.staop1,
    staop2 = src.staop2,
    staop3 = src.staop3,
    staop4 = src.staop4,
    staop5 = src.staop5,
    stacoll1 = src.stacoll1,
    stacoll2 = src.stacoll2,
    stacoll3 = src.stacoll3,
    stacoll4 = src.stacoll4,
    stacoll5 = src.stacoll5,
    stanumbers1 = src.stanumbers1,
    stanumbers2 = src.stanumbers2,
    stanumbers3 = src.stanumbers3,
    stanumbers4 = src.stanumbers4,
    stanumbers5 = src.stanumbers5,
    stavalues1 = src.stavalues1,
    stavalues2 = src.stavalues2,
    stavalues3 = src.stavalues3,
    stavalues4 = src.stavalues4,
    stavalues5 = src.stavalues5
FROM pg_statistic AS src
WHERE tgt.starelid = '{target}'::regclass
  AND tgt.staattnum = (
      SELECT attnum FROM pg_attribute
      WHERE attrelid = '{target}'::regclass AND attname = 'x'
  )
  AND tgt.stainherit = false
  AND src.starelid = '{source}'::regclass
  AND src.staattnum = (
      SELECT attnum FROM pg_attribute
      WHERE attrelid = '{source}'::regclass AND attname = 'x'
  )
  AND src.stainherit = false;
"""
    psql_exec(args, sql)


def predicate_to_sql(predicate: dict, alias: str = "f") -> str:
    pred_type = predicate["predicate_type"]
    value = float(predicate["value"])
    if pred_type in {"<", "<="}:
        return f"{alias}.x <= {value:.17g}"
    if pred_type in {">", ">="}:
        return f"{alias}.x >= {value:.17g}"
    if pred_type == "BETWEEN":
        lo, hi = sorted((float(predicate["value"]), float(predicate["value_upper"])))
        return f"{alias}.x BETWEEN {lo:.17g} AND {hi:.17g}"
    return f"{alias}.x = {value:.17g}"


def generate_predicates(fresh_boundaries: Sequence[float]) -> List[dict]:
    levels = cdf_levels(fresh_boundaries)
    predicates: List[dict] = []
    for prob in [0.006, 0.012, 0.025, 0.045, 0.075, 0.11, 0.17, 0.26, 0.38, 0.55, 0.78]:
        predicates.append({
            "predicate_id": f"le_{prob:g}",
            "predicate_type": "<=",
            "value": inverse_piecewise_cdf(fresh_boundaries, levels, prob),
            "value_upper": None,
        })
    for prob in [0.006, 0.012, 0.025, 0.045, 0.075, 0.11, 0.17, 0.26, 0.38, 0.55]:
        predicates.append({
            "predicate_id": f"ge_{prob:g}",
            "predicate_type": ">=",
            "value": inverse_piecewise_cdf(fresh_boundaries, levels, 1.0 - prob),
            "value_upper": None,
        })
    for lo_p, hi_p in [
        (0.018, 0.036),
        (0.055, 0.095),
        (0.13, 0.20),
        (0.28, 0.40),
        (0.46, 0.62),
        (0.66, 0.84),
        (0.82, 0.94),
    ]:
        predicates.append({
            "predicate_id": f"bt_{lo_p:g}_{hi_p:g}",
            "predicate_type": "BETWEEN",
            "value": inverse_piecewise_cdf(fresh_boundaries, levels, lo_p),
            "value_upper": inverse_piecewise_cdf(fresh_boundaries, levels, hi_p),
        })
    return predicates


def build_queries(predicates: Sequence[dict]) -> List[QuerySpec]:
    queries: List[QuerySpec] = []
    for predicate in predicates:
        condition = predicate_to_sql(predicate, "f")
        pred_id = str(predicate["predicate_id"])
        families = [
            (
                "scan",
                f"SELECT f.id FROM {SCHEMA}.fact AS f WHERE {condition}",
                f"SELECT count(*) FROM {SCHEMA}.fact AS f WHERE {condition}",
            ),
            (
                "join",
                f"SELECT f.id, d.payload FROM {SCHEMA}.fact AS f JOIN {SCHEMA}.dim AS d ON d.dim_id = f.dim_id WHERE {condition}",
                f"SELECT count(*) FROM {SCHEMA}.fact AS f JOIN {SCHEMA}.dim AS d ON d.dim_id = f.dim_id WHERE {condition}",
            ),
            (
                "join_dim_filter",
                f"SELECT f.id, d.payload FROM {SCHEMA}.fact AS f JOIN {SCHEMA}.dim AS d ON d.dim_id = f.dim_id WHERE {condition} AND d.payload < 15",
                f"SELECT count(*) FROM {SCHEMA}.fact AS f JOIN {SCHEMA}.dim AS d ON d.dim_id = f.dim_id WHERE {condition} AND d.payload < 15",
            ),
        ]
        for family, select_sql, count_sql in families:
            queries.append(QuerySpec(
                query_id=f"{family}:{pred_id}",
                family=family,
                predicate_id=pred_id,
                predicate_type=predicate["predicate_type"],
                value=float(predicate["value"]),
                value_upper=predicate.get("value_upper"),
                select_sql=select_sql,
                count_sql=count_sql,
            ))
    return queries


def planner_prefix(args: argparse.Namespace) -> str:
    return f"""
SET max_parallel_workers_per_gather = 0;
SET jit = off;
SET random_page_cost = {args.random_page_cost};
SET effective_cache_size = '{args.effective_cache_size}';
"""


def explain_json(args: argparse.Namespace, sql: str) -> dict:
    output = psql_scalar(args, args.dbname, planner_prefix(args) + f"EXPLAIN (FORMAT JSON) {sql}")
    return json.loads(output)[0]["Plan"]


def count_rows(args: argparse.Namespace, sql: str) -> int:
    return int(psql_scalar(args, args.dbname, sql))


def plan_nodes(plan: dict) -> List[dict]:
    nodes = [plan]
    for child in plan.get("Plans", []) or []:
        nodes.extend(plan_nodes(child))
    return nodes


def node_token(node: dict) -> str:
    parts = [str(node.get("Node Type", ""))]
    for key in ["Join Type", "Relation Name", "Index Name"]:
        if key in node:
            parts.append(f"{key}={node[key]}")
    return "[" + ";".join(parts) + "]"


def plan_signature(plan: dict) -> str:
    children = "".join(plan_signature(child) for child in plan.get("Plans", []) or [])
    return node_token(plan) + children


def plan_summary(plan: dict) -> Tuple[str, str]:
    nodes = plan_nodes(plan)
    scans = []
    joins = []
    for node in nodes:
        node_type = str(node.get("Node Type", ""))
        if "Scan" in node_type:
            suffix = ""
            if "Relation Name" in node:
                suffix += f":{node['Relation Name']}"
            if "Index Name" in node:
                suffix += f":{node['Index Name']}"
            scans.append(node_type + suffix)
        if "Join" in node_type or node_type == "Nested Loop":
            join_type = node.get("Join Type")
            joins.append(node_type if not join_type else f"{node_type}:{join_type}")
    return "|".join(scans), "|".join(joins)


def card_qerr(estimate: float, truth: float) -> float:
    estimate = max(float(estimate), 1.0)
    truth = max(float(truth), 1.0)
    return max(estimate / truth, truth / estimate)


def run_planner_capture(args: argparse.Namespace, queries: Sequence[QuerySpec]) -> List[PlanRow]:
    true_rows = {query.query_id: count_rows(args, query.count_sql) for query in queries}
    rows: List[PlanRow] = []
    for method in METHOD_ORDER:
        copy_pg_statistic_from_source(args, method)
        for query in queries:
            plan = explain_json(args, query.select_sql)
            scans, joins = plan_summary(plan)
            est_rows = float(plan.get("Plan Rows", 0.0))
            rows.append(PlanRow(
                config_id=args.config_id,
                seed=args.seed,
                row_count=args.rows,
                dim_rows=args.dim_rows,
                drift_family=args.drift_family,
                query_id=query.query_id,
                family=query.family,
                predicate_id=query.predicate_id,
                predicate_type=query.predicate_type,
                value=query.value,
                value_upper=query.value_upper,
                true_rows=true_rows[query.query_id],
                method=method,
                plan_rows=est_rows,
                row_qerr=card_qerr(est_rows, true_rows[query.query_id]),
                total_cost=float(plan.get("Total Cost", 0.0)),
                startup_cost=float(plan.get("Startup Cost", 0.0)),
                root_node=str(plan.get("Node Type", "")),
                scan_nodes=scans,
                join_nodes=joins,
                plan_signature=plan_signature(plan),
            ))
    return rows


def aggregate(rows: Sequence[PlanRow]) -> List[dict]:
    by_query: Dict[Tuple[str, str], Dict[str, PlanRow]] = defaultdict(dict)
    for row in rows:
        by_query[(row.config_id, row.query_id)][row.method] = row

    grouped: Dict[Tuple[str, str], List[PlanRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.method, "all")].append(row)
        grouped[(row.method, row.family)].append(row)

    summary = []
    for family in ["all", "scan", "join", "join_dim_filter"]:
        stale_group = grouped[("stale", family)]
        stale_qerr = geomean([row.row_qerr for row in stale_group])
        relevant_queries = [
            query_key for query_key, method_rows in by_query.items()
            if family == "all" or method_rows["stale"].family == family
        ]
        changed_queries = [
            query_key for query_key in relevant_queries
            if by_query[query_key]["stale"].plan_signature != by_query[query_key]["fresh"].plan_signature
        ]
        unchanged_queries = [
            query_key for query_key in relevant_queries
            if by_query[query_key]["stale"].plan_signature == by_query[query_key]["fresh"].plan_signature
        ]

        for method in METHOD_ORDER:
            method_rows = grouped[(method, family)]
            qerr_gm = geomean([row.row_qerr for row in method_rows])
            fresh_matches = 0
            root_matches = 0
            scan_matches = 0
            join_matches = 0
            beats_stale = 0
            recovered = 0
            new_deviations = 0
            for query_key in relevant_queries:
                row = by_query[query_key][method]
                stale = by_query[query_key]["stale"]
                fresh = by_query[query_key]["fresh"]
                fresh_matches += row.plan_signature == fresh.plan_signature
                root_matches += row.root_node == fresh.root_node
                scan_matches += row.scan_nodes == fresh.scan_nodes
                join_matches += row.join_nodes == fresh.join_nodes
                beats_stale += row.row_qerr < stale.row_qerr
                if query_key in changed_queries and row.plan_signature == fresh.plan_signature:
                    recovered += 1
                if query_key in unchanged_queries and row.plan_signature != fresh.plan_signature:
                    new_deviations += 1

            summary.append({
                "family": family,
                "method": method,
                "n": len(method_rows),
                "row_qerr_gm": qerr_gm,
                "qerr_improvement_pct": pct_improvement(stale_qerr, qerr_gm),
                "beats_stale_qerr_frac": beats_stale / max(len(relevant_queries), 1),
                "fresh_plan_match_frac": fresh_matches / max(len(relevant_queries), 1),
                "fresh_root_match_frac": root_matches / max(len(relevant_queries), 1),
                "fresh_scan_match_frac": scan_matches / max(len(relevant_queries), 1),
                "fresh_join_match_frac": join_matches / max(len(relevant_queries), 1),
                "stale_fresh_changed_queries": len(changed_queries),
                "plan_recovery_frac": recovered / max(len(changed_queries), 1),
                "new_plan_deviation_frac": new_deviations / max(len(unchanged_queries), 1),
            })
    return summary


def write_outputs(
    args: argparse.Namespace,
    rows: Sequence[PlanRow],
    summary: Sequence[dict],
    method_boundaries: Dict[str, List[float]],
    hybrid_choice: str,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "plan_rows.json", "w") as f:
        json.dump([asdict(row) for row in rows], f, indent=2)
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(list(summary), f, indent=2)
    with open(args.output_dir / "method_boundaries.json", "w") as f:
        json.dump({
            "hybrid_choice": hybrid_choice,
            "boundaries": method_boundaries,
        }, f, indent=2)

    with open(args.output_dir / "plan_rows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    with open(args.output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    write_latex_table(args.output_dir, summary)
    write_text_summary(args.output_dir, summary, hybrid_choice)


def method_label(method: str) -> str:
    return {
        "stale": "Stale",
        "isomer": "ISOMER",
        "oasis": "OASIS-noProj",
        "oasis_projected": "OASIS",
        "oasis_soft_projection": "Soft",
        "hybrid": "Hybrid",
        "aggressive_hybrid": "Aggressive",
        "calibrated_hybrid": "Router",
        "fresh": "Fresh",
    }[method]


def write_latex_table(output_dir: Path, summary: Sequence[dict]) -> None:
    rows = [row for row in summary if row["family"] == "all"]
    by_method = {row["method"]: row for row in rows}
    with open(output_dir / "table_postgres_planner_stats_injection.tex", "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("  \\centering\n")
        f.write("  \\small\n")
        f.write("  \\caption{PostgreSQL planner-only statistics-injection evidence. True rows are measured with \\texttt{COUNT(*)}; estimated rows and plan shapes are read from \\texttt{EXPLAIN (FORMAT JSON)}. No query runtime is measured.}\n")
        f.write("  \\label{tab:postgres_planner_stats_injection}\n")
        f.write("  \\setlength{\\tabcolsep}{4pt}\n")
        f.write("  \\begin{tabular}{lrrrrr}\n")
        f.write("    \\toprule\n")
        f.write("    Method & Row QE & QE Imp. & Fresh Plan & Recovery & New Deviations \\\\\n")
        f.write("    \\midrule\n")
        for method in METHOD_ORDER:
            row = by_method[method]
            f.write(
                f"    {method_label(method)} & {row['row_qerr_gm']:.3f} & "
                f"{row['qerr_improvement_pct']:.1f}\\% & "
                f"{row['fresh_plan_match_frac'] * 100:.1f}\\% & "
                f"{row['plan_recovery_frac'] * 100:.1f}\\% & "
                f"{row['new_plan_deviation_frac'] * 100:.1f}\\% \\\\\n"
            )
        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n")


def write_text_summary(output_dir: Path, summary: Sequence[dict], hybrid_choice: str) -> None:
    rows = [row for row in summary if row["family"] == "all"]
    by_method = {row["method"]: row for row in rows}
    stale = by_method["stale"]
    lines = [
        "PostgreSQL planner-only stats injection",
        "=" * 48,
        "No wall-clock runtime is measured. True rows come from COUNT(*);",
        "estimated rows and plan shapes come from EXPLAIN (FORMAT JSON).",
        "",
        f"Queries per method: {stale['n']}",
        f"Stale/fresh plan-shape disagreements: {stale['stale_fresh_changed_queries']}",
        f"Hybrid selected: {hybrid_choice}",
        f"Config: {getattr(output_dir, 'name', '')}",
        "",
        "Method          RowQE  QEImp  BeatStale  FreshPlan  Recovery  NewDev",
        "-" * 78,
    ]
    for method in METHOD_ORDER:
        row = by_method[method]
        lines.append(
            f"{method:<15s} {row['row_qerr_gm']:5.3f}  "
            f"{row['qerr_improvement_pct']:5.1f}%  "
            f"{row['beats_stale_qerr_frac'] * 100:8.1f}%  "
            f"{row['fresh_plan_match_frac'] * 100:9.1f}%  "
            f"{row['plan_recovery_frac'] * 100:8.1f}%  "
            f"{row['new_plan_deviation_frac'] * 100:6.1f}%"
        )
    lines.extend(["", "By family:"])
    for family in ["scan", "join", "join_dim_filter"]:
        lines.append(f"  {family}:")
        for method in METHOD_ORDER:
            row = next(item for item in summary if item["family"] == family and item["method"] == method)
            lines.append(
                f"    {method:<15s} RowQE={row['row_qerr_gm']:.3f} "
                f"FreshPlan={row['fresh_plan_match_frac'] * 100:.1f}% "
                f"Recovery={row['plan_recovery_frac'] * 100:.1f}%"
            )

    text = "\n".join(lines)
    (output_dir / "summary.txt").write_text(text + "\n")
    print(text)


def run_experiment(args: argparse.Namespace) -> Tuple[List[PlanRow], List[dict], Dict[str, List[float]], str]:
    ensure_postgres(args)
    setup_schema(args)

    if not getattr(args, "config_id", ""):
        args.config_id = f"{args.drift_family}_rows{args.rows}_seed{args.seed}"

    initial, fresh, dim_ids = generate_data(args.rows, args.dim_rows, args.seed, drift_family=args.drift_family)
    sample = make_feedback_sample(initial, fresh, args)
    model = MlpHistogramModelV2.load(str(args.model_path))
    method_boundaries, hybrid_choice = build_method_boundaries(
        sample,
        model=model,
        num_buckets=args.num_buckets,
        max_observations=args.max_observations,
        aggressive_damping_grid=args.aggressive_damping_grid,
        aggressive_recent_windows=args.aggressive_recent_windows,
        soft_projection_strength=args.soft_projection_strength,
        soft_projection_recency_decay=args.soft_projection_recency_decay,
        soft_projection_target_blend=args.soft_projection_target_blend,
        soft_projection_window=args.soft_projection_window,
        soft_projection_iters=args.soft_projection_iters,
        soft_projection_lr=args.soft_projection_lr,
        soft_projection_tol=args.soft_projection_tol,
        soft_projection_active_set=args.soft_projection_active_set,
        soft_projection_conflict_aware=args.soft_projection_conflict_aware,
        soft_projection_conflict_ref_window=args.soft_projection_conflict_ref_window,
        soft_projection_conflict_tau=args.soft_projection_conflict_tau,
        soft_projection_conflict_floor=args.soft_projection_conflict_floor,
        projection_iters=args.projection_iters,
        projection_tol=args.projection_tol,
    )

    load_base_tables(args, fresh, dim_ids)
    load_all_stat_sources(args, initial, fresh, method_boundaries)

    predicates = generate_predicates(method_boundaries["fresh"])
    queries = build_queries(predicates)
    rows = run_planner_capture(args, queries)
    summary = aggregate(rows)
    write_outputs(args, rows, summary, method_boundaries, hybrid_choice)
    return rows, summary, method_boundaries, hybrid_choice


def clone_args(args: argparse.Namespace, **updates) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def batch_config_id(drift_family: str, rows: int, seed: int) -> str:
    return f"{drift_family}_rows{rows}_seed{seed}"


def config_summary_rows(config_id: str, summary: Sequence[dict]) -> List[dict]:
    rows = []
    drift_family, row_count, seed = parse_config_id(config_id)
    for row in summary:
        item = {
            "config_id": config_id,
            "drift_family": drift_family,
            "row_count": row_count,
            "seed": seed,
        }
        item.update(row)
        rows.append(item)
    return rows


def parse_config_id(config_id: str) -> Tuple[str, int, int]:
    parts = config_id.rsplit("_rows", 1)
    if len(parts) != 2:
        return config_id, 0, 0
    drift_family = parts[0]
    row_part, seed_part = parts[1].split("_seed", 1)
    return drift_family, int(row_part), int(seed_part)


def mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def sample_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def aggregate_config_means(config_rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in config_rows:
        grouped[(row["family"], row["method"])].append(row)

    result = []
    for family in ["all", "scan", "join", "join_dim_filter"]:
        for method in METHOD_ORDER:
            rows = grouped[(family, method)]
            result.append({
                "family": family,
                "method": method,
                "n_configs": len(rows),
                "row_qerr_gm_mean": mean([float(row["row_qerr_gm"]) for row in rows]),
                "row_qerr_gm_std": sample_std([float(row["row_qerr_gm"]) for row in rows]),
                "qerr_improvement_pct_mean": mean([float(row["qerr_improvement_pct"]) for row in rows]),
                "qerr_improvement_pct_std": sample_std([float(row["qerr_improvement_pct"]) for row in rows]),
                "fresh_plan_match_frac_mean": mean([float(row["fresh_plan_match_frac"]) for row in rows]),
                "fresh_plan_match_frac_std": sample_std([float(row["fresh_plan_match_frac"]) for row in rows]),
                "plan_recovery_frac_mean": mean([float(row["plan_recovery_frac"]) for row in rows]),
                "plan_recovery_frac_std": sample_std([float(row["plan_recovery_frac"]) for row in rows]),
                "new_plan_deviation_frac_mean": mean([float(row["new_plan_deviation_frac"]) for row in rows]),
                "new_plan_deviation_frac_std": sample_std([float(row["new_plan_deviation_frac"]) for row in rows]),
                "stale_fresh_changed_queries_mean": mean([float(row["stale_fresh_changed_queries"]) for row in rows]),
            })
    return result


def write_batch_outputs(
    output_dir: Path,
    all_rows: Sequence[PlanRow],
    aggregate_summary: Sequence[dict],
    config_rows: Sequence[dict],
    config_means: Sequence[dict],
    hybrid_choices: Dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "batch_plan_rows.json", "w") as f:
        json.dump([asdict(row) for row in all_rows], f, indent=2)
    with open(output_dir / "batch_summary.json", "w") as f:
        json.dump(list(aggregate_summary), f, indent=2)
    with open(output_dir / "batch_config_summary.json", "w") as f:
        json.dump(list(config_rows), f, indent=2)
    with open(output_dir / "batch_config_means.json", "w") as f:
        json.dump(list(config_means), f, indent=2)
    with open(output_dir / "batch_hybrid_choices.json", "w") as f:
        json.dump(hybrid_choices, f, indent=2)

    with open(output_dir / "batch_plan_rows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(asdict(row))
    with open(output_dir / "batch_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_summary[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_summary)
    with open(output_dir / "batch_config_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(config_rows[0].keys()))
        writer.writeheader()
        writer.writerows(config_rows)
    with open(output_dir / "batch_config_means.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(config_means[0].keys()))
        writer.writeheader()
        writer.writerows(config_means)

    write_batch_latex_table(output_dir, config_means)
    write_batch_text_summary(output_dir, aggregate_summary, config_means, hybrid_choices)


def write_batch_latex_table(output_dir: Path, config_means: Sequence[dict]) -> None:
    rows = [row for row in config_means if row["family"] == "all"]
    by_method = {row["method"]: row for row in rows}
    with open(output_dir / "table_postgres_planner_stats_injection_batch.tex", "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("  \\centering\n")
        f.write("  \\small\n")
        f.write("  \\caption{Multi-configuration PostgreSQL planner-only statistics-injection evidence. Values are means over configurations; parentheses show one standard deviation across configurations. True rows use \\texttt{COUNT(*)}; estimated rows and plan shapes use \\texttt{EXPLAIN (FORMAT JSON)}. No query runtime is measured.}\n")
        f.write("  \\label{tab:postgres_planner_stats_injection_batch}\n")
        f.write("  \\setlength{\\tabcolsep}{3pt}\n")
        f.write("  \\resizebox{\\textwidth}{!}{%\n")
        f.write("  \\begin{tabular}{lrrrr}\n")
        f.write("    \\toprule\n")
        f.write("    Method & Row QE & Fresh Plan & Recovery & New Deviations \\\\\n")
        f.write("    \\midrule\n")
        for method in METHOD_ORDER:
            row = by_method[method]
            f.write(
                f"    {method_label(method)} & "
                f"{row['row_qerr_gm_mean']:.3f} ({row['row_qerr_gm_std']:.3f}) & "
                f"{row['fresh_plan_match_frac_mean'] * 100:.1f}\\% ({row['fresh_plan_match_frac_std'] * 100:.1f}) & "
                f"{row['plan_recovery_frac_mean'] * 100:.1f}\\% ({row['plan_recovery_frac_std'] * 100:.1f}) & "
                f"{row['new_plan_deviation_frac_mean'] * 100:.1f}\\% ({row['new_plan_deviation_frac_std'] * 100:.1f}) \\\\\n"
            )
        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("  }\n")
        f.write("\\end{table}\n")


def write_batch_text_summary(
    output_dir: Path,
    aggregate_summary: Sequence[dict],
    config_means: Sequence[dict],
    hybrid_choices: Dict[str, str],
) -> None:
    aggregate_all = {row["method"]: row for row in aggregate_summary if row["family"] == "all"}
    mean_all = {row["method"]: row for row in config_means if row["family"] == "all"}
    lines = [
        "Multi-configuration PostgreSQL planner-only stats injection",
        "=" * 64,
        "No wall-clock runtime is measured. True rows come from COUNT(*);",
        "estimated rows and plan shapes come from EXPLAIN (FORMAT JSON).",
        "",
        f"Configurations: {len(hybrid_choices)}",
        f"Queries per configuration per method: {aggregate_all['stale']['n'] // max(len(hybrid_choices), 1)}",
        "",
        "Aggregate over all query instances:",
        "Method          RowQE  QEImp  BeatStale  FreshPlan  Recovery  NewDev",
        "-" * 78,
    ]
    for method in METHOD_ORDER:
        row = aggregate_all[method]
        lines.append(
            f"{method:<15s} {row['row_qerr_gm']:5.3f}  "
            f"{row['qerr_improvement_pct']:5.1f}%  "
            f"{row['beats_stale_qerr_frac'] * 100:8.1f}%  "
            f"{row['fresh_plan_match_frac'] * 100:9.1f}%  "
            f"{row['plan_recovery_frac'] * 100:8.1f}%  "
            f"{row['new_plan_deviation_frac'] * 100:6.1f}%"
        )

    lines.extend(["", "Mean across configurations:", "Method          RowQE(mean±sd)  FreshPlan  Recovery  NewDev", "-" * 72])
    for method in METHOD_ORDER:
        row = mean_all[method]
        lines.append(
            f"{method:<15s} {row['row_qerr_gm_mean']:5.3f}±{row['row_qerr_gm_std']:.3f}  "
            f"{row['fresh_plan_match_frac_mean'] * 100:8.1f}%  "
            f"{row['plan_recovery_frac_mean'] * 100:8.1f}%  "
            f"{row['new_plan_deviation_frac_mean'] * 100:6.1f}%"
        )

    lines.extend(["", "Hybrid choices:"])
    for config_id, choice in sorted(hybrid_choices.items()):
        lines.append(f"  {config_id}: {choice}")

    text = "\n".join(lines)
    (output_dir / "summary.txt").write_text(text + "\n")
    print(text)


def run_batch(args: argparse.Namespace) -> None:
    ensure_postgres(args)
    all_rows: List[PlanRow] = []
    config_rows: List[dict] = []
    hybrid_choices: Dict[str, str] = {}

    total = len(args.batch_drift_families) * len(args.batch_rows) * len(args.batch_seeds)
    index = 0
    for drift_family in args.batch_drift_families:
        for row_count in args.batch_rows:
            for seed in args.batch_seeds:
                index += 1
                config_id = batch_config_id(drift_family, row_count, seed)
                print(f"\n[{index}/{total}] Running {config_id}")
                config_args = clone_args(
                    args,
                    batch=False,
                    drift_family=drift_family,
                    rows=row_count,
                    dim_rows=max(args.min_dim_rows, int(row_count * args.dim_rows_ratio)),
                    seed=seed,
                    config_id=config_id,
                    output_dir=args.output_dir / "configs" / config_id,
                )
                rows, summary, _, hybrid_choice = run_experiment(config_args)
                all_rows.extend(rows)
                config_rows.extend(config_summary_rows(config_id, summary))
                hybrid_choices[config_id] = hybrid_choice

    aggregate_summary = aggregate(all_rows)
    config_means = aggregate_config_means(config_rows)
    write_batch_outputs(args.output_dir, all_rows, aggregate_summary, config_rows, config_means, hybrid_choices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PostgreSQL planner-only stats-injection evidence experiment")
    parser.add_argument("--pg-bin", type=Path, default=Path("/Volumes/QUQ/pg/pgsql/bin"))
    parser.add_argument("--data-dir", type=Path, default=Path("/Volumes/QUQ/pg/data"))
    parser.add_argument("--socket-dir", type=Path, default=Path("/Volumes/QUQ/pg/run"))
    parser.add_argument("--log-dir", type=Path, default=Path("/Volumes/QUQ/pg/logs"))
    parser.add_argument("--pg-port", type=int, default=55432)
    parser.add_argument("--pg-user", type=str, default="postgres")
    parser.add_argument("--dbname", type=str, default="oasis_e2e")
    parser.add_argument("--output-dir", type=Path,
                        default=_REPO_DIR / "experiments" / "results" / "postgres_planner_stats_injection_20260529")
    parser.add_argument("--model-path", type=Path,
                        default=_REPO_DIR / "experiments" / "results" / "synthetic_paper_suite_rerun_20260529" / "models" / "oasis_k16.json")
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--dim-rows", type=int, default=12_000)
    parser.add_argument("--drift-family", type=str, default="left_shift",
                        choices=["left_shift", "right_shift", "bimodal_shift"])
    parser.add_argument("--config-id", type=str, default="")
    parser.add_argument("--stat-source-rows", type=int, default=0,
                        help="Rows for learned-stat source tables; 0 reuses --rows.")
    parser.add_argument("--num-buckets", type=int, default=10)
    parser.add_argument("--max-observations", type=int, default=16)
    parser.add_argument("--num-feedback", type=int, default=0,
                        help="Exp 2 sparse-feedback sweep: keep this many evenly spaced feedback predicates "
                             "(0 or >=16 keeps all 16). Reduces the projection's feedback constraints, not just the MLP window.")
    parser.add_argument("--aggressive-damping-grid", type=float, nargs="+",
                        default=[0.35, 0.50, 0.65, 0.80, 0.95])
    parser.add_argument("--aggressive-recent-windows", type=int, nargs="+",
                        default=[4, 8, 12])
    parser.add_argument("--soft-projection-strength", type=float, default=30.0)
    parser.add_argument("--soft-projection-recency-decay", type=float, default=0.80)
    parser.add_argument("--soft-projection-target-blend", type=float, default=1.0)
    parser.add_argument("--soft-projection-window", type=int, default=0,
                        help="Use only the most recent N observations for soft projection; 0 uses the full window.")
    parser.add_argument("--soft-projection-iters", type=int, default=500)
    parser.add_argument("--soft-projection-lr", type=float, default=0.05)
    parser.add_argument("--soft-projection-tol", type=float, default=1e-9)
    parser.add_argument("--soft-projection-active-set", action="store_true",
                        help="Apply soft projection only to the latest hard-feasible feedback suffix.")
    parser.add_argument("--soft-projection-conflict-aware", action="store_true",
                        help="Down-weight feedback constraints contradicted by the most recent observations.")
    parser.add_argument("--soft-projection-conflict-ref-window", type=int, default=8,
                        help="Number of most-recent observations treated as the trusted conflict reference.")
    parser.add_argument("--soft-projection-conflict-tau", type=float, default=0.05,
                        help="Conflict bandwidth; smaller tau suppresses contradicted observations more aggressively.")
    parser.add_argument("--soft-projection-conflict-floor", type=float, default=0.0,
                        help="Minimum residual weight for a contradicted observation (0 fully removes it).")
    parser.add_argument("--projection-iters", type=int, default=200)
    parser.add_argument("--projection-tol", type=float, default=1e-4)
    parser.add_argument("--pg-statistics-target", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--random-page-cost", type=float, default=1.5)
    parser.add_argument("--effective-cache-size", type=str, default="1GB")
    parser.add_argument("--batch", action="store_true",
                        help="Run a multi-configuration batch; --output-dir is used as the batch root.")
    parser.add_argument("--batch-seeds", type=int, nargs="+",
                        default=[20260529, 20260530, 20260531])
    parser.add_argument("--batch-rows", type=int, nargs="+",
                        default=[100_000, 200_000])
    parser.add_argument("--batch-drift-families", type=str, nargs="+",
                        default=["left_shift", "bimodal_shift"],
                        choices=["left_shift", "right_shift", "bimodal_shift"])
    parser.add_argument("--dim-rows-ratio", type=float, default=0.06,
                        help="Batch mode dimension rows as a fraction of fact rows.")
    parser.add_argument("--min-dim-rows", type=int, default=5_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch:
        run_batch(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
