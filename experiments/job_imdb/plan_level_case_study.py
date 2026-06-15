#!/usr/bin/env python3
"""Targeted JOB plan-level scanner for OASIS cross-column repair.

This script looks for canonical JOB queries whose predicates induce a local
conjunction on title(production_year, kind_id), computes PostgreSQL's local
estimate, the true local count, and the OASIS/IPF repaired estimate, then saves
the surrounding full-query EXPLAIN JSON for plan inspection.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import crosscol_pg_inject_experiment as xcol


PSQL: list[str] = []


def q(sql: str) -> list[list[str]]:
    out = subprocess.run(PSQL + ["-tAF", "\t", "-c", sql], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:500] + " | SQL: " + sql[:240])
    return [line.split("\t") for line in out.stdout.splitlines() if line != ""]


def scalar(sql: str) -> str | None:
    rows = q(sql)
    return rows[0][0] if rows and rows[0] else None


def sql_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


@dataclass
class TitleKindModel:
    years: list[float]
    kinds: list[float]
    rep: np.ndarray
    avi: np.ndarray
    n: int
    year_idx: dict[float, int]
    kind_idx: dict[float, int]


def build_title_kind_model(k: int, seed: int) -> TitleKindModel:
    xcol.PSQL = PSQL
    sa = xcol.col_spec("title", "production_year")
    sb = xcol.col_spec("title", "kind_id")
    p = xcol.build_joint("title", "production_year", "kind_id", sa, sb)
    pn = p / p.sum()
    avi = (p.sum(1, keepdims=True) @ p.sum(0, keepdims=True)) / (p.sum() ** 2)
    rng = random.Random(seed)
    fb = xcol.rects(xcol.nbins(sa), xcol.nbins(sb), k, rng)
    ft = [pn[i0:i1, j0:j1].sum() for (i0, i1, j0, j1) in fb]
    rep = xcol.ipf2d(avi, fb, ft)
    years = list(sa[1])
    kinds = list(sb[1])
    return TitleKindModel(
        years=years,
        kinds=kinds,
        rep=rep,
        avi=avi,
        n=int(scalar("SELECT count(*) FROM title WHERE production_year IS NOT NULL AND kind_id IS NOT NULL")),
        year_idx={v: i for i, v in enumerate(years)},
        kind_idx={v: i for i, v in enumerate(kinds)},
    )


def strip_sql(sql: str) -> str:
    sql = re.sub(r"--.*", "", sql)
    return re.sub(r"\s+", " ", sql).strip().rstrip(";")


def aliases(sql: str) -> dict[str, str]:
    return {alias: table for table, alias in re.findall(r"\b([a-z_]+)\s+AS\s+([a-z][a-z0-9_]*)\b", sql, re.I)}


def extract_numeric_condition(sql: str, alias: str, col: str) -> tuple[str, set[float]] | None:
    colref = rf"{re.escape(alias)}\.{re.escape(col)}"
    m = re.search(rf"\b{colref}\s+BETWEEN\s+(-?\d+)\s+AND\s+(-?\d+)\b", sql, re.I)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return f"{alias}.{col} BETWEEN {lo} AND {hi}", {float(v) for v in range(lo, hi + 1)}
    m = re.search(rf"\b{colref}\s*(=|>=|>|<=|<)\s*(-?\d+)\b", sql, re.I)
    if not m:
        return None
    op, raw = m.group(1), int(m.group(2))
    if op == "=":
        vals = {float(raw)}
    elif op == ">":
        vals = {v for v in MODEL.years if v > raw}
    elif op == ">=":
        vals = {v for v in MODEL.years if v >= raw}
    elif op == "<":
        vals = {v for v in MODEL.years if v < raw}
    else:
        vals = {v for v in MODEL.years if v <= raw}
    return f"{alias}.{col} {op} {raw}", vals


def extract_kind_condition(sql: str, kt_alias: str) -> tuple[str, set[float]] | None:
    colref = rf"{re.escape(kt_alias)}\.kind"
    m = re.search(rf"\b{colref}\s+IN\s*\(([^)]*)\)", sql, re.I)
    if m:
        vals = re.findall(r"'((?:[^']|'')*)'", m.group(1))
        if not vals:
            return None
        cond = f"{kt_alias}.kind IN (" + ",".join(sql_quote(v.replace("''", "'")) for v in vals) + ")"
        rows = q("SELECT id::double precision FROM kind_type WHERE kind IN (" + ",".join(sql_quote(v.replace("''", "'")) for v in vals) + ")")
        return cond, {float(r[0]) for r in rows}
    m = re.search(rf"\b{colref}\s*=\s*'((?:[^']|'')*)'", sql, re.I)
    if not m:
        return None
    val = m.group(1).replace("''", "'")
    rows = q(f"SELECT id::double precision FROM kind_type WHERE kind = {sql_quote(val)}")
    return f"{kt_alias}.kind = {sql_quote(val)}", {float(r[0]) for r in rows}


def plan_json(sql: str) -> dict:
    out = subprocess.run(PSQL + ["-tA", "-c", "EXPLAIN (FORMAT JSON) " + sql], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:500])
    return json.loads(out.stdout.strip())[0]["Plan"]


def explain_rows(sql: str) -> int:
    return int(plan_json(sql)["Plan Rows"])


def count_rows(sql: str) -> int:
    return int(scalar("SELECT count(*) FROM (" + sql + ") AS q"))


def qerr(est: float, true: float) -> float:
    est, true = max(est, 1e-9), max(true, 1e-9)
    return max(est / true, true / est)


def leaf_aliases(node: dict) -> set[str]:
    if "Plans" not in node:
        return {node["Alias"]} if "Alias" in node else set()
    out: set[str] = set()
    for child in node.get("Plans", []):
        out |= leaf_aliases(child)
    return out


def collect_nodes(node: dict, out: list[dict]) -> set[str]:
    leaves = leaf_aliases(node)
    row = {
        "node_type": node.get("Node Type"),
        "join_type": node.get("Join Type"),
        "plan_rows": node.get("Plan Rows"),
        "total_cost": node.get("Total Cost"),
        "relation": node.get("Relation Name"),
        "alias": node.get("Alias"),
        "leaf_aliases": sorted(leaves),
    }
    out.append(row)
    for child in node.get("Plans", []):
        collect_nodes(child, out)
    return leaves


def summarize_plan(node: dict) -> dict:
    nodes: list[dict] = []
    collect_nodes(node, nodes)
    joins = [n for n in nodes if n["node_type"] and "Join" in n["node_type"]]
    scans = [n for n in nodes if n["relation"]]
    return {
        "root": {"node_type": node.get("Node Type"), "plan_rows": node.get("Plan Rows"), "total_cost": node.get("Total Cost")},
        "joins": joins,
        "scans": scans,
    }


def smallest_branch(summary: dict, wanted: set[str]) -> dict | None:
    candidates = [n for n in summary["joins"] + summary["scans"] if wanted <= set(n["leaf_aliases"])]
    if not candidates:
        return None
    return min(candidates, key=lambda n: (len(n["leaf_aliases"]), n.get("total_cost") or 0))


def oasis_estimate(model: TitleKindModel, years: set[float], kinds: set[float]) -> tuple[float, float]:
    yi = [model.year_idx[v] for v in years if v in model.year_idx]
    ki = [model.kind_idx[v] for v in kinds if v in model.kind_idx]
    if not yi or not ki:
        return 0.0, 0.0
    return float(model.rep[np.ix_(yi, ki)].sum() * model.n), float(model.avi[np.ix_(yi, ki)].sum() * model.n)


MODEL: TitleKindModel


def scan_query(path: Path, model: TitleKindModel) -> list[dict]:
    global MODEL
    MODEL = model
    raw = path.read_text()
    sql = strip_sql(raw)
    als = aliases(sql)
    title_aliases = [a for a, t in als.items() if t.lower() == "title"]
    kt_aliases = [a for a, t in als.items() if t.lower() == "kind_type"]
    if not title_aliases or not kt_aliases:
        return []

    full_plan = plan_json(sql)
    summary = summarize_plan(full_plan)
    out: list[dict] = []
    for ta in title_aliases:
        year = extract_numeric_condition(sql, ta, "production_year")
        if not year:
            continue
        for ka in kt_aliases:
            join_patterns = [
                rf"\b{re.escape(ka)}\.id\s*=\s*{re.escape(ta)}\.kind_id\b",
                rf"\b{re.escape(ta)}\.kind_id\s*=\s*{re.escape(ka)}\.id\b",
            ]
            if not any(re.search(p, sql, re.I) for p in join_patterns):
                continue
            kind = extract_kind_condition(sql, ka)
            if not kind:
                continue
            local_sql = (
                f"SELECT 1 FROM title AS {ta}, kind_type AS {ka} "
                f"WHERE {year[0]} AND {kind[0]} AND {ka}.id = {ta}.kind_id"
            )
            true_rows = count_rows(local_sql)
            pg_rows = explain_rows(local_sql)
            oasis_rows, avi_rows = oasis_estimate(model, year[1], kind[1])
            branch = smallest_branch(summary, {ta, ka})
            out.append(
                {
                    "query": path.name,
                    "title_alias": ta,
                    "kind_alias": ka,
                    "year_condition": year[0],
                    "kind_condition": kind[0],
                    "local_sql": local_sql,
                    "true_rows": true_rows,
                    "pg_local_rows": pg_rows,
                    "avi_true_rows": avi_rows,
                    "oasis_rows": oasis_rows,
                    "pg_local_qerr": qerr(pg_rows, true_rows),
                    "avi_true_qerr": qerr(avi_rows, true_rows),
                    "oasis_qerr": qerr(oasis_rows, true_rows),
                    "affected_branch": branch,
                    "plan_summary": summary,
                    "full_query_sql": sql,
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="/home/tianqc/job_pg/queries")
    ap.add_argument("--out", default="/home/tianqc/job_pg/results_plan_level/title_kind_candidates.json")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--psql", default=os.environ.get("PSQL_BIN", "psql"))
    ap.add_argument("--host", default=os.environ.get("PGHOST", ""))
    ap.add_argument("--port", default=os.environ.get("PGPORT", ""))
    ap.add_argument("--db", default=os.environ.get("PGDATABASE", "imdb"))
    ap.add_argument("--user", default=os.environ.get("PGUSER", ""))
    args = ap.parse_args()

    global PSQL
    PSQL = [args.psql, "-d", args.db]
    if args.host:
        PSQL += ["-h", args.host]
    if args.port:
        PSQL += ["-p", str(args.port)]
    if args.user:
        PSQL += ["-U", args.user]

    model = build_title_kind_model(args.k, args.seed)
    results: list[dict] = []
    for path in sorted(Path(args.queries).glob("*.sql")):
        try:
            results.extend(scan_query(path, model))
        except Exception as exc:
            results.append({"query": path.name, "error": str(exc)})

    results.sort(key=lambda r: r.get("pg_local_qerr", 0), reverse=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out} with {len(results)} rows")
    print(f"{'query':7} {'aliases':9} {'true':>9} {'PG':>9} {'AVI*':>9} {'OASIS':>9} {'PGq':>7} {'Oq':>7} branch")
    for r in results[:20]:
        if "error" in r:
            print(f"{r['query']:7} ERROR {r['error'][:120]}")
            continue
        b = r.get("affected_branch") or {}
        aliases_s = f"{r['title_alias']},{r['kind_alias']}"
        print(
            f"{r['query']:7} {aliases_s:9} {r['true_rows']:9.0f} {r['pg_local_rows']:9.0f} "
            f"{r['avi_true_rows']:9.0f} {r['oasis_rows']:9.0f} {r['pg_local_qerr']:7.2f} "
            f"{r['oasis_qerr']:7.2f} {b.get('node_type')} {b.get('leaf_aliases')}"
        )


if __name__ == "__main__":
    main()
