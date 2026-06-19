#!/usr/bin/env python3
"""Scan JOB variants for additional OASIS plan-level Rows-hint candidates.

The scanner emits candidate JSON compatible with plan_level_hint_eval.py.  It
keeps the previous title.production_year x title.kind_id path and adds a
movie_companies.company_type_id x company_id path through company_type/company_name.
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


def strip_sql(sql: str) -> str:
    sql = re.sub(r"--.*", "", sql)
    return re.sub(r"\s+", " ", sql).strip().rstrip(";")


def aliases(sql: str) -> dict[str, str]:
    return {alias: table for table, alias in re.findall(r"\b([a-z_]+)\s+AS\s+([a-z][a-z0-9_]*)\b", sql, re.I)}


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
    out.append(
        {
            "node_type": node.get("Node Type"),
            "join_type": node.get("Join Type"),
            "plan_rows": node.get("Plan Rows"),
            "total_cost": node.get("Total Cost"),
            "relation": node.get("Relation Name"),
            "alias": node.get("Alias"),
            "leaf_aliases": sorted(leaves),
        }
    )
    for child in node.get("Plans", []):
        collect_nodes(child, out)
    return leaves


def summarize_plan(node: dict) -> dict:
    nodes: list[dict] = []
    collect_nodes(node, nodes)
    return {
        "root": {"node_type": node.get("Node Type"), "plan_rows": node.get("Plan Rows"), "total_cost": node.get("Total Cost")},
        "joins": [n for n in nodes if n["node_type"] and "Join" in n["node_type"]],
        "scans": [n for n in nodes if n["relation"]],
    }


def smallest_branch(summary: dict, wanted: set[str]) -> dict | None:
    candidates = [n for n in summary["joins"] + summary["scans"] if wanted <= set(n["leaf_aliases"])]
    if not candidates:
        return None
    return min(candidates, key=lambda n: (len(n["leaf_aliases"]), n.get("total_cost") or 0))


@dataclass
class PairModel:
    table: str
    col_a: str
    col_b: str
    spec_a: tuple[str, list[float]]
    spec_b: tuple[str, list[float]]
    rep: np.ndarray
    avi: np.ndarray
    n: int
    idx_a: dict[float, int] | None
    idx_b: dict[float, int] | None


def build_pair_model(table: str, col_a: str, col_b: str, k: int, seed: int) -> PairModel:
    xcol.PSQL = PSQL
    sa = xcol.col_spec(table, col_a)
    sb = xcol.col_spec(table, col_b)
    p = xcol.build_joint(table, col_a, col_b, sa, sb)
    pn = p / p.sum()
    avi = (p.sum(1, keepdims=True) @ p.sum(0, keepdims=True)) / (p.sum() ** 2)
    rng = random.Random(seed)
    fb = xcol.rects(xcol.nbins(sa), xcol.nbins(sb), k, rng)
    ft = [pn[i0:i1, j0:j1].sum() for (i0, i1, j0, j1) in fb]
    rep = xcol.ipf2d(avi, fb, ft)
    n = int(scalar(f"SELECT count(*) FROM {table} WHERE {col_a} IS NOT NULL AND {col_b} IS NOT NULL"))
    return PairModel(
        table=table,
        col_a=col_a,
        col_b=col_b,
        spec_a=sa,
        spec_b=sb,
        rep=rep,
        avi=avi,
        n=n,
        idx_a={v: i for i, v in enumerate(sa[1])} if sa[0] == "cat" else None,
        idx_b={v: i for i, v in enumerate(sb[1])} if sb[0] == "cat" else None,
    )


def num_bin_indices(spec: tuple[str, list[float]], values: set[float]) -> list[int]:
    edges = spec[1]
    out: set[int] = set()
    for v in values:
        if v < edges[0] or v > edges[-1]:
            continue
        idx = np.searchsorted(edges[1:-1], v, side="right")
        out.add(min(max(int(idx), 0), len(edges) - 2))
    return sorted(out)


def estimate_rect(model: PairModel, a_idx: list[int], b_idx: list[int]) -> tuple[float, float]:
    if not a_idx or not b_idx:
        return 0.0, 0.0
    rep = float(model.rep[np.ix_(a_idx, b_idx)].sum() * model.n)
    avi = float(model.avi[np.ix_(a_idx, b_idx)].sum() * model.n)
    return rep, avi


def extract_numeric_condition(sql: str, alias: str, col: str, allowed: list[float]) -> tuple[str, set[float]] | None:
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
        vals = {v for v in allowed if v > raw}
    elif op == ">=":
        vals = {v for v in allowed if v >= raw}
    elif op == "<":
        vals = {v for v in allowed if v < raw}
    else:
        vals = {v for v in allowed if v <= raw}
    return f"{alias}.{col} {op} {raw}", vals


def extract_text_dimension_ids(sql: str, alias: str, table: str, text_col: str, id_col: str = "id") -> tuple[str, set[float]] | None:
    colref = rf"{re.escape(alias)}\.{re.escape(text_col)}"
    m = re.search(rf"\b{colref}\s+IN\s*\(([^)]*)\)", sql, re.I)
    if m:
        vals = [v.replace("''", "'") for v in re.findall(r"'((?:[^']|'')*)'", m.group(1))]
        if not vals:
            return None
        cond = f"{alias}.{text_col} IN (" + ",".join(sql_quote(v) for v in vals) + ")"
        rows = q(f"SELECT {id_col}::double precision FROM {table} AS {alias} WHERE {cond}")
        return cond, {float(r[0]) for r in rows}
    m = re.search(rf"\b{colref}\s*=\s*'((?:[^']|'')*)'", sql, re.I)
    if m:
        val = m.group(1).replace("''", "'")
        cond = f"{alias}.{text_col} = {sql_quote(val)}"
        rows = q(f"SELECT {id_col}::double precision FROM {table} AS {alias} WHERE {cond}")
        return cond, {float(r[0]) for r in rows}
    return None


def extract_alias_filters(sql: str, alias: str) -> str:
    where = re.split(r"\bWHERE\b", sql, flags=re.I, maxsplit=1)
    if len(where) != 2:
        return ""
    body = where[1]
    filters: list[str] = []
    simple = rf"{re.escape(alias)}\.[a-z_]+\s*(?:=|!=|<>|LIKE|NOT\s+LIKE)\s*'[^']*'"
    filters.extend(m.group(0) for m in re.finditer(simple, body, re.I))
    in_pat = rf"{re.escape(alias)}\.[a-z_]+\s+IN\s*\([^)]*\)"
    filters.extend(m.group(0) for m in re.finditer(in_pat, body, re.I))
    paren_pat = rf"\([^)]*{re.escape(alias)}\.[^)]*\)"
    filters.extend(m.group(0) for m in re.finditer(paren_pat, body, re.I) if re.search(r"\bOR\b|\bLIKE\b", m.group(0), re.I))
    cleaned: list[str] = []
    for f in filters:
        if re.search(rf"\b{re.escape(alias)}\.id\s*=", f, re.I):
            continue
        if f not in cleaned:
            cleaned.append(f)
    return " AND ".join(cleaned)


def scan_title_kind(path: Path, sql: str, als: dict[str, str], summary: dict, model: PairModel) -> list[dict]:
    out: list[dict] = []
    allowed_years = model.spec_a[1]
    for ta, table in als.items():
        if table.lower() != "title":
            continue
        year = extract_numeric_condition(sql, ta, "production_year", allowed_years)
        if not year:
            continue
        for ka, kt_table in als.items():
            if kt_table.lower() != "kind_type":
                continue
            if not re.search(rf"\b{re.escape(ka)}\.id\s*=\s*{re.escape(ta)}\.kind_id\b|\b{re.escape(ta)}\.kind_id\s*=\s*{re.escape(ka)}\.id\b", sql, re.I):
                continue
            kind = extract_text_dimension_ids(sql, ka, "kind_type", "kind")
            if not kind:
                continue
            local_sql = f"SELECT 1 FROM title AS {ta}, kind_type AS {ka} WHERE {year[0]} AND {kind[0]} AND {ka}.id = {ta}.kind_id"
            yidx = [model.idx_a[v] for v in year[1] if model.idx_a and v in model.idx_a]
            kidx = [model.idx_b[v] for v in kind[1] if model.idx_b and v in model.idx_b]
            oasis_rows, avi_rows = estimate_rect(model, yidx, kidx)
            true_rows = count_rows(local_sql)
            pg_rows = explain_rows(local_sql)
            out.append(make_candidate(path, "title.production_year/kind_id", [ta, ka], year[0] + " AND " + kind[0], local_sql, true_rows, pg_rows, avi_rows, oasis_rows, summary, {ta, ka}))
    return out


def scan_movie_companies(path: Path, sql: str, als: dict[str, str], summary: dict, model: PairModel) -> list[dict]:
    out: list[dict] = []
    for ma, table in als.items():
        if table.lower() != "movie_companies":
            continue
        for cta, ct_table in als.items():
            if ct_table.lower() != "company_type":
                continue
            if not re.search(rf"\b{re.escape(cta)}\.id\s*=\s*{re.escape(ma)}\.company_type_id\b|\b{re.escape(ma)}\.company_type_id\s*=\s*{re.escape(cta)}\.id\b", sql, re.I):
                continue
            for cna, cn_table in als.items():
                if cn_table.lower() != "company_name":
                    continue
                if not re.search(rf"\b{re.escape(cna)}\.id\s*=\s*{re.escape(ma)}\.company_id\b|\b{re.escape(ma)}\.company_id\s*=\s*{re.escape(cna)}\.id\b", sql, re.I):
                    continue
                ct_filters = extract_alias_filters(sql, cta)
                cn_filters = extract_alias_filters(sql, cna)
                if not ct_filters and not cn_filters:
                    continue
                ct_where = ct_filters or "true"
                cn_where = cn_filters or "true"
                local_sql = (
                    f"SELECT 1 FROM movie_companies AS {ma}, company_type AS {cta}, company_name AS {cna} "
                    f"WHERE {ct_where} AND {cn_where} AND {cta}.id = {ma}.company_type_id AND {cna}.id = {ma}.company_id"
                )
                ct_ids = {float(r[0]) for r in q(f"SELECT id::double precision FROM company_type AS {cta} WHERE {ct_where}")}
                ct_idx = [model.idx_a[v] for v in ct_ids if model.idx_a and v in model.idx_a]
                inner_edges = ",".join(f"{x:.6f}" for x in model.spec_b[1][1:-1])
                bucket_expr = f"width_bucket({cna}.id::double precision, ARRAY[{inner_edges}])"
                cn_bins = [int(float(r[0])) - 1 for r in q(f"SELECT DISTINCT {bucket_expr} FROM company_name AS {cna} WHERE {cn_where}")]
                cn_bins = sorted({i for i in cn_bins if 0 <= i < xcol.nbins(model.spec_b)})
                oasis_rows, avi_rows = estimate_rect(model, ct_idx, cn_bins)
                true_rows = count_rows(local_sql)
                pg_rows = explain_rows(local_sql)
                predicates = " AND ".join(x for x in [ct_filters, cn_filters] if x)
                out.append(make_candidate(path, "movie_companies.company_type_id/company_id", [ma, cta, cna], predicates, local_sql, true_rows, pg_rows, avi_rows, oasis_rows, summary, {ma, cta, cna}))
    return out


def make_candidate(path: Path, pair: str, local_aliases: list[str], predicates: str, local_sql: str, true_rows: int, pg_rows: int, avi_rows: float, oasis_rows: float, summary: dict, wanted: set[str]) -> dict:
    branch = smallest_branch(summary, wanted)
    return {
        "query": path.name,
        "pair": pair,
        "local_hint_aliases": sorted(local_aliases),
        "predicate_summary": predicates,
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
        "full_query_sql": None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="/home/tianqc/experiments/build/imdb_testbed_v1/queries_il")
    ap.add_argument("--out", default="/home/tianqc/job_pg/results_plan_level/multipair_candidates_variants.json")
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

    models = {
        "title": build_pair_model("title", "production_year", "kind_id", args.k, args.seed),
        "movie_companies": build_pair_model("movie_companies", "company_type_id", "company_id", args.k, args.seed),
    }
    results: list[dict] = []
    stats = {
        "files_scanned": 0,
        "title_kind_candidates": 0,
        "movie_companies_candidates": 0,
        "aka_title_candidates": 0,
        "cast_info_nr_order_role_candidates": 0,
    }
    for path in sorted(Path(args.queries).glob("*.sql")):
        stats["files_scanned"] += 1
        try:
            sql = strip_sql(path.read_text())
            als = aliases(sql)
            summary = summarize_plan(plan_json(sql))
            title_rows = scan_title_kind(path, sql, als, summary, models["title"])
            mc_rows = scan_movie_companies(path, sql, als, summary, models["movie_companies"])
            for row in title_rows + mc_rows:
                row["full_query_sql"] = sql
            stats["title_kind_candidates"] += len(title_rows)
            stats["movie_companies_candidates"] += len(mc_rows)
            results.extend(title_rows)
            results.extend(mc_rows)
        except Exception as exc:
            results.append({"query": path.name, "error": str(exc)})

    results.sort(key=lambda r: (r.get("pg_local_qerr", 0), r.get("true_rows", 0)), reverse=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stats": stats, "candidates": results}, indent=2))
    flat = out.with_name(out.stem + "_flat.json")
    flat.write_text(json.dumps(results, indent=2))
    print(f"wrote {out} and {flat} with {len(results)} rows")
    print(json.dumps(stats, indent=2))
    print(f"{'query':14} {'pair':43} {'aliases':14} {'true':>9} {'PG':>9} {'OASIS':>9} {'PGq':>7} {'Oq':>7} branch")
    for r in results[:30]:
        if "error" in r:
            print(f"{r['query']:14} ERROR {r['error'][:120]}")
            continue
        b = r.get("affected_branch") or {}
        print(
            f"{r['query']:14} {r['pair'][:43]:43} {','.join(r['local_hint_aliases'])[:14]:14} "
            f"{r['true_rows']:9.0f} {r['pg_local_rows']:9.0f} {r['oasis_rows']:9.0f} "
            f"{r['pg_local_qerr']:7.2f} {r['oasis_qerr']:7.2f} {b.get('node_type')} {b.get('leaf_aliases')}"
        )


if __name__ == "__main__":
    main()
