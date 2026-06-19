#!/usr/bin/env python3
"""Evaluate pg_hint_plan Rows injections for plan-level OASIS JOB cases."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


PSQL: list[str] = []
STATEMENT_TIMEOUT = "20min"


def run_explain(sql: str, analyze: bool, hint: str | None = None) -> dict:
    opts = "ANALYZE, FORMAT JSON" if analyze else "FORMAT JSON"
    hint_sql = f"/*+ {hint} */ " if hint else ""
    wrapped = f"EXPLAIN {hint_sql}({opts}) {sql}"
    out = subprocess.run(PSQL + ["-qAtX", "-v", "ON_ERROR_STOP=1", "-c", wrapped], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:1200])
    text = out.stdout.strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise RuntimeError("could not find JSON EXPLAIN output: " + text[:500])
    return json.loads(text[start : end + 1])[0]


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
            "actual_rows": node.get("Actual Rows"),
            "actual_loops": node.get("Actual Loops"),
            "total_cost": node.get("Total Cost"),
            "actual_total_time": node.get("Actual Total Time"),
            "relation": node.get("Relation Name"),
            "alias": node.get("Alias"),
            "leaf_aliases": sorted(leaves),
        }
    )
    for child in node.get("Plans", []):
        collect_nodes(child, out)
    return leaves


def all_nodes(plan: dict) -> list[dict]:
    nodes: list[dict] = []
    collect_nodes(plan, nodes)
    return nodes


def smallest_node(plan: dict, wanted: set[str]) -> dict | None:
    nodes = [n for n in all_nodes(plan) if wanted <= set(n["leaf_aliases"])]
    if not nodes:
        return None
    return min(nodes, key=lambda n: (len(n["leaf_aliases"]), n.get("total_cost") or 0))


def plan_signature(node: dict) -> object:
    kids = [plan_signature(c) for c in node.get("Plans", [])]
    leaves = sorted(leaf_aliases(node))
    return [node.get("Node Type"), leaves, kids]


def short_join_order(plan: dict) -> str:
    joins = [n for n in all_nodes(plan) if n["node_type"] and "Join" in n["node_type"]]
    parts = []
    for n in joins:
        aliases = ",".join(n["leaf_aliases"])
        parts.append(f"{n['node_type']}[{aliases}]")
    return " -> ".join(parts[:8])


def qerr(est: float, true: float) -> float:
    est, true = max(est, 1e-9), max(true, 1e-9)
    return max(est / true, true / est)


def rows_hint(aliases: list[str], rows: float) -> str:
    inside = " ".join(aliases)
    return f"Rows({inside} #{max(rows, 1.0):.6g})"


def evaluate_case(case: dict, granularity: str) -> dict:
    sql = case["full_query_sql"]
    aliases = case["affected_branch"]["leaf_aliases"]
    if granularity == "local":
        if "local_hint_aliases" in case:
            aliases = sorted(case["local_hint_aliases"])
        else:
            aliases = sorted([case["title_alias"], case["kind_alias"]])
    wanted = set(aliases)

    default_an = run_explain(sql, analyze=True)
    default_plan = default_an["Plan"]
    default_node = smallest_node(default_plan, wanted)
    if not default_node or default_node.get("actual_rows") is None:
        raise RuntimeError("could not locate affected branch in default ANALYZE")

    if granularity == "local":
        branch_pg = float(case["pg_local_rows"])
        branch_actual = float(case["true_rows"])
        oasis_branch = float(case["oasis_rows"])
        oracle_branch = branch_actual
    else:
        branch_pg = float(default_node["plan_rows"])
        branch_actual = float(default_node["actual_rows"])
        oasis_branch = branch_pg * (float(case["oasis_rows"]) / max(float(case["pg_local_rows"]), 1.0))
        oracle_branch = branch_actual

    methods = [
        ("default", None, branch_pg),
        ("oasis", rows_hint(aliases, oasis_branch), oasis_branch),
        ("oracle", rows_hint(aliases, oracle_branch), oracle_branch),
    ]
    seen: dict[str, dict] = {"default": default_an}
    sig_to_runtime: dict[str, float] = {json.dumps(plan_signature(default_plan)): float(default_an["Execution Time"])}

    method_rows = []
    for name, hint, injected_rows in methods:
        if name == "default":
            exp = default_an
        else:
            exp = run_explain(sql, analyze=False, hint=hint)
            sig = json.dumps(plan_signature(exp["Plan"]))
            if sig in sig_to_runtime:
                runtime = sig_to_runtime[sig]
                analyzed = False
            else:
                exp_an = run_explain(sql, analyze=True, hint=hint)
                runtime = float(exp_an["Execution Time"])
                sig_to_runtime[sig] = runtime
                exp = exp_an
                analyzed = True
            exp["reused_runtime"] = not analyzed
            exp["Execution Time"] = runtime
            seen[name] = exp
        node = smallest_node(exp["Plan"], wanted)
        method_rows.append(
            {
                "method": name,
                "hint": hint,
                "injected_branch_rows": injected_rows,
                "branch_plan_rows": node.get("plan_rows") if node else None,
                "branch_actual_rows": node.get("actual_rows") if node else None,
                "branch_qerr_vs_default_actual": qerr(injected_rows, branch_actual),
                "execution_ms": float(exp["Execution Time"]),
                "plan_signature": plan_signature(exp["Plan"]),
                "join_order": short_join_order(exp["Plan"]),
                "reused_runtime": exp.get("reused_runtime", False),
            }
        )

    default_ms = method_rows[0]["execution_ms"]
    for row in method_rows:
        row["runtime_ratio_vs_default"] = default_ms / row["execution_ms"] if row["execution_ms"] > 0 else None
        row["same_plan_as_default"] = row["plan_signature"] == method_rows[0]["plan_signature"]
        row["same_plan_as_oracle"] = row["plan_signature"] == method_rows[-1]["plan_signature"]

    return {
        "query": case["query"],
        "pair": case.get("pair", "title.production_year/kind_id"),
        "granularity": granularity,
        "aliases": aliases,
        "year_condition": case.get("year_condition"),
        "kind_condition": case.get("kind_condition"),
        "predicate_summary": case.get("predicate_summary"),
        "local": {
            "true_rows": case["true_rows"],
            "pg_rows": case["pg_local_rows"],
            "avi_true_rows": case["avi_true_rows"],
            "oasis_rows": case["oasis_rows"],
            "pg_qerr": case["pg_local_qerr"],
            "oasis_qerr": case["oasis_qerr"],
        },
        "branch_default": default_node,
        "branch_actual_rows": branch_actual,
        "branch_oasis_rows": oasis_branch,
        "methods": method_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="/home/tianqc/job_pg/results_plan_level/title_kind_candidates.json")
    ap.add_argument("--out", default="/home/tianqc/job_pg/results_plan_level/hint_eval.json")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--only", default="", help="Comma-separated query filenames to evaluate")
    ap.add_argument("--statement-timeout", default="20min")
    ap.add_argument("--granularity", choices=["branch", "local"], default="branch")
    ap.add_argument("--psql", default=os.environ.get("PSQL_BIN", "psql"))
    ap.add_argument("--host", default=os.environ.get("PGHOST", ""))
    ap.add_argument("--port", default=os.environ.get("PGPORT", ""))
    ap.add_argument("--db", default=os.environ.get("PGDATABASE", "imdb"))
    ap.add_argument("--user", default=os.environ.get("PGUSER", ""))
    args = ap.parse_args()

    global PSQL, STATEMENT_TIMEOUT
    STATEMENT_TIMEOUT = args.statement_timeout
    pgoptions = os.environ.get("PGOPTIONS", "")
    pg_hint_options = (
        f" -c session_preload_libraries=pg_hint_plan"
        f" -c pg_hint_plan.enable_hint=on"
        f" -c statement_timeout={STATEMENT_TIMEOUT}"
    )
    os.environ["PGOPTIONS"] = pgoptions + pg_hint_options
    PSQL = [args.psql, "-d", args.db]
    if args.host:
        PSQL += ["-h", args.host]
    if args.port:
        PSQL += ["-p", str(args.port)]
    if args.user:
        PSQL += ["-U", args.user]

    cases = json.loads(Path(args.candidates).read_text())
    cases = [c for c in cases if "error" not in c and c.get("affected_branch")]
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c["query"] in wanted]
    cases.sort(key=lambda c: (c.get("pg_local_qerr", 0), c.get("true_rows", 0)), reverse=True)
    selected = cases[: args.limit]

    results = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for case in selected:
        print(f"evaluating {case['query']} aliases={case['affected_branch']['leaf_aliases']}", flush=True)
        try:
            result = evaluate_case(case, args.granularity)
            results.append(result)
            methods = {m["method"]: m for m in result["methods"]}
            print(
                f"  default={methods['default']['execution_ms']:.2f} ms "
                f"oasis={methods['oasis']['execution_ms']:.2f} ms "
                f"oracle={methods['oracle']['execution_ms']:.2f} ms "
                f"oasis_same_oracle={methods['oasis']['same_plan_as_oracle']}",
                flush=True,
            )
        except Exception as exc:
            results.append({"query": case["query"], "error": str(exc), "case": case})
            print(f"  ERROR {exc}", flush=True)
        out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
