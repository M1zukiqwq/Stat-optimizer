#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


PATHS = [
    ("variants/local positive", "/home/tianqc/job_pg/results_plan_level/hint_eval_variants_80_local.json"),
    ("variants/title remaining local", "/home/tianqc/job_pg/results_plan_level/hint_eval_title_remaining_local.json"),
    ("variants/movie_companies improved local", "/home/tianqc/job_pg/results_plan_level/hint_eval_multipair_mc_improved_local.json"),
    ("variants/local zero", "/home/tianqc/job_pg/results_plan_level/hint_eval_variants_zero_local.json"),
    ("variants/downstream branch", "/home/tianqc/job_pg/results_plan_level/hint_eval_variants_80.json"),
    ("canonical/downstream branch", "/home/tianqc/job_pg/results_plan_level/hint_eval_short.json"),
]


def main() -> None:
    lines: list[str] = []
    lines.append("# OASIS JOB Plan-Level Rows-Hint Case Study\n\n")
    lines.append(
        "Generated on remote chen against PostgreSQL 14.17 / imdb. "
        "Hints use pg_hint_plan built from upstream PG14 branch.\n\n"
    )
    stats_path = Path("/home/tianqc/job_pg/results_plan_level/multipair_candidates_variants.json")
    if stats_path.exists():
        stats_doc = json.loads(stats_path.read_text())
        stats = stats_doc.get("stats", {})
        candidate_lookup = {
            (r.get("query"), r.get("pair", "title.production_year/kind_id")): r
            for r in stats_doc.get("candidates", [])
            if "error" not in r
        }
        lines.append("## Scan Scope\n\n")
        lines.append("| item | count |\n|---|---:|\n")
        for key in [
            "files_scanned",
            "title_kind_candidates",
            "movie_companies_candidates",
            "aka_title_candidates",
            "cast_info_nr_order_role_candidates",
        ]:
            lines.append(f"| {key} | {stats.get(key, 0)} |\n")
        lines.append("\n")
    else:
        candidate_lookup = {}

    all_ok = []
    all_err = []
    for _, path in PATHS:
        p = Path(path)
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        all_ok.extend(r for r in data if "error" not in r)
        all_err.extend(r for r in data if "error" in r)

    positives = []
    boundaries = []
    for r in all_ok:
        methods = {x["method"]: x for x in r["methods"]}
        d, o, ora = methods["default"], methods["oasis"], methods["oracle"]
        speed = d["execution_ms"] / o["execution_ms"] if o["execution_ms"] else 0
        oracle_speed = d["execution_ms"] / ora["execution_ms"] if ora["execution_ms"] else 0
        changed = not o["same_plan_as_default"]
        if changed and speed >= 1.5 and r["local"]["true_rows"] > 0:
            positives.append((speed, r, methods))
        elif changed or speed > 1.15 or oracle_speed > 1.5 or o["execution_ms"] > d["execution_ms"] * 1.15:
            boundaries.append((max(speed, oracle_speed, o["execution_ms"] / d["execution_ms"]), r, methods))

    positives.sort(reverse=True, key=lambda x: x[0])
    boundaries.sort(reverse=True, key=lambda x: x[0])
    lines.append("## Clean Positive Cases\n\n")
    if positives:
        lines.append("| query | pair | aliases | predicates | PG rows | OASIS rows | actual rows | PG qerr | OASIS qerr | default ms | OASIS ms | oracle ms | speedup | same oracle |\n")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for speed, r, methods in positives:
            d, o, ora = methods["default"], methods["oasis"], methods["oracle"]
            pair = r.get("pair", "title.production_year/kind_id")
            meta = candidate_lookup.get((r["query"], pair), {})
            predicates = r.get("predicate_summary") or meta.get("predicate_summary") or r.get("year_condition") or ""
            lines.append(
                f"| {r['query']} | {pair} | "
                f"{','.join(r['aliases'])} | {str(predicates)[:80]} | "
                f"{r['local']['pg_rows']:.1f} | {r['local']['oasis_rows']:.1f} | {r['local']['true_rows']:.1f} | "
                f"{r['local']['pg_qerr']:.2f} | {r['local']['oasis_qerr']:.2f} | "
                f"{d['execution_ms']:.1f} | {o['execution_ms']:.1f} | {ora['execution_ms']:.1f} | "
                f"{speed:.1f}x | {o['same_plan_as_oracle']} |\n"
            )
    else:
        lines.append("No positive-count case met plan-change and >=1.5x OASIS speedup.\n")
    lines.append("\n")

    lines.append("## Null / Boundary / Negative Cases\n\n")
    lines.append(f"Total completed evaluations across batches: {len(all_ok)}; errors/timeouts: {len(all_err)}.\n\n")
    if boundaries:
        lines.append("| query | pair | PG qerr | OASIS qerr | default ms | OASIS ms | oracle ms | same default | same oracle | note |\n")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|---|---|\n")
        for _, r, methods in boundaries[:18]:
            d, o, ora = methods["default"], methods["oasis"], methods["oracle"]
            note = []
            if o["execution_ms"] < d["execution_ms"] * 0.85:
                note.append("positive but below/duplicate threshold")
            if ora["execution_ms"] < d["execution_ms"] / 1.5 and not (o["execution_ms"] < d["execution_ms"] / 1.5):
                note.append("oracle-only faster")
            if o["execution_ms"] > d["execution_ms"] * 1.15:
                note.append("OASIS slower")
            if not o["same_plan_as_default"] and not note:
                note.append("plan changed, little runtime effect")
            lines.append(
                f"| {r['query']} | {r.get('pair', 'title.production_year/kind_id')} | "
                f"{r['local']['pg_qerr']:.2f} | {r['local']['oasis_qerr']:.2f} | "
                f"{d['execution_ms']:.1f} | {o['execution_ms']:.1f} | {ora['execution_ms']:.1f} | "
                f"{o['same_plan_as_default']} | {o['same_plan_as_oracle']} | {'; '.join(note)} |\n"
            )
    lines.append("\n")

    for label, path in PATHS:
        if not Path(path).exists():
            continue
        data = json.loads(Path(path).read_text())
        ok = [r for r in data if "error" not in r]
        err = [r for r in data if "error" in r]
        lines.append(f"## {label}\n\n")
        lines.append(f"Cases completed: {len(ok)}; errors/timeouts: {len(err)}.\n\n")
        rows = []
        for r in ok:
            methods = {x["method"]: x for x in r["methods"]}
            d, o, ora = methods["default"], methods["oasis"], methods["oracle"]
            ratio = d["execution_ms"] / o["execution_ms"] if o["execution_ms"] else 0
            oracle_ratio = d["execution_ms"] / ora["execution_ms"] if ora["execution_ms"] else 0
            changed = (not o["same_plan_as_default"]) or (not ora["same_plan_as_default"])
            if ratio > 1.15 or oracle_ratio > 1.15 or changed:
                rows.append((ratio, oracle_ratio, r, methods))
        rows.sort(reverse=True, key=lambda x: max(x[0], x[1]))
        if not rows:
            lines.append("No plan changes or >1.15x runtime changes.\n\n")
            continue
        lines.append(
            "| query | granularity | local PG qerr | local OASIS qerr | default ms | "
            "OASIS ms | oracle ms | OASIS same oracle | note |\n"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|\n")
        for ratio, oracle_ratio, r, methods in rows[:12]:
            d, o, ora = methods["default"], methods["oasis"], methods["oracle"]
            note: list[str] = []
            if o["execution_ms"] < d["execution_ms"] * 0.85:
                note.append(f"OASIS {ratio:.1f}x faster")
            if ora["execution_ms"] < d["execution_ms"] * 0.85 and not (
                o["execution_ms"] < d["execution_ms"] * 0.85
            ):
                note.append(f"oracle {oracle_ratio:.1f}x faster")
            if o["execution_ms"] > d["execution_ms"] * 1.15:
                note.append(f"OASIS {o['execution_ms'] / d['execution_ms']:.1f}x slower")
            if not note and ((not o["same_plan_as_default"]) or (not ora["same_plan_as_default"])):
                note.append("plan changed")
            lines.append(
                f"| {r['query']} | {r.get('granularity', 'branch')} | "
                f"{r['local']['pg_qerr']:.2f} | {r['local']['oasis_qerr']:.2f} | "
                f"{d['execution_ms']:.1f} | {o['execution_ms']:.1f} | {ora['execution_ms']:.1f} | "
                f"{o['same_plan_as_oracle']} | {'; '.join(note)} |\n"
            )
        lines.append("\n")

    out = Path("/home/tianqc/job_pg/results_plan_level/plan_level_summary.md")
    out.write_text("".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
