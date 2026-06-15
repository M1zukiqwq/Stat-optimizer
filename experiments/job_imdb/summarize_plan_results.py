#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


PATHS = [
    ("variants/local positive", "/home/tianqc/job_pg/results_plan_level/hint_eval_variants_80_local.json"),
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
    for label, path in PATHS:
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
