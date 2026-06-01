#!/usr/bin/env python3
"""Generate plan-change analysis tables from PostgreSQL batch output.

The PostgreSQL planner-only experiment already records one row per
configuration/query/method. This script derives two publication artifacts:

* family-level plan-shape recovery for scan/join query families;
* representative plan-change examples where corrected statistics match the
  fresh-statistics plan and stale statistics do not.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


METHODS = ["stale", "isomer", "oasis", "oasis_projected", "hybrid", "fresh"]
FAMILIES = ["scan", "join", "join_dim_filter"]


def method_label(method: str) -> str:
    return {
        "stale": "Stale",
        "isomer": "ISOMER",
        "oasis": "OASIS-noProj",
        "oasis_projected": "OASIS",
        "hybrid": "Hybrid",
        "fresh": "Fresh",
    }[method]


def family_label(family: str) -> str:
    return {
        "scan": "Selection",
        "join": "Join",
        "join_dim_filter": "Join + dimension filter",
    }[family]


def read_rows(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def group_by_query(rows: Iterable[dict]) -> Dict[Tuple[str, str], Dict[str, dict]]:
    grouped: Dict[Tuple[str, str], Dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[(row["config_id"], row["query_id"])][row["method"]] = row
    return grouped


def pct(value: float) -> str:
    return f"{value * 100:.1f}\\%"


def family_breakdown(grouped: Dict[Tuple[str, str], Dict[str, dict]]) -> List[dict]:
    result = []
    for family in FAMILIES:
        items = [methods for methods in grouped.values() if methods["fresh"]["family"] == family]
        changed = [
            methods for methods in items
            if methods["stale"]["plan_signature"] != methods["fresh"]["plan_signature"]
        ]
        unchanged = [
            methods for methods in items
            if methods["stale"]["plan_signature"] == methods["fresh"]["plan_signature"]
        ]
        for method in ["stale", "oasis", "isomer", "oasis_projected", "hybrid"]:
            fresh_match = sum(
                methods[method]["plan_signature"] == methods["fresh"]["plan_signature"]
                for methods in items
            ) / max(len(items), 1)
            recovery = sum(
                methods[method]["plan_signature"] == methods["fresh"]["plan_signature"]
                for methods in changed
            ) / max(len(changed), 1)
            new_dev = sum(
                methods[method]["plan_signature"] != methods["fresh"]["plan_signature"]
                for methods in unchanged
            ) / max(len(unchanged), 1)
            result.append({
                "family": family,
                "method": method,
                "n": len(items),
                "changed": len(changed),
                "fresh_plan_match_frac": fresh_match,
                "plan_recovery_frac": recovery,
                "new_plan_deviation_frac": new_dev,
            })
    return result


def plan_descriptor(row: dict) -> str:
    root = row["root_node"]
    scans = row["scan_nodes"].replace("|", "; ")
    joins = row["join_nodes"].replace("|", "; ")
    if joins:
        return f"{root}: {joins}; {scans}"
    return f"{root}: {scans}"


def example_rows(
    grouped: Dict[Tuple[str, str], Dict[str, dict]],
    limit: int = 4,
    min_true_rows: int = 10,
) -> List[dict]:
    candidates = []
    for (config_id, query_id), methods in grouped.items():
        if any(method not in methods for method in METHODS):
            continue
        stale = methods["stale"]
        fresh = methods["fresh"]
        projected = methods["oasis_projected"]
        if stale["plan_signature"] == fresh["plan_signature"]:
            continue
        if projected["plan_signature"] != fresh["plan_signature"]:
            continue
        if stale["root_node"] == fresh["root_node"]:
            continue
        if int(float(fresh["true_rows"])) < min_true_rows:
            continue
        score = float(stale["row_qerr"]) - float(projected["row_qerr"])
        candidates.append((score, config_id, query_id, methods))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    selected = []
    seen_families = set()
    for _, config_id, query_id, methods in candidates:
        family = methods["fresh"]["family"]
        if family in seen_families and len(seen_families) < len(FAMILIES):
            continue
        seen_families.add(family)
        selected.append({
            "config_id": config_id,
            "query_id": query_id,
            "family": family,
            "predicate": methods["fresh"]["predicate_id"],
            "true_rows": methods["fresh"]["true_rows"],
            "stale_rows": methods["stale"]["plan_rows"],
            "projected_rows": methods["oasis_projected"]["plan_rows"],
            "fresh_rows": methods["fresh"]["plan_rows"],
            "stale_qerr": methods["stale"]["row_qerr"],
            "projected_qerr": methods["oasis_projected"]["row_qerr"],
            "fresh_qerr": methods["fresh"]["row_qerr"],
            "stale_plan": plan_descriptor(methods["stale"]),
            "projected_plan": plan_descriptor(methods["oasis_projected"]),
            "fresh_plan": plan_descriptor(methods["fresh"]),
        })
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        selected_ids = {(row["config_id"], row["query_id"]) for row in selected}
        for _, config_id, query_id, methods in candidates:
            if (config_id, query_id) in selected_ids:
                continue
            selected.append({
                "config_id": config_id,
                "query_id": query_id,
                "family": methods["fresh"]["family"],
                "predicate": methods["fresh"]["predicate_id"],
                "true_rows": methods["fresh"]["true_rows"],
                "stale_rows": methods["stale"]["plan_rows"],
                "projected_rows": methods["oasis_projected"]["plan_rows"],
                "fresh_rows": methods["fresh"]["plan_rows"],
                "stale_qerr": methods["stale"]["row_qerr"],
                "projected_qerr": methods["oasis_projected"]["row_qerr"],
                "fresh_qerr": methods["fresh"]["row_qerr"],
                "stale_plan": plan_descriptor(methods["stale"]),
                "projected_plan": plan_descriptor(methods["oasis_projected"]),
                "fresh_plan": plan_descriptor(methods["fresh"]),
            })
            if len(selected) >= limit:
                break
    return selected


def write_family_table(path: Path, rows: Sequence[dict]) -> None:
    by_key = {(row["family"], row["method"]): row for row in rows}
    with path.open("w") as handle:
        handle.write("\\begin{table}[t]\n")
        handle.write("  \\centering\n")
        handle.write("  \\small\n")
        handle.write("  \\caption{PostgreSQL planner-only plan-shape breakdown by query family. Recovery is measured only on queries where stale and fresh statistics produce different plan shapes; NewDev is measured where stale already matches fresh.}\n")
        handle.write("  \\label{tab:postgres_plan_family_breakdown}\n")
        handle.write("  \\setlength{\\tabcolsep}{4pt}\n")
        handle.write("  \\resizebox{\\textwidth}{!}{%\n")
        handle.write("  \\begin{tabular}{lrrrrrrrrr}\n")
        handle.write("    \\toprule\n")
        handle.write("    \\multirow{2}{*}{Family} & \\multirow{2}{*}{Changed} & \\multicolumn{3}{c}{Stale} & \\multicolumn{3}{c}{OASIS} & \\multicolumn{2}{c}{OASIS} \\\\\n")
        handle.write("    \\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}\n")
        handle.write("     & & FreshPlan & Recovery & NewDev & FreshPlan & Recovery & NewDev & FreshPlan & NewDev \\\\\n")
        handle.write("    \\midrule\n")
        for family in FAMILIES:
            stale = by_key[(family, "stale")]
            oasis = by_key[(family, "oasis")]
            projected = by_key[(family, "oasis_projected")]
            handle.write(
                f"    {family_label(family)} & {stale['changed']}/{stale['n']} & "
                f"{pct(stale['fresh_plan_match_frac'])} & {pct(stale['plan_recovery_frac'])} & {pct(stale['new_plan_deviation_frac'])} & "
                f"{pct(oasis['fresh_plan_match_frac'])} & {pct(oasis['plan_recovery_frac'])} & {pct(oasis['new_plan_deviation_frac'])} & "
                f"{pct(projected['fresh_plan_match_frac'])} & {pct(projected['new_plan_deviation_frac'])} \\\\\n"
            )
        handle.write("    \\bottomrule\n")
        handle.write("  \\end{tabular}%\n")
        handle.write("  }\n")
        handle.write("\\end{table}\n")


def shorten(text: str, max_len: int = 92) -> str:
    text = text.replace("_", "\\_")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def write_examples_table(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w") as handle:
        handle.write("\\begin{table*}[t]\n")
        handle.write("  \\centering\n")
        handle.write("  \\scriptsize\n")
        handle.write("  \\caption{Representative PostgreSQL plan changes caused by corrected statistics. Each row is a query where stale statistics disagree with fresh statistics and OASIS matches the fresh plan shape.}\n")
        handle.write("  \\label{tab:postgres_plan_examples}\n")
        handle.write("  \\setlength{\\tabcolsep}{3pt}\n")
        handle.write("  \\resizebox{\\textwidth}{!}{%\n")
        handle.write("  \\begin{tabular}{llllrrrl}\n")
        handle.write("    \\toprule\n")
        handle.write("    Family & Predicate & True & Method & Est. rows & Row QE & Root & Plan summary \\\\\n")
        handle.write("    \\midrule\n")
        for row in rows:
            for method, label in [("stale", "Stale"), ("projected", "OASIS"), ("fresh", "Fresh")]:
                handle.write(
                    f"    {family_label(row['family'])} & {row['predicate'].replace('_', '\\_')} & {row['true_rows']} & {label} & "
                    f"{float(row[f'{method}_rows']):.0f} & {float(row[f'{method}_qerr']):.1f} & "
                    f"{shorten(row[f'{method}_plan'].split(':', 1)[0], 32)} & {shorten(row[f'{method}_plan'])} \\\\\n"
                )
            handle.write("    \\addlinespace\n")
        handle.write("    \\bottomrule\n")
        handle.write("  \\end{tabular}%\n")
        handle.write("  }\n")
        handle.write("\\end{table*}\n")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: Sequence[dict], examples: Sequence[dict]) -> None:
    lines = ["PostgreSQL plan-change analysis", "=" * 36, ""]
    for row in rows:
        if row["method"] != "oasis_projected":
            continue
        lines.append(
            f"{family_label(row['family'])}: changed={row['changed']}/{row['n']}, "
            f"FreshPlan={row['fresh_plan_match_frac'] * 100:.1f}%, "
            f"Recovery={row['plan_recovery_frac'] * 100:.1f}%, "
            f"NewDev={row['new_plan_deviation_frac'] * 100:.1f}%"
        )
    lines.extend(["", "Examples:"])
    for row in examples:
        lines.append(
            f"- {row['config_id']} {row['query_id']}: "
            f"stale {row['stale_plan']} -> OASIS {row['projected_plan']}"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PostgreSQL planner-only plan changes")
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=Path("experiments/results/postgres_planner_stats_injection_batch_20260529"),
    )
    parser.add_argument("--example-limit", type=int, default=3)
    parser.add_argument("--min-true-rows", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.batch_dir / "batch_plan_rows.csv")
    grouped = group_by_query(rows)
    breakdown = family_breakdown(grouped)
    examples = example_rows(grouped, limit=args.example_limit, min_true_rows=args.min_true_rows)

    write_csv(args.batch_dir / "plan_family_breakdown.csv", breakdown)
    write_csv(args.batch_dir / "plan_change_examples.csv", examples)
    write_family_table(args.batch_dir / "table_postgres_plan_family_breakdown.tex", breakdown)
    write_examples_table(args.batch_dir / "table_postgres_plan_examples.tex", examples)
    write_summary(args.batch_dir / "plan_change_analysis.txt", breakdown, examples)

    print((args.batch_dir / "plan_change_analysis.txt").read_text())


if __name__ == "__main__":
    main()
