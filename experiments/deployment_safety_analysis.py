#!/usr/bin/env python3
"""Assemble deployment-safety evidence from cached OASIS experiments.

The underlying experiments already produce row-estimation, plan-shape,
feedback-budget, feedback-noise, and trace-grounded summaries.  This script is
presentation-only: it derives a compact table that explains when the deployment
form should use projection or residual-based gating instead of trusting plain
OASIS unconditionally.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def geomean(values: Iterable[float]) -> float:
    vals = [max(float(value), 1e-12) for value in values]
    if not vals:
        return 1.0
    return math.exp(sum(math.log(value) for value in vals) / len(vals))


def pct(value: float) -> str:
    return f"{value * 100:.1f}\\%"


def method_label(method: str) -> str:
    return {
        "stale": "S",
        "isomer": "I",
        "oasis": "O",
        "oasis_projected": "P",
        "hybrid": "H",
        "fresh": "F",
    }.get(method, method)


def choice_mix(choices: Dict[str, int]) -> str:
    total = sum(int(value) for value in choices.values())
    if total <= 0:
        return "--"
    ordered = ["isomer", "oasis_projected", "oasis", "stale"]
    parts = []
    for method in ordered:
        count = int(choices.get(method, 0))
        if count:
            parts.append(f"{method_label(method)} {count / total * 100:.0f}\\%")
    return ", ".join(parts)


def build_safety_rows(args: argparse.Namespace) -> List[dict]:
    pg_rows = read_csv(args.postgres_summary)
    pg = {(row["family"], row["method"]): row for row in pg_rows}

    budget_rows = read_csv(args.feedback_budget_summary)
    budget = {(int(row["feedback_k"]), row["method"]): row for row in budget_rows}
    budget_choices = read_json(args.feedback_budget_choices)

    noise_rows = read_csv(args.feedback_noise_summary)
    noise = {(float(row["noise_sigma"]), row["method"]): row for row in noise_rows}
    noise_choices = read_json(args.feedback_noise_choices)

    trace_rows = read_csv(args.trace_summary)
    trace_choices = read_json(args.trace_choices)
    trace_qerr = {
        method: geomean(float(row["qerror_gm"]) for row in trace_rows if row["method"] == method)
        for method in ["stale", "isomer", "oasis", "oasis_projected", "hybrid", "fresh"]
    }
    all_trace_choices: Dict[str, int] = {}
    for choices in trace_choices.values():
        for method, count in choices.items():
            all_trace_choices[method] = all_trace_choices.get(method, 0) + int(count)

    rows = [
        {
            "scenario": "PostgreSQL planner",
            "risk_signal": "New plan deviations",
            "uncontrolled": f"OASIS-noProj {pct(float(pg[('all', 'oasis')]['new_plan_deviation_frac']))}",
            "guarded": f"OASIS {pct(float(pg[('all', 'oasis_projected')]['new_plan_deviation_frac']))}",
            "secondary": f"FreshPlan {pct(float(pg[('all', 'oasis_projected')]['fresh_plan_match_frac']))}",
            "gate": "projection",
        },
        {
            "scenario": "PostgreSQL joins",
            "risk_signal": "Join new deviations",
            "uncontrolled": f"OASIS-noProj {pct(float(pg[('join', 'oasis')]['new_plan_deviation_frac']))}",
            "guarded": f"OASIS {pct(float(pg[('join', 'oasis_projected')]['new_plan_deviation_frac']))}",
            "secondary": f"Recovery {pct(float(pg[('join', 'oasis_projected')]['plan_recovery_frac']))}",
            "gate": "projection",
        },
        {
            "scenario": "Sparse feedback ($K=2$)",
            "risk_signal": "Join-optimal choices",
            "uncontrolled": f"Stale {pct(float(budget[(2, 'stale')]['join_optimal_match_frac']))}",
            "guarded": f"Hybrid {pct(float(budget[(2, 'hybrid')]['join_optimal_match_frac']))}",
            "secondary": f"NewRisk {pct(float(budget[(2, 'hybrid')]['new_risk_loss_frac']))}",
            "gate": choice_mix(budget_choices["2"]),
        },
        {
            "scenario": "Full feedback ($K=16$)",
            "risk_signal": "Join-optimal choices",
            "uncontrolled": f"Stale {pct(float(budget[(16, 'stale')]['join_optimal_match_frac']))}",
            "guarded": f"Hybrid {pct(float(budget[(16, 'hybrid')]['join_optimal_match_frac']))}",
            "secondary": f"NewRisk {pct(float(budget[(16, 'hybrid')]['new_risk_loss_frac']))}",
            "gate": choice_mix(budget_choices["16"]),
        },
        {
            "scenario": "10\\% feedback noise",
            "risk_signal": "Join-optimal choices",
            "uncontrolled": f"Stale {pct(float(noise[(0.1, 'stale')]['join_optimal_match_frac']))}",
            "guarded": f"Hybrid {pct(float(noise[(0.1, 'hybrid')]['join_optimal_match_frac']))}",
            "secondary": f"NewRisk {pct(float(noise[(0.1, 'hybrid')]['new_risk_loss_frac']))}",
            "gate": choice_mix(noise_choices["0.1"]),
        },
        {
            "scenario": "DML trace sanity",
            "risk_signal": "Selectivity Q-error",
            "uncontrolled": f"Stale {trace_qerr['stale']:.3f}",
            "guarded": f"Hybrid {trace_qerr['hybrid']:.3f}",
            "secondary": f"OASIS {trace_qerr['oasis_projected']:.3f}",
            "gate": choice_mix(all_trace_choices),
        },
    ]
    return rows


def write_csv_rows(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def escape_latex(text: str) -> str:
    return text.replace("%", "\\%")


def write_latex_table(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{table*}[t]\n")
        handle.write("  \\centering\n")
        handle.write("  \\small\n")
        handle.write("  \\caption{Deployment safety checks. Projection and residual-based Hybrid gating are evaluated without oracle/fresh information. P/I/O/S denote OASIS (full two-stage), ISOMER, OASIS-noProj, and stale choices in the residual gate.}\n")
        handle.write("  \\label{tab:deployment_safety}\n")
        handle.write("  \\setlength{\\tabcolsep}{4pt}\n")
        handle.write("  \\resizebox{\\textwidth}{!}{%\n")
        handle.write("  \\begin{tabular}{lllll}\n")
        handle.write("    \\toprule\n")
        handle.write("    Check & Risk signal & Unguarded state & Calibrated/gated state & Gate signal \\\\\n")
        handle.write("    \\midrule\n")
        for row in rows:
            calibrated = f"{row['guarded']}; {row['secondary']}"
            handle.write(
                f"    {row['scenario']} & {row['risk_signal']} & {row['uncontrolled']} & "
                f"{calibrated} & {row['gate']} \\\\\n"
            )
        handle.write("    \\bottomrule\n")
        handle.write("  \\end{tabular}%\n")
        handle.write("  }\n")
        handle.write("\\end{table*}\n")


def write_summary(path: Path, rows: Sequence[dict]) -> None:
    lines = [
        "Deployment safety checks",
        "=" * 32,
        "P/I/O/S denote OASIS (full two-stage), ISOMER, OASIS-noProj, and stale choices.",
        "",
    ]
    for row in rows:
        lines.append(
            f"{row['scenario']}: {row['risk_signal']} | "
            f"{row['uncontrolled']} -> {row['guarded']} ({row['secondary']}); gate={row['gate']}"
        )
    text = "\n".join(lines)
    path.write_text(text + "\n", encoding="utf-8")
    print(text)


def run(args: argparse.Namespace) -> None:
    rows = build_safety_rows(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output_dir / "deployment_safety_summary.csv", rows)
    with (args.output_dir / "deployment_safety_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    write_latex_table(args.output_dir / "table_deployment_safety.tex", rows)
    write_summary(args.output_dir / "summary.txt", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deployment-safety table from cached summaries")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "experiments" / "results" / "deployment_safety_20260529")
    parser.add_argument("--postgres-summary", type=Path,
                        default=ROOT / "experiments" / "results" / "postgres_planner_stats_injection_batch_20260529" / "batch_summary.csv")
    parser.add_argument("--feedback-budget-summary", type=Path,
                        default=ROOT / "experiments" / "results" / "feedback_budget_sensitivity_20260529" / "summary.csv")
    parser.add_argument("--feedback-budget-choices", type=Path,
                        default=ROOT / "experiments" / "results" / "feedback_budget_sensitivity_20260529" / "hybrid_choices.json")
    parser.add_argument("--feedback-noise-summary", type=Path,
                        default=ROOT / "experiments" / "results" / "feedback_noise_robustness_20260529" / "summary.csv")
    parser.add_argument("--feedback-noise-choices", type=Path,
                        default=ROOT / "experiments" / "results" / "feedback_noise_robustness_20260529" / "hybrid_choices.json")
    parser.add_argument("--trace-summary", type=Path,
                        default=ROOT / "experiments" / "results" / "trace_grounded_drift_20260529" / "summary.csv")
    parser.add_argument("--trace-choices", type=Path,
                        default=ROOT / "experiments" / "results" / "trace_grounded_drift_20260529" / "hybrid_choices.json")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
