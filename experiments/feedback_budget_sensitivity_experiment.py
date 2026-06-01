#!/usr/bin/env python3
"""Feedback-budget sensitivity for OASIS-style statistics correction.

The trained OASIS checkpoint has a fixed K=16 tensor width. To study smaller
feedback budgets without retraining, this experiment truncates each sample to
the K most recent observations and still tensorizes it into the checkpoint's
native padded width. All feedback-driven methods see exactly the same truncated
observation window.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_PIPELINE_DIR = _REPO_DIR / "cdf_kll_ml_pipeline"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from histogram_types import KllFeedbackSample
from json_histogram_parser import load_feedback_sample
from mlp_histogram_model_v2 import MlpHistogramModelV2
from optimizer_decision_proxy_experiment import (
    METHOD_ORDER,
    CostProxyConfig,
    build_method_boundaries,
    boundaries_from_quantiles,
    estimate_selectivity,
    generate_predicates,
    geomean,
    iter_sample_paths,
    pct_improvement,
    qerr,
    regret_for_join,
    regret_for_scan,
)


@dataclass
class BudgetDecisionRow:
    feedback_k: int
    q_mods: int
    case_id: str
    predicate_id: int
    predicate_type: str
    table_rows: int
    true_selectivity: float
    method: str
    estimated_selectivity: float
    selectivity_qerr: float
    scan_choice: str
    scan_optimal: str
    scan_regret: float
    join_choice: str
    join_optimal: str
    join_regret: float


def truncate_sample(sample: KllFeedbackSample, feedback_k: int) -> KllFeedbackSample:
    observations = sorted(sample.observations, key=lambda obs: obs.timestamp)[-feedback_k:]
    return KllFeedbackSample(
        prior=sample.prior,
        observations=list(observations),
        corrected_quantile_values=sample.corrected_quantile_values,
        source_path=sample.source_path,
    )


def build_boundaries_for_window(
    sample: KllFeedbackSample,
    model: MlpHistogramModelV2,
    num_buckets: int,
    model_window: int,
) -> Tuple[Dict[str, List[float]], str]:
    return build_method_boundaries(
        sample,
        model=model,
        num_buckets=num_buckets,
        max_observations=model_window,
    )


def aggregate_rows(rows: Sequence[BudgetDecisionRow], risk_threshold: float) -> List[dict]:
    grouped: Dict[Tuple[int, str], List[BudgetDecisionRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.feedback_k, row.method)].append(row)

    summary = []
    for feedback_k in sorted({row.feedback_k for row in rows}):
        stale_rows = grouped[(feedback_k, "stale")]
        stale_by_key = {
            (row.feedback_k, row.q_mods, row.case_id, row.predicate_id): row
            for row in stale_rows
        }
        fresh_by_key = {
            (row.feedback_k, row.q_mods, row.case_id, row.predicate_id): row
            for row in grouped[(feedback_k, "fresh")]
        }
        stale_sel_qerr = geomean([row.selectivity_qerr for row in stale_rows])
        stale_join_regret = geomean([row.join_regret for row in stale_rows])

        for method in METHOD_ORDER:
            method_rows = grouped[(feedback_k, method)]
            sel_qerr = geomean([row.selectivity_qerr for row in method_rows])
            scan_regret = geomean([row.scan_regret for row in method_rows])
            join_regret = geomean([row.join_regret for row in method_rows])
            scan_match = sum(row.scan_choice == row.scan_optimal for row in method_rows) / max(len(method_rows), 1)
            join_match = sum(row.join_choice == row.join_optimal for row in method_rows) / max(len(method_rows), 1)

            risky = 0
            resolved = 0
            losses = 0
            fresh_match = 0
            for row in method_rows:
                key = (row.feedback_k, row.q_mods, row.case_id, row.predicate_id)
                stale = stale_by_key.get(key)
                fresh = fresh_by_key.get(key)
                if stale is not None:
                    if stale.join_regret >= risk_threshold:
                        risky += 1
                        if row.join_regret < risk_threshold:
                            resolved += 1
                    elif row.join_regret >= risk_threshold:
                        losses += 1
                if fresh is not None and row.join_choice == fresh.join_choice:
                    fresh_match += 1

            summary.append({
                "feedback_k": feedback_k,
                "method": method,
                "n": len(method_rows),
                "selectivity_qerr_gm": sel_qerr,
                "selectivity_qerr_improvement_pct": pct_improvement(stale_sel_qerr, sel_qerr),
                "scan_regret_gm": scan_regret,
                "join_regret_gm": join_regret,
                "join_regret_improvement_pct": pct_improvement(stale_join_regret, join_regret),
                "scan_optimal_match_frac": scan_match,
                "join_optimal_match_frac": join_match,
                "join_fresh_match_frac": fresh_match / max(len(method_rows), 1),
                "risky_stale_cases": risky,
                "risk_resolved_frac": resolved / max(risky, 1),
                "new_risk_loss_frac": losses / max(len(method_rows), 1),
            })
    return summary


def method_label(method: str) -> str:
    return {
        "stale": "Stale",
        "isomer": "ISOMER",
        "oasis": "OASIS-noProj",
        "oasis_projected": "OASIS",
        "hybrid": "Hybrid",
        "aggressive_hybrid": "Aggressive",
        "fresh": "Fresh",
    }[method]


def write_latex_table(output_dir: Path, summary: Sequence[dict], hybrid_choices: Dict[int, Counter]) -> None:
    by_key = {(row["feedback_k"], row["method"]): row for row in summary}
    path = output_dir / "table_feedback_budget_sensitivity.tex"
    with path.open("w") as handle:
        handle.write("\\begin{table*}[t]\n")
        handle.write("  \\centering\n")
        handle.write("  \\small\n")
        handle.write("  \\caption{Feedback-budget sensitivity in the optimizer-decision proxy. All methods see only the $K$ most recent feedback observations; OASIS uses the same fixed-width checkpoint with padding. Values are geometric means over all held-out predicates.}\n")
        handle.write("  \\label{tab:feedback_budget_sensitivity}\n")
        handle.write("  \\setlength{\\tabcolsep}{4pt}\n")
        handle.write("  \\resizebox{\\textwidth}{!}{%\n")
        handle.write("  \\begin{tabular}{crrrrrrrrrr}\n")
        handle.write("    \\toprule\n")
        handle.write("    $K$ & Stale QE & ISOMER QE & OASIS-noProj QE & OASIS QE & Hybrid QE & Aggressive QE & OASIS-noProj JoinOpt & OASIS JoinOpt & Hybrid JoinOpt & Hybrid choice \\\\\n")
        handle.write("    \\midrule\n")
        for feedback_k in sorted(hybrid_choices):
            stale = by_key[(feedback_k, "stale")]
            isomer = by_key[(feedback_k, "isomer")]
            oasis = by_key[(feedback_k, "oasis")]
            projected = by_key[(feedback_k, "oasis_projected")]
            hybrid = by_key[(feedback_k, "hybrid")]
            aggressive = by_key[(feedback_k, "aggressive_hybrid")]
            choices = hybrid_choices[feedback_k]
            total = sum(choices.values())
            choice_text = ", ".join(
                f"{method_label(method)} {choices[method] / total * 100:.0f}\\%"
                for method in ["isomer", "oasis_projected", "oasis", "stale"]
                if choices.get(method, 0)
            )
            handle.write(
                f"    {feedback_k} & {stale['selectivity_qerr_gm']:.3f} & "
                f"{isomer['selectivity_qerr_gm']:.3f} & {oasis['selectivity_qerr_gm']:.3f} & "
                f"{projected['selectivity_qerr_gm']:.3f} & {hybrid['selectivity_qerr_gm']:.3f} & "
                f"{aggressive['selectivity_qerr_gm']:.3f} & "
                f"{oasis['join_optimal_match_frac'] * 100:.1f}\\% & "
                f"{projected['join_optimal_match_frac'] * 100:.1f}\\% & "
                f"{hybrid['join_optimal_match_frac'] * 100:.1f}\\% & {choice_text} \\\\\n"
            )
        handle.write("    \\bottomrule\n")
        handle.write("  \\end{tabular}%\n")
        handle.write("  }\n")
        handle.write("\\end{table*}\n")


def write_outputs(
    output_dir: Path,
    rows: Sequence[BudgetDecisionRow],
    summary: Sequence[dict],
    hybrid_choices: Dict[int, Counter],
    write_rows: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(list(summary), handle, indent=2)
    with (output_dir / "hybrid_choices.json").open("w") as handle:
        json.dump({str(k): dict(v) for k, v in hybrid_choices.items()}, handle, indent=2)
    if write_rows:
        with (output_dir / "decision_rows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
    write_latex_table(output_dir, summary, hybrid_choices)
    write_text_summary(output_dir, summary, hybrid_choices)


def write_text_summary(output_dir: Path, summary: Sequence[dict], hybrid_choices: Dict[int, Counter]) -> None:
    by_key = {(row["feedback_k"], row["method"]): row for row in summary}
    lines = [
        "Feedback-budget sensitivity",
        "=" * 36,
        "All methods see the same truncated feedback window.",
        "",
        "K  Method          SelQE  JoinReg  JoinOpt  RiskResolved",
        "-" * 66,
    ]
    for feedback_k in sorted(hybrid_choices):
        for method in METHOD_ORDER:
            row = by_key[(feedback_k, method)]
            lines.append(
                f"{feedback_k:<2d} {method:<15s} {row['selectivity_qerr_gm']:5.3f}  "
                f"{row['join_regret_gm']:7.4f}  "
                f"{row['join_optimal_match_frac'] * 100:7.1f}%  "
                f"{row['risk_resolved_frac'] * 100:12.1f}%"
            )
        choices = hybrid_choices[feedback_k]
        total = sum(choices.values())
        choice_text = ", ".join(
            f"{method}={choices[method] / total * 100:.1f}%"
            for method in sorted(choices)
        )
        lines.append(f"   Hybrid choices: {choice_text}")
        lines.append("")
    text = "\n".join(lines)
    (output_dir / "summary.txt").write_text(text + "\n")
    print(text)


def run_experiment(args: argparse.Namespace) -> None:
    model = MlpHistogramModelV2.load(str(args.model_path))
    model_window = int(getattr(model, "max_observations", 16))
    for feedback_k in args.feedback_budgets:
        if feedback_k > model_window:
            raise ValueError(f"feedback budget K={feedback_k} exceeds model window {model_window}")

    cfg = CostProxyConfig(
        seq_tuple_cost=args.seq_tuple_cost,
        index_startup_cost=args.index_startup_cost,
        index_tuple_cost=args.index_tuple_cost,
        dim_rows=args.dim_rows,
        hash_build_tuple_cost=args.hash_build_tuple_cost,
        hash_probe_tuple_cost=args.hash_probe_tuple_cost,
        nl_lookup_cost=args.nl_lookup_cost,
    )

    sample_paths = iter_sample_paths(args.data_root, args.q_values, args.max_cases_per_q)
    if not sample_paths:
        raise FileNotFoundError(f"No test samples found under {args.data_root}")

    rows: List[BudgetDecisionRow] = []
    hybrid_choices: Dict[int, Counter] = {feedback_k: Counter() for feedback_k in args.feedback_budgets}

    for sample_index, (q_mods, path) in enumerate(sample_paths):
        sample = load_feedback_sample(str(path))
        fresh = boundaries_from_quantiles(sample.corrected_quantile_values or sample.prior.quantile_values)
        rng = random.Random(args.seed + q_mods * 100_000 + sample_index)
        table_rows = int(10 ** rng.uniform(math.log10(args.min_table_rows), math.log10(args.max_table_rows)))
        predicates = generate_predicates(
            fresh,
            rng=rng,
            count=args.predicates_per_case,
            min_true_selectivity=args.min_true_selectivity,
        )

        for feedback_k in args.feedback_budgets:
            window_sample = truncate_sample(sample, feedback_k)
            method_boundaries, hybrid_choice = build_boundaries_for_window(
                window_sample,
                model=model,
                num_buckets=args.num_buckets,
                model_window=model_window,
            )
            hybrid_choices[feedback_k][hybrid_choice] += 1

            for pred_id, predicate in enumerate(predicates):
                true_sel = estimate_selectivity(fresh, predicate)
                for method in METHOD_ORDER:
                    estimated = estimate_selectivity(method_boundaries[method], predicate)
                    scan_choice, scan_optimal, scan_regret = regret_for_scan(estimated, true_sel, table_rows, cfg)
                    join_choice, join_optimal, join_regret = regret_for_join(estimated, true_sel, table_rows, cfg)
                    rows.append(BudgetDecisionRow(
                        feedback_k=feedback_k,
                        q_mods=q_mods,
                        case_id=path.stem,
                        predicate_id=pred_id,
                        predicate_type=predicate["predicate_type"],
                        table_rows=table_rows,
                        true_selectivity=true_sel,
                        method=method,
                        estimated_selectivity=estimated,
                        selectivity_qerr=qerr(estimated, true_sel),
                        scan_choice=scan_choice,
                        scan_optimal=scan_optimal,
                        scan_regret=scan_regret,
                        join_choice=join_choice,
                        join_optimal=join_optimal,
                        join_regret=join_regret,
                    ))

        if (sample_index + 1) % 50 == 0:
            print(f"Processed {sample_index + 1}/{len(sample_paths)} samples")

    summary = aggregate_rows(rows, risk_threshold=args.risk_threshold)
    write_outputs(args.output_dir, rows, summary, hybrid_choices, write_rows=args.write_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feedback-budget sensitivity experiment")
    parser.add_argument("--data-root", type=Path,
                        default=_REPO_DIR / "experiments" / "results" / "synthetic_paper_suite_rerun_20260529" / "compound_data")
    parser.add_argument("--model-path", type=Path,
                        default=_REPO_DIR / "experiments" / "results" / "synthetic_paper_suite_rerun_20260529" / "models" / "oasis_k16.json")
    parser.add_argument("--output-dir", type=Path,
                        default=_REPO_DIR / "experiments" / "results" / "feedback_budget_sensitivity_20260529")
    parser.add_argument("--feedback-budgets", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--q-values", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--max-cases-per-q", type=int, default=128)
    parser.add_argument("--predicates-per-case", type=int, default=32)
    parser.add_argument("--num-buckets", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-true-selectivity", type=float, default=1e-4)
    parser.add_argument("--risk-threshold", type=float, default=1.05)
    parser.add_argument("--min-table-rows", type=float, default=100_000)
    parser.add_argument("--max-table-rows", type=float, default=10_000_000)
    parser.add_argument("--write-rows", action="store_true")

    parser.add_argument("--seq-tuple-cost", type=float, default=1.0)
    parser.add_argument("--index-startup-cost", type=float, default=100.0)
    parser.add_argument("--index-tuple-cost", type=float, default=8.0)
    parser.add_argument("--dim-rows", type=float, default=50_000.0)
    parser.add_argument("--hash-build-tuple-cost", type=float, default=1.0)
    parser.add_argument("--hash-probe-tuple-cost", type=float, default=0.20)
    parser.add_argument("--nl-lookup-cost", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
