#!/usr/bin/env python3
"""
Multi-seed experiment wrapper for confidence intervals.

Runs the full synthetic suite with multiple seeds, aggregates results
with mean +/- std, and produces tables with confidence intervals.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent

sys.path.insert(0, str(_REPO_DIR / "cdf_kll_ml_pipeline"))
sys.path.insert(0, str(_SCRIPT_DIR))

from run_synthetic_paper_suite import (
    METHODS,
    MAIN_Q_VALUES,
    TRAIN_Q_VALUES,
    DIST_TYPES,
    evaluate_main_suite,
    evaluate_distribution_suite,
    parse_args as suite_parse_args,
    MethodSummary,
    DistSummary,
    write_json,
    write_csv,
)


def run_single_seed(seed: int, output_root: Path, args_overrides: dict) -> Path:
    """Run the full suite with a given seed and return the output directory."""
    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Build args namespace
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=seed_dir)
    parser.add_argument("--suites", nargs="+", default=["main", "distribution"])
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--num-buckets", type=int, default=10)
    parser.add_argument("--max-observations", type=int, default=16)
    parser.add_argument("--train-samples-per-q", type=int, default=1000)
    parser.add_argument("--test-samples-per-q", type=int, default=128)
    parser.add_argument("--stholes-mode", choices=["flat", "tree"], default="tree")
    parser.add_argument("--train-lr", type=float, default=3e-4)
    parser.add_argument("--train-epochs", type=int, default=150)
    parser.add_argument("--train-alpha", type=float, default=1e-4)
    parser.add_argument("--distribution-q", type=int, default=10)
    parser.add_argument("--distribution-cases", type=int, default=200)
    parser.add_argument("--initial-rows", type=int, default=5000)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--activation-clip", type=float, default=10.0)
    parser.add_argument("--attention-score-clip", type=float, default=20.0)
    parser.add_argument("--parameter-clip", type=float, default=2.0)

    known_args = parser.parse_known_args([])[0]
    for key, value in args_overrides.items():
        setattr(known_args, key, value)
    setattr(known_args, "output_root", seed_dir)
    setattr(known_args, "seed", seed)
    setattr(known_args, "force_retrain", True)

    # Run main suite
    main_summaries = evaluate_main_suite(known_args, seed_dir)

    # Run distribution suite
    try:
        dist_summaries = evaluate_distribution_suite(known_args, seed_dir)
    except Exception as e:
        print(f"Warning: distribution suite failed for seed {seed}: {e}")
        dist_summaries = []

    return seed_dir


def aggregate_results(seed_dirs: Sequence[Path], output_root: Path) -> None:
    """Aggregate results across seeds and compute confidence intervals."""
    all_main: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    all_dist: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for seed_dir in seed_dirs:
        # Load main results
        main_file = seed_dir / "main" / "summary.json"
        if main_file.exists():
            with open(main_file) as f:
                summaries = json.load(f)
            for s in summaries:
                key = (s.get("q_mods", 0), s["method"])
                all_main[key].append(s["qerror_mean"])

        # Load distribution results
        dist_file = seed_dir / "distribution" / "summary.json"
        if dist_file.exists():
            with open(dist_file) as f:
                rows = json.load(f)
            for row in rows:
                dist_name = row["distribution"]
                method = row["method"]
                all_dist[dist_name][method].append(row["qerror_mean"])

    # Compute statistics for main results
    n_seeds = len(seed_dirs)
    results = []
    for (q_mods, method), values in sorted(all_main.items()):
        if not values:
            continue
        mean = sum(values) / len(values)
        if len(values) > 1:
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
            ci95 = 1.96 * std / math.sqrt(len(values))
        else:
            std = 0.0
            ci95 = 0.0
        results.append({
            "q_mods": q_mods,
            "method": method,
            "qerror_mean": mean,
            "qerror_std": std,
            "ci95": ci95,
            "n_seeds": len(values),
        })

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "aggregated_main.json", results)
    write_csv(output_root / "aggregated_main.csv", results)

    # Write LaTeX table with confidence intervals
    write_aggregated_table(results, output_root / "table_qerror_ci.tex")

    # Distribution aggregation
    dist_results = []
    for dist_name in DIST_TYPES:
        for method in METHODS:
            values = all_dist.get(dist_name, {}).get(method, [])
            if not values:
                continue
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)) if len(values) > 1 else 0.0
            dist_results.append({
                "distribution": dist_name,
                "method": method,
                "qerror_mean": mean,
                "qerror_std": std,
                "n_seeds": len(values),
            })

    write_json(output_root / "aggregated_distribution.json", dist_results)
    write_csv(output_root / "aggregated_distribution.csv", dist_results)

    print(f"\nAggregated results from {n_seeds} seeds written to {output_root}")


def write_aggregated_table(results: List[dict], path: Path) -> None:
    """Write LaTeX table with mean +/- CI."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Group by q_mods
    by_q: Dict[int, Dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_q[r["q_mods"]][r["method"]] = r

    display_methods = ["LinInterp", "FeedAvg", "STHoles", "QuickSel-H", "ISOMER", "OASIS"]
    q_values = sorted(by_q.keys())

    with open(path, "w") as handle:
        handle.write("\\begin{table*}[!htb]\n")
        handle.write("  \\centering\n")
        handle.write("  \\small\n")
        handle.write("  \\caption{Q-Error comparison across drift intensities with 95\\% confidence intervals ($\\downarrow$ better). "
                      "Results aggregated over multiple random seeds. Bold = best per row.}\n")
        handle.write("  \\label{tab:qerror_ci}\n")
        handle.write("  \\setlength{\\tabcolsep}{3pt}\n")

        # Header
        col_count = 1 + len(display_methods)  # q + methods
        handle.write("  \\begin{tabular}{c | " + " ".join("rr" for _ in display_methods) + "}\n")
        handle.write("    \\toprule\n")
        header_cells = [f"\\multicolumn{{2}}{{c}}{{{m}}}" for m in display_methods]
        handle.write("    $q$ & " + " & ".join(header_cells) + " \\\\\n")
        handle.write("    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\\cmidrule(lr){12-13}\n")
        handle.write("    & Q-Err & CI & Q-Err & CI & Q-Err & CI & Q-Err & CI & Q-Err & CI & Q-Err & CI \\\\\n")
        handle.write("    \\midrule\n")

        for q in q_values:
            per_q = by_q.get(q, {})
            prior_data = per_q.get("Prior", {})
            prior_qerr = prior_data.get("qerror_mean", 0)

            # Find best non-Prior method
            best_qerr = float("inf")
            for m in display_methods:
                d = per_q.get(m, {})
                qerr = d.get("qerror_mean", float("inf"))
                if qerr < best_qerr:
                    best_qerr = qerr

            cells = [f"    \\textbf{{{q:2d}}}"]
            for m in display_methods:
                d = per_q.get(m, {})
                qerr = d.get("qerror_mean", 0)
                ci = d.get("ci95", 0)
                improvement = (prior_qerr - qerr) / max(prior_qerr, 1e-12) * 100 if prior_qerr > 0 else 0

                qerr_str = f"{qerr:.3f}"
                ci_str = f"$\\pm${ci:.3f}" if ci > 0.001 else "$\\pm$0.000"

                if abs(qerr - best_qerr) < 1e-6:
                    qerr_str = f"\\textbf{{{qerr_str}}}"
                    ci_str = f"\\textbf{{{ci_str}}}"

                cells.append(f"& {qerr_str} & {ci_str}")

            handle.write(" ".join(cells) + " \\\\\n")

        handle.write("    \\bottomrule\n")
        handle.write("  \\end{tabular}\n")
        handle.write("\\end{table*}\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed experiment runner")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--output-root", type=Path, default=_REPO_DIR / "experiments" / "results" / "multiseed_suite")
    parser.add_argument("--num-buckets", type=int, default=10)
    parser.add_argument("--max-observations", type=int, default=16)
    parser.add_argument("--train-samples-per-q", type=int, default=1000)
    parser.add_argument("--test-samples-per-q", type=int, default=128)
    parser.add_argument("--stholes-mode", choices=["flat", "tree"], default="tree")
    parser.add_argument("--distribution-q", type=int, default=10)
    parser.add_argument("--distribution-cases", type=int, default=200)
    args = parser.parse_args()

    args_overrides = {
        "num_buckets": args.num_buckets,
        "max_observations": args.max_observations,
        "train_samples_per_q": args.train_samples_per_q,
        "test_samples_per_q": args.test_samples_per_q,
        "stholes_mode": args.stholes_mode,
        "distribution_q": args.distribution_q,
        "distribution_cases": args.distribution_cases,
    }

    seed_dirs = []
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Running seed = {seed}")
        print(f"{'='*60}")
        seed_dir = run_single_seed(seed, args.output_root, args_overrides)
        seed_dirs.append(seed_dir)

    print(f"\n{'='*60}")
    print("Aggregating results across seeds")
    print(f"{'='*60}")
    aggregate_results(seed_dirs, args.output_root / "aggregated")


if __name__ == "__main__":
    main()
