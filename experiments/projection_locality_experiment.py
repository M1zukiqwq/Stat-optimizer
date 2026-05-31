#!/usr/bin/env python3
"""Projection-initialization and feedback-locality diagnostics for OASIS.

The experiment is intentionally single-column and optimizer-facing. It answers
two questions that are hard to see from aggregate downstream tables:

1. If the same feedback-consistency projection is initialized from different
   marginals, does the learned OASIS marginal improve held-out future
   predicates over the stale/ISOMER initialization?
2. How do the methods behave as future predicates move away from the feedback
   predicates that supplied the projection constraints?
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_PIPELINE_DIR = _REPO_DIR / "cdf_kll_ml_pipeline"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from histogram_math import evaluate_piecewise_cdf, inverse_piecewise_cdf
from json_histogram_parser import load_feedback_sample
from mlp_histogram_model_v2 import MlpHistogramModelV2
from optimizer_decision_proxy_experiment import (
    boundaries_from_quantiles,
    estimate_selectivity,
    generate_predicates,
    geomean,
    isomer_boundaries,
    iter_sample_paths,
    oasis_boundaries,
    observations_to_dicts,
    pct_improvement,
    qerr,
)


METHOD_ORDER = [
    "stale",
    "proj_stale",
    "oasis_no_proj",
    "proj_oasis",
    "proj_fresh",
    "fresh",
]

METHOD_LABELS = {
    "stale": "Stale",
    "proj_stale": "Proj(stale)/ISOMER",
    "oasis_no_proj": "OASIS-noProj",
    "proj_oasis": "Proj(OASIS)",
    "proj_fresh": "Proj(fresh)",
    "fresh": "Fresh",
}


@dataclass
class PredicateRow:
    q_mods: int
    case_id: str
    predicate_id: int
    predicate_type: str
    true_selectivity: float
    feedback_distance: float
    locality_bin: str
    method: str
    estimated_selectivity: float
    qerror: float


def cdf_levels(boundaries: Sequence[float]) -> List[float]:
    buckets = max(len(boundaries) - 1, 1)
    return [idx / buckets for idx in range(len(boundaries))]


def cdf_value(boundaries: Sequence[float], value: float) -> float:
    return evaluate_piecewise_cdf(boundaries, cdf_levels(boundaries), float(value))


def predicate_endpoints(predicate: Dict[str, object]) -> List[float]:
    pred_type = str(predicate["predicate_type"])
    value = float(predicate["value"])
    if pred_type == "BETWEEN":
        upper = float(predicate.get("value_upper", value))
        lo, hi = sorted((value, upper))
        return [lo, hi]
    return [value]


def observation_endpoints(observation: Dict[str, object]) -> List[float]:
    pred_type = str(observation["predicate_type"])
    value = float(observation["value"])
    if pred_type == "BETWEEN":
        upper = float(observation.get("value_upper", value))
        lo, hi = sorted((value, upper))
        return [lo, hi]
    return [value]


def feedback_distance(
    fresh_boundaries: Sequence[float],
    predicate: Dict[str, object],
    observations: Sequence[Dict[str, object]],
) -> float:
    """Directed Hausdorff distance from future predicate endpoints to feedback.

    Distances are measured in fresh CDF/quantile space. For a BETWEEN predicate,
    both endpoints must be close to feedback boundaries for the predicate to be
    considered local.
    """
    feedback_ps: List[float] = []
    for observation in observations:
        for endpoint in observation_endpoints(observation):
            feedback_ps.append(cdf_value(fresh_boundaries, endpoint))
    if not feedback_ps:
        return 1.0

    pred_ps = [cdf_value(fresh_boundaries, endpoint) for endpoint in predicate_endpoints(predicate)]
    return max(min(abs(pred_p - fb_p) for fb_p in feedback_ps) for pred_p in pred_ps)


def locality_bin(distance: float, near_threshold: float, mid_threshold: float) -> str:
    if distance <= near_threshold:
        return f"near<= {near_threshold:.2f}"
    if distance <= mid_threshold:
        return f"mid<= {mid_threshold:.2f}"
    return f"far> {mid_threshold:.2f}"


def build_method_boundaries(
    sample,
    model: MlpHistogramModelV2,
    observations: Sequence[Dict[str, object]],
    num_buckets: int,
    max_observations: int,
) -> Dict[str, List[float]]:
    stale = boundaries_from_quantiles(sample.prior.quantile_values)
    fresh = boundaries_from_quantiles(sample.corrected_quantile_values or sample.prior.quantile_values)
    oasis = oasis_boundaries(sample, model, max_observations)

    return {
        "stale": stale,
        "proj_stale": isomer_boundaries(stale, observations, num_buckets),
        "oasis_no_proj": oasis,
        "proj_oasis": isomer_boundaries(oasis, observations, num_buckets),
        "proj_fresh": isomer_boundaries(fresh, observations, num_buckets),
        "fresh": fresh,
    }


def generate_threshold_predicates(
    fresh_boundaries: Sequence[float],
    rng: random.Random,
    count: int,
    min_true_selectivity: float,
) -> List[dict]:
    levels = cdf_levels(fresh_boundaries)
    predicates: List[dict] = []
    attempts = 0
    while len(predicates) < count and attempts < count * 40:
        attempts += 1
        pred_type = rng.choice(["<=", ">="])
        if pred_type == "<=":
            p = rng.uniform(min_true_selectivity, 1.0 - min_true_selectivity)
            value = inverse_piecewise_cdf(fresh_boundaries, levels, p)
        else:
            p = rng.uniform(min_true_selectivity, 1.0 - min_true_selectivity)
            value = inverse_piecewise_cdf(fresh_boundaries, levels, 1.0 - p)
        predicates.append({"predicate_type": pred_type, "value": value, "value_upper": None})
    return predicates


def summarize_method_rows(rows: Sequence[PredicateRow], group_fields: Sequence[str]) -> List[dict]:
    grouped: Dict[Tuple[object, ...], List[PredicateRow]] = defaultdict(list)
    for row in rows:
        key = tuple(getattr(row, field) for field in group_fields) + (row.method,)
        grouped[key].append(row)

    summary: List[dict] = []
    group_keys = sorted({key[:-1] for key in grouped})
    for group_key in group_keys:
        stale = grouped.get(group_key + ("stale",), [])
        isomer = grouped.get(group_key + ("proj_stale",), [])
        stale_qerr = geomean([row.qerror for row in stale])
        isomer_by_pred = {
            (row.q_mods, row.case_id, row.predicate_id): row.qerror
            for row in isomer
        }

        for method in METHOD_ORDER:
            method_rows = grouped.get(group_key + (method,), [])
            if not method_rows:
                continue
            qerrors = [row.qerror for row in method_rows]
            wins_vs_isomer = 0
            ties_vs_isomer = 0
            comparable = 0
            for row in method_rows:
                key = (row.q_mods, row.case_id, row.predicate_id)
                baseline = isomer_by_pred.get(key)
                if baseline is None:
                    continue
                comparable += 1
                if row.qerror < baseline - 1e-12:
                    wins_vs_isomer += 1
                elif abs(row.qerror - baseline) <= 1e-12:
                    ties_vs_isomer += 1

            item = {field: value for field, value in zip(group_fields, group_key)}
            item.update(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "n_predicates": len(method_rows),
                    "qerror_gm": geomean(qerrors),
                    "qerror_mean": sum(qerrors) / max(len(qerrors), 1),
                    "improvement_vs_stale_pct": pct_improvement(stale_qerr, geomean(qerrors)),
                    "win_vs_isomer_frac": wins_vs_isomer / max(comparable, 1),
                    "tie_vs_isomer_frac": ties_vs_isomer / max(comparable, 1),
                }
            )
            summary.append(item)
    return summary


def pivot_projection_summary(summary: Sequence[dict]) -> List[dict]:
    by_group: Dict[object, Dict[str, dict]] = defaultdict(dict)
    for row in summary:
        by_group[row["q_mods"]][row["method"]] = row

    result = []
    for q_mods in sorted(by_group, key=lambda value: 999 if value == "all" else int(value)):
        methods = by_group[q_mods]
        isomer = methods["proj_stale"]["qerror_gm"]
        full = methods["proj_oasis"]["qerror_gm"]
        result.append(
            {
                "q_mods": q_mods,
                "stale_qerr_gm": methods["stale"]["qerror_gm"],
                "proj_stale_isomer_qerr_gm": isomer,
                "oasis_no_proj_qerr_gm": methods["oasis_no_proj"]["qerror_gm"],
                "proj_oasis_qerr_gm": full,
                "proj_fresh_qerr_gm": methods["proj_fresh"]["qerror_gm"],
                "fresh_qerr_gm": methods["fresh"]["qerror_gm"],
                "proj_oasis_vs_isomer_pct": pct_improvement(isomer, full),
                "proj_oasis_win_vs_isomer_frac": methods["proj_oasis"]["win_vs_isomer_frac"],
            }
        )
    return result


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_projection_table(output_dir: Path, pivot_rows: Sequence[dict]) -> None:
    path = output_dir / "table_projection_initialization.tex"
    with path.open("w") as handle:
        handle.write("\\begin{table*}[t]\n")
        handle.write("  \\centering\n")
        handle.write("  \\small\n")
        handle.write("  \\caption{Projection-initialization ablation on held-out future predicates (in-distribution compound drift; geometric-mean selectivity Q-error). The projection operator and feedback constraints are identical across columns; only the marginal that initializes the projection changes. \\textbf{Proj(stale)} initializes from the stale histogram and is exactly ISOMER; \\textbf{Proj(OASIS)} initializes from the learned stage and is the full two-stage OASIS; \\textbf{Proj(fresh)} initializes from the post-drift marginal as a reference upper bound. Win frac is the fraction of held-out predicates on which the full OASIS beats ISOMER.}\n")
        handle.write("  \\label{tab:projection_initialization}\n")
        handle.write("  \\setlength{\\tabcolsep}{4pt}\n")
        handle.write("  \\resizebox{\\textwidth}{!}{%\n")
        handle.write("  \\begin{tabular}{c | rrrrrr | rr}\n")
        handle.write("    \\toprule\n")
        handle.write("    $q$ & Stale & Proj(stale) & OASIS-noProj & Proj(OASIS) & Proj(fresh) & Fresh & Proj(OASIS) vs ISOMER & Win frac \\\\\n")
        handle.write("    \\midrule\n")
        for row in pivot_rows:
            q_label = row["q_mods"]
            if q_label == "all":
                handle.write("    \\midrule\n")
                q_text = "\\textbf{All}"
            else:
                q_text = str(q_label)
            handle.write(
                f"    {q_text} & {row['stale_qerr_gm']:.3f} & "
                f"{row['proj_stale_isomer_qerr_gm']:.3f} & "
                f"{row['oasis_no_proj_qerr_gm']:.3f} & "
                f"{row['proj_oasis_qerr_gm']:.3f} & "
                f"{row['proj_fresh_qerr_gm']:.3f} & "
                f"{row['fresh_qerr_gm']:.3f} & "
                f"{row['proj_oasis_vs_isomer_pct']:+.1f}\\% & "
                f"{row['proj_oasis_win_vs_isomer_frac'] * 100:.1f}\\% \\\\\n"
            )
        handle.write("    \\bottomrule\n")
        handle.write("  \\end{tabular}%\n")
        handle.write("  }\n")
        handle.write("\\end{table*}\n")


def write_locality_table(output_dir: Path, locality_summary: Sequence[dict]) -> None:
    by_bin_method = {
        (row["locality_bin"], row["method"]): row
        for row in locality_summary
        if row["q_group"] == "all"
    }
    bins = sorted(
        [row["locality_bin"] for row in locality_summary if row["q_group"] == "all" and row["method"] == "stale"],
        key=locality_sort_key,
    )

    path = output_dir / "table_feedback_locality.tex"
    with path.open("w") as handle:
        handle.write("\\begin{table*}[t]\n")
        handle.write("  \\centering\n")
        handle.write("  \\small\n")
        handle.write("  \\caption{Feedback-locality diagnostic on the mixed predicate population. Held-out future predicates are binned by the directed Hausdorff distance from their endpoints to the nearest feedback endpoint in fresh-CDF space. Values are geometric-mean selectivity Q-error; ISOMER is projection from a stale start and OASIS is the full two-stage system. OASIS win is the per-predicate fraction on which the full OASIS beats ISOMER.}\n")
        handle.write("  \\label{tab:feedback_locality}\n")
        handle.write("  \\setlength{\\tabcolsep}{4pt}\n")
        handle.write("  \\resizebox{\\textwidth}{!}{%\n")
        handle.write("  \\begin{tabular}{l | r | rrrrr | rr}\n")
        handle.write("    \\toprule\n")
        handle.write("    Locality & Preds & Stale & ISOMER & OASIS-noProj & OASIS & Fresh & OASIS vs ISOMER & OASIS win \\\\\n")
        handle.write("    \\midrule\n")
        for bin_name in bins:
            stale = by_bin_method[(bin_name, "stale")]
            isomer = by_bin_method[(bin_name, "proj_stale")]
            raw = by_bin_method[(bin_name, "oasis_no_proj")]
            full = by_bin_method[(bin_name, "proj_oasis")]
            fresh = by_bin_method[(bin_name, "fresh")]
            handle.write(
                f"    {bin_name} & {stale['n_predicates']} & "
                f"{stale['qerror_gm']:.3f} & {isomer['qerror_gm']:.3f} & "
                f"{raw['qerror_gm']:.3f} & {full['qerror_gm']:.3f} & "
                f"{fresh['qerror_gm']:.3f} & "
                f"{pct_improvement(isomer['qerror_gm'], full['qerror_gm']):+.1f}\\% & "
                f"{full['win_vs_isomer_frac'] * 100:.1f}\\% \\\\\n"
            )
        handle.write("    \\bottomrule\n")
        handle.write("  \\end{tabular}%\n")
        handle.write("  }\n")
        handle.write("\\end{table*}\n")


def write_text_summary(output_dir: Path, pivot_rows: Sequence[dict], locality_summary: Sequence[dict]) -> None:
    lines = [
        "Projection initialization and feedback locality diagnostics",
        "=" * 64,
        "Metric: geometric-mean selectivity Q-error on held-out future predicates.",
        "",
        "Projection initialization ablation:",
        "q      Stale  ISOMER  OASIS-noProj  Proj(OASIS)  Fresh  ProjVsISOMER  Win",
        "-" * 88,
    ]
    for row in pivot_rows:
        q_label = str(row["q_mods"])
        lines.append(
            f"{q_label:<6s} {row['stale_qerr_gm']:6.3f}  "
            f"{row['proj_stale_isomer_qerr_gm']:6.3f}  "
            f"{row['oasis_no_proj_qerr_gm']:12.3f}  "
            f"{row['proj_oasis_qerr_gm']:11.3f}  "
            f"{row['fresh_qerr_gm']:5.3f}  "
            f"{row['proj_oasis_vs_isomer_pct']:+12.1f}%  "
            f"{row['proj_oasis_win_vs_isomer_frac'] * 100:5.1f}%"
        )

    lines.extend(
        [
            "",
            "Feedback locality bins:",
            "bin          preds  Stale  ISOMER  OASIS-noProj  OASIS  Fresh  OASISvsISOMER  Win",
            "-" * 92,
        ]
    )
    by_bin_method = {
        (row["locality_bin"], row["method"]): row
        for row in locality_summary
        if row["q_group"] == "all"
    }
    bins = sorted(
        [row["locality_bin"] for row in locality_summary if row["q_group"] == "all" and row["method"] == "stale"],
        key=locality_sort_key,
    )
    for bin_name in bins:
        stale = by_bin_method[(bin_name, "stale")]
        isomer = by_bin_method[(bin_name, "proj_stale")]
        raw = by_bin_method[(bin_name, "oasis_no_proj")]
        full = by_bin_method[(bin_name, "proj_oasis")]
        fresh = by_bin_method[(bin_name, "fresh")]
        lines.append(
            f"{bin_name:<12s} {stale['n_predicates']:5d}  "
            f"{stale['qerror_gm']:6.3f}  {isomer['qerror_gm']:6.3f}  "
            f"{raw['qerror_gm']:12.3f}  {full['qerror_gm']:5.3f}  "
            f"{fresh['qerror_gm']:5.3f}  "
            f"{pct_improvement(isomer['qerror_gm'], full['qerror_gm']):+13.1f}%  "
            f"{full['win_vs_isomer_frac'] * 100:5.1f}%"
        )

    text = "\n".join(lines)
    (output_dir / "summary.txt").write_text(text + "\n")
    print(text)


def locality_sort_key(name: str) -> int:
    if name.startswith("near"):
        return 0
    if name.startswith("mid"):
        return 1
    return 2


def run_experiment(args: argparse.Namespace) -> None:
    model = MlpHistogramModelV2.load(str(args.model_path))
    sample_paths = iter_sample_paths(args.data_root, args.q_values, args.max_cases_per_q)
    if not sample_paths:
        raise FileNotFoundError(f"No samples found under {args.data_root}")

    rows: List[PredicateRow] = []
    for sample_index, (q_mods, path) in enumerate(sample_paths):
        sample = load_feedback_sample(str(path))
        observations = observations_to_dicts(sample)
        method_boundaries = build_method_boundaries(
            sample,
            model=model,
            observations=observations,
            num_buckets=args.num_buckets,
            max_observations=args.max_observations,
        )
        fresh = method_boundaries["fresh"]
        rng = random.Random(args.seed + q_mods * 100_000 + sample_index)
        if args.predicate_mode == "threshold":
            predicates = generate_threshold_predicates(
                fresh,
                rng=rng,
                count=args.predicates_per_case,
                min_true_selectivity=args.min_true_selectivity,
            )
        else:
            predicates = generate_predicates(
                fresh,
                rng=rng,
                count=args.predicates_per_case,
                min_true_selectivity=args.min_true_selectivity,
            )

        for predicate_id, predicate in enumerate(predicates):
            true_sel = estimate_selectivity(fresh, predicate)
            distance = feedback_distance(fresh, predicate, observations)
            bin_name = locality_bin(distance, args.near_threshold, args.mid_threshold)
            for method in METHOD_ORDER:
                estimated = estimate_selectivity(method_boundaries[method], predicate)
                rows.append(
                    PredicateRow(
                        q_mods=q_mods,
                        case_id=path.stem,
                        predicate_id=predicate_id,
                        predicate_type=str(predicate["predicate_type"]),
                        true_selectivity=true_sel,
                        feedback_distance=distance,
                        locality_bin=bin_name,
                        method=method,
                        estimated_selectivity=estimated,
                        qerror=qerr(estimated, true_sel),
                    )
                )

        if (sample_index + 1) % 50 == 0:
            print(f"Processed {sample_index + 1}/{len(sample_paths)} samples")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    predicate_rows = [asdict(row) for row in rows]
    if args.write_rows:
        write_csv(output_dir / "predicate_rows.csv", predicate_rows)
        write_json(output_dir / "predicate_rows.json", predicate_rows)

    projection_summary = summarize_method_rows(rows, ["q_mods"])
    all_projection_summary = summarize_method_rows(
        [
            PredicateRow(
                q_mods=-1,
                case_id=f"q{row.q_mods}_{row.case_id}",
                predicate_id=row.predicate_id,
                predicate_type=row.predicate_type,
                true_selectivity=row.true_selectivity,
                feedback_distance=row.feedback_distance,
                locality_bin=row.locality_bin,
                method=row.method,
                estimated_selectivity=row.estimated_selectivity,
                qerror=row.qerror,
            )
            for row in rows
        ],
        ["q_mods"],
    )
    for row in all_projection_summary:
        row["q_mods"] = "all"
    projection_summary.extend(all_projection_summary)
    projection_pivot = pivot_projection_summary(projection_summary)

    locality_rows: List[PredicateRow] = []
    for row in rows:
        copied = PredicateRow(**asdict(row))
        setattr(copied, "q_group", "all")
        locality_rows.append(copied)
        copied_q = PredicateRow(**asdict(row))
        setattr(copied_q, "q_group", f"q={row.q_mods}")
        locality_rows.append(copied_q)

    # summarize_method_rows uses dataclass attributes dynamically, so q_group is
    # attached above instead of being part of the persisted predicate row schema.
    locality_summary = summarize_method_rows(locality_rows, ["q_group", "locality_bin"])

    write_json(output_dir / "projection_summary.json", projection_summary)
    write_csv(output_dir / "projection_summary.csv", projection_summary)
    write_json(output_dir / "projection_pivot_summary.json", projection_pivot)
    write_csv(output_dir / "projection_pivot_summary.csv", projection_pivot)
    write_json(output_dir / "locality_summary.json", locality_summary)
    write_csv(output_dir / "locality_summary.csv", locality_summary)
    write_projection_table(output_dir, projection_pivot)
    write_locality_table(output_dir, locality_summary)
    write_text_summary(output_dir, projection_pivot, locality_summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Projection initialization and feedback-locality diagnostics")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_REPO_DIR / "experiments" / "results" / "synthetic_paper_suite_rerun_20260529" / "compound_data",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=_REPO_DIR / "experiments" / "results" / "synthetic_paper_suite_rerun_20260529" / "models" / "oasis_k16.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_DIR / "experiments" / "results" / "projection_locality_20260531",
    )
    parser.add_argument("--q-values", type=int, nargs="+", default=[1, 3, 5, 10, 15, 20, 25, 30])
    parser.add_argument("--max-cases-per-q", type=int, default=128)
    parser.add_argument("--predicates-per-case", type=int, default=64)
    parser.add_argument("--num-buckets", type=int, default=10)
    parser.add_argument("--max-observations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-true-selectivity", type=float, default=1e-4)
    parser.add_argument("--near-threshold", type=float, default=0.05)
    parser.add_argument("--mid-threshold", type=float, default=0.15)
    parser.add_argument("--predicate-mode", choices=["mixed", "threshold"], default="mixed")
    parser.add_argument("--write-rows", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
