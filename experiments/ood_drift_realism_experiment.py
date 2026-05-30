#!/usr/bin/env python3
"""Out-of-distribution drift realism suite for OASIS.

The main OASIS checkpoint is trained on compound synthetic drift. This
experiment evaluates the same checkpoint on deployment-inspired drift families
that are not part of that training generator: batch loads, range shifts, skew
evolution, outlier bursts, multimodal drift, and seasonal/mixed drift.
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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_PIPELINE_DIR = _REPO_DIR / "cdf_kll_ml_pipeline"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from extended_drift_generators import DriftPattern, ExtendedMemoryTable
from histogram_math import clamp01, evaluate_piecewise_cdf
from histogram_types import DEFAULT_QUANTILE_LEVELS, FeedbackObservation, KllFeedbackSample, KllPrior
from mlp_histogram_model_v2 import MlpHistogramModelV2
from optimizer_decision_proxy_experiment import (
    boundaries_from_quantiles,
    choose_hybrid,
    estimate_selectivity,
    feedback_residual,
    isomer_boundaries,
    oasis_boundaries,
    observations_to_dicts,
)


METHOD_ORDER = ["stale", "isomer", "oasis", "oasis_projected", "hybrid", "fresh"]
PATTERN_ORDER = [
    DriftPattern.BATCH_LOAD,
    DriftPattern.RANGE_SHIFT,
    DriftPattern.SKEW_EVOLUTION,
    DriftPattern.OUTLIER_BURST,
    DriftPattern.MULTI_MODAL,
    DriftPattern.SEASONAL,
]


@dataclass
class OODRow:
    pattern: str
    case_id: int
    method: str
    qerror: float
    selectivity_mae: float
    quantile_mae: float
    feedback_residual: float
    beats_stale_qerr: bool


def method_label(method: str) -> str:
    return {
        "stale": "Stale",
        "isomer": "ISOMER",
        "oasis": "OASIS",
        "oasis_projected": "OASIS-Proj",
        "hybrid": "Hybrid",
        "fresh": "Fresh",
    }[method]


def pattern_label(pattern: str) -> str:
    return {
        "batch_load": "Batch load",
        "range_shift": "Range shift",
        "skew_evol": "Skew evolution",
        "outlier": "Outlier burst",
        "multimodal": "Multimodal",
        "seasonal": "Seasonal/mixed",
    }[pattern]


def geomean(values: Sequence[float]) -> float:
    vals = [max(float(value), 1e-12) for value in values]
    return math.exp(sum(math.log(value) for value in vals) / max(len(vals), 1))


def mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def cdf_levels(boundaries: Sequence[float]) -> List[float]:
    return [idx / (len(boundaries) - 1) for idx in range(len(boundaries))]


def cdf(boundaries: Sequence[float], value: float) -> float:
    return evaluate_piecewise_cdf(list(boundaries), cdf_levels(boundaries), value)


def q_error(pred: Sequence[float], truth: Sequence[float], points: Sequence[float], eps: float = 1e-6) -> float:
    errors = []
    for point in points:
        est = max(cdf(pred, point), eps)
        act = max(cdf(truth, point), eps)
        errors.append(max(est / act, act / est))
    return geomean(errors)


def selectivity_mae(pred: Sequence[float], truth: Sequence[float], points: Sequence[float]) -> float:
    return mean([abs(cdf(pred, point) - cdf(truth, point)) for point in points])


def quantile_mae(pred: Sequence[float], truth: Sequence[float]) -> float:
    return mean([abs(float(a) - float(b)) for a, b in zip(pred[1:-1], truth[1:-1])])


def quantile_boundaries(table: ExtendedMemoryTable, num_buckets: int) -> List[float]:
    levels = [idx / num_buckets for idx in range(1, num_buckets)]
    quantiles = table.get_quantiles(levels)
    return [0.0] + [clamp01(value) for value in quantiles] + [1.0]


def initial_data(rng: random.Random, size: int) -> Tuple[List[float], int]:
    mode = rng.choice(["gaussian_mixture", "uniform", "bimodal", "triangular"])
    data: List[float] = []
    if mode == "gaussian_mixture":
        centers = [rng.uniform(0.15, 0.85) for _ in range(rng.randint(2, 4))]
        for _ in range(size):
            data.append(clamp01(rng.normalvariate(rng.choice(centers), 0.11)))
    elif mode == "uniform":
        data = [rng.uniform(0.0, 1.0) for _ in range(size)]
    elif mode == "bimodal":
        for _ in range(size):
            center = 0.25 if rng.random() < 0.5 else 0.75
            data.append(clamp01(rng.normalvariate(center, 0.08)))
    else:
        data = [rng.triangular(0.0, 0.5, 1.0) for _ in range(size)]
    null_count = int(size * rng.uniform(0.01, 0.06))
    return data, null_count


def pattern_kwargs(pattern: DriftPattern, rng: random.Random) -> dict:
    if pattern == DriftPattern.BATCH_LOAD:
        return {
            "batch_size_range": (80, 300),
            "target_region": rng.uniform(0.65, 0.95),
        }
    if pattern == DriftPattern.RANGE_SHIFT:
        return {
            "shift_direction": rng.choice(["expand", "shift"]),
            "shift_magnitude": rng.uniform(0.08, 0.22),
        }
    if pattern == DriftPattern.SKEW_EVOLUTION:
        return {
            "initial_skew": rng.uniform(0.2, 0.55),
            "target_skew": rng.uniform(0.72, 0.93),
        }
    if pattern == DriftPattern.OUTLIER_BURST:
        return {
            "outlier_ratio": rng.uniform(0.10, 0.25),
            "burst_frequency": rng.choice([2, 3, 4]),
        }
    if pattern == DriftPattern.MULTI_MODAL:
        return {
            "n_modes": rng.choice([2, 3, 4]),
            "mode_separation": rng.uniform(0.18, 0.28),
        }
    if pattern == DriftPattern.SEASONAL:
        return {
            "period": rng.choice([3, 4, 6]),
            "amplitude": rng.uniform(0.12, 0.28),
        }
    return {}


def draw_observation(
    rng: random.Random,
    table: ExtendedMemoryTable,
    stale_boundaries: Sequence[float],
    timestamp: datetime,
) -> FeedbackObservation:
    pred_type = rng.choices(["<=", ">=", "BETWEEN", "="], weights=[0.33, 0.33, 0.26, 0.08], k=1)[0]
    value_upper: Optional[float] = None
    if table.data:
        sorted_data = sorted(table.data)
        if pred_type == "BETWEEN":
            lo_idx = rng.randint(0, max(0, len(sorted_data) - 2))
            hi_idx = rng.randint(lo_idx + 1, len(sorted_data) - 1)
            value = sorted_data[lo_idx]
            value_upper = sorted_data[hi_idx]
        else:
            value = sorted_data[rng.randint(0, len(sorted_data) - 1)]
    else:
        value = rng.random()
        if pred_type == "BETWEEN":
            value_upper = min(1.0, value + rng.uniform(0.02, 0.25))

    predicate = {"predicate_type": pred_type, "value": value, "value_upper": value_upper}
    return FeedbackObservation(
        predicate_type=pred_type,
        value=float(value),
        value_upper=value_upper,
        actual_selectivity=table.query_conditional_sel(pred_type, float(value), value_upper),
        estimated_selectivity=estimate_selectivity(stale_boundaries, predicate),
        timestamp=timestamp,
    )


def build_case(
    pattern: DriftPattern,
    case_id: int,
    args: argparse.Namespace,
) -> KllFeedbackSample:
    seed = args.seed + case_id * 7919 + PATTERN_ORDER.index(pattern) * 1_000_003
    rng = random.Random(seed)
    np.random.seed(seed % (2**32 - 1))

    data, null_count = initial_data(rng, args.initial_rows)
    table = ExtendedMemoryTable(data, null_count)
    prior_null = table.get_null_fraction()
    stale_boundaries = quantile_boundaries(table, args.num_buckets)
    kwargs = pattern_kwargs(pattern, rng)

    base_time = datetime(2026, 5, 29, tzinfo=timezone.utc)
    observations: List[FeedbackObservation] = []
    for obs_index in range(args.observations):
        table.apply_drift_by_pattern(pattern, rng, args.q_mods, **kwargs)
        observations.append(
            draw_observation(
                rng,
                table,
                stale_boundaries=stale_boundaries,
                timestamp=base_time + timedelta(minutes=obs_index),
            )
        )

    fresh_boundaries = quantile_boundaries(table, args.num_buckets)
    return KllFeedbackSample(
        prior=KllPrior(
            min_value=0.0,
            max_value=1.0,
            null_fraction=prior_null,
            quantile_levels=list(DEFAULT_QUANTILE_LEVELS),
            quantile_values=stale_boundaries[1:-1],
            value_type="double",
        ),
        observations=observations,
        corrected_quantile_values=fresh_boundaries[1:-1],
        source_path=f"ood:{pattern.value}:{case_id}",
    )


def method_boundaries(sample: KllFeedbackSample, model: MlpHistogramModelV2, num_buckets: int, model_window: int) -> Tuple[Dict[str, List[float]], str]:
    observations = observations_to_dicts(sample)
    stale = boundaries_from_quantiles(sample.prior.quantile_values)
    fresh = boundaries_from_quantiles(sample.corrected_quantile_values or sample.prior.quantile_values)
    isomer = isomer_boundaries(stale, observations, num_buckets)
    oasis = oasis_boundaries(sample, model, model_window)
    oasis_projected = isomer_boundaries(oasis, observations, num_buckets)
    boundaries = {
        "stale": stale,
        "isomer": isomer,
        "oasis": oasis,
        "oasis_projected": oasis_projected,
        "fresh": fresh,
    }
    hybrid_choice, hybrid = choose_hybrid(boundaries, observations)
    boundaries["hybrid"] = hybrid
    return boundaries, hybrid_choice


def metric_points(seed: int, count: int) -> List[float]:
    rng = random.Random(seed)
    return sorted(rng.uniform(0.005, 0.995) for _ in range(count))


def aggregate(rows: Sequence[OODRow], hybrid_choices: Dict[str, Counter]) -> List[dict]:
    grouped: Dict[Tuple[str, str], List[OODRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.pattern, row.method)].append(row)

    results = []
    for pattern in [pattern.value for pattern in PATTERN_ORDER]:
        stale_rows = grouped[(pattern, "stale")]
        stale_qerr = geomean([row.qerror for row in stale_rows])
        for method in METHOD_ORDER:
            method_rows = grouped[(pattern, method)]
            qerr_gm = geomean([row.qerror for row in method_rows])
            results.append({
                "pattern": pattern,
                "method": method,
                "n_cases": len(method_rows),
                "qerror_gm": qerr_gm,
                "qerror_improvement_pct": (stale_qerr - qerr_gm) / max(stale_qerr, 1e-12) * 100,
                "selectivity_mae_mean": mean([row.selectivity_mae for row in method_rows]),
                "quantile_mae_mean": mean([row.quantile_mae for row in method_rows]),
                "feedback_residual_mean": mean([row.feedback_residual for row in method_rows if math.isfinite(row.feedback_residual)]),
                "beats_stale_frac": mean([1.0 if row.beats_stale_qerr else 0.0 for row in method_rows]),
                "hybrid_isomer_frac": hybrid_choices[pattern]["isomer"] / max(sum(hybrid_choices[pattern].values()), 1),
                "hybrid_oasis_projected_frac": hybrid_choices[pattern]["oasis_projected"] / max(sum(hybrid_choices[pattern].values()), 1),
                "hybrid_oasis_frac": hybrid_choices[pattern]["oasis"] / max(sum(hybrid_choices[pattern].values()), 1),
                "hybrid_stale_frac": hybrid_choices[pattern]["stale"] / max(sum(hybrid_choices[pattern].values()), 1),
            })
    return results


def write_table(output_dir: Path, summary: Sequence[dict]) -> None:
    by_key = {(row["pattern"], row["method"]): row for row in summary}
    with (output_dir / "table_ood_drift_realism.tex").open("w") as handle:
        handle.write("\\begin{table*}[t]\n")
        handle.write("  \\centering\n")
        handle.write("  \\small\n")
        handle.write("  \\caption{Out-of-distribution drift realism suite. OASIS is trained only on compound drift and evaluated on deployment-inspired drift families. Values are geometric mean selectivity Q-error over held-out cases; Hybrid uses only feedback-window residuals for selection.}\n")
        handle.write("  \\label{tab:ood_drift_realism}\n")
        handle.write("  \\setlength{\\tabcolsep}{4pt}\n")
        handle.write("  \\resizebox{\\textwidth}{!}{%\n")
        handle.write("  \\begin{tabular}{lrrrrrrr}\n")
        handle.write("    \\toprule\n")
        handle.write("    Drift family & Stale & ISOMER & OASIS & OASIS-Proj & Hybrid & Fresh & Hybrid choice \\\\\n")
        handle.write("    \\midrule\n")
        for pattern in [pattern.value for pattern in PATTERN_ORDER]:
            stale = by_key[(pattern, "stale")]
            isomer = by_key[(pattern, "isomer")]
            oasis = by_key[(pattern, "oasis")]
            projected = by_key[(pattern, "oasis_projected")]
            hybrid = by_key[(pattern, "hybrid")]
            fresh = by_key[(pattern, "fresh")]
            method_values = {
                "isomer": isomer["qerror_gm"],
                "oasis": oasis["qerror_gm"],
                "oasis_projected": projected["qerror_gm"],
                "hybrid": hybrid["qerror_gm"],
            }
            best_method = min(method_values, key=method_values.get)

            def fmt(method: str, value: float) -> str:
                text = f"{value:.3f}"
                return f"\\textbf{{{text}}}" if method == best_method else text

            choice_text = (
                f"I {hybrid['hybrid_isomer_frac'] * 100:.0f}\\%, "
                f"P {hybrid['hybrid_oasis_projected_frac'] * 100:.0f}\\%, "
                f"O {hybrid['hybrid_oasis_frac'] * 100:.0f}\\%"
            )
            handle.write(
                f"    {pattern_label(pattern)} & {stale['qerror_gm']:.3f} & "
                f"{fmt('isomer', isomer['qerror_gm'])} & {fmt('oasis', oasis['qerror_gm'])} & "
                f"{fmt('oasis_projected', projected['qerror_gm'])} & {fmt('hybrid', hybrid['qerror_gm'])} & "
                f"{fresh['qerror_gm']:.3f} & {choice_text} \\\\\n"
            )
        handle.write("    \\bottomrule\n")
        handle.write("  \\end{tabular}%\n")
        handle.write("  }\n")
        handle.write("\\end{table*}\n")


def write_outputs(output_dir: Path, rows: Sequence[OODRow], summary: Sequence[dict], hybrid_choices: Dict[str, Counter], write_rows: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(list(summary), handle, indent=2)
    with (output_dir / "hybrid_choices.json").open("w") as handle:
        json.dump({pattern: dict(counter) for pattern, counter in hybrid_choices.items()}, handle, indent=2)
    if write_rows:
        with (output_dir / "case_rows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
    write_table(output_dir, summary)
    write_summary_text(output_dir, summary, hybrid_choices)


def write_summary_text(output_dir: Path, summary: Sequence[dict], hybrid_choices: Dict[str, Counter]) -> None:
    by_key = {(row["pattern"], row["method"]): row for row in summary}
    lines = [
        "OOD drift realism suite",
        "=" * 32,
        "OASIS checkpoint is trained on compound drift only.",
        "",
        "Pattern        Stale  ISOMER  OASIS  Proj  Hybrid  Fresh",
        "-" * 68,
    ]
    for pattern in [pattern.value for pattern in PATTERN_ORDER]:
        row = lambda method: by_key[(pattern, method)]["qerror_gm"]
        lines.append(
            f"{pattern:<14s} {row('stale'):5.3f}  {row('isomer'):6.3f}  "
            f"{row('oasis'):5.3f}  {row('oasis_projected'):5.3f}  "
            f"{row('hybrid'):6.3f}  {row('fresh'):5.3f}"
        )
        total = sum(hybrid_choices[pattern].values())
        if total:
            lines.append(
                "  Hybrid choices: "
                + ", ".join(f"{method}={count / total * 100:.1f}%" for method, count in sorted(hybrid_choices[pattern].items()))
            )
    text = "\n".join(lines)
    (output_dir / "summary.txt").write_text(text + "\n")
    print(text)


def run_experiment(args: argparse.Namespace) -> None:
    model = MlpHistogramModelV2.load(str(args.model_path))
    model_window = int(getattr(model, "max_observations", 16))
    patterns = [DriftPattern(value) for value in args.patterns]

    rows: List[OODRow] = []
    hybrid_choices: Dict[str, Counter] = {pattern.value: Counter() for pattern in patterns}
    total = len(patterns) * args.cases_per_pattern
    done = 0
    for pattern in patterns:
        for case_id in range(args.cases_per_pattern):
            sample = build_case(pattern, case_id, args)
            true_boundaries = boundaries_from_quantiles(sample.corrected_quantile_values or sample.prior.quantile_values)
            boundaries, hybrid_choice = method_boundaries(sample, model, args.num_buckets, model_window)
            hybrid_choices[pattern.value][hybrid_choice] += 1
            points = metric_points(args.seed + PATTERN_ORDER.index(pattern) * 100_000 + case_id, args.metric_points)
            observations = observations_to_dicts(sample)
            stale_qerr = q_error(boundaries["stale"], true_boundaries, points)
            for method in METHOD_ORDER:
                method_qerr = q_error(boundaries[method], true_boundaries, points)
                rows.append(OODRow(
                    pattern=pattern.value,
                    case_id=case_id,
                    method=method,
                    qerror=method_qerr,
                    selectivity_mae=selectivity_mae(boundaries[method], true_boundaries, points),
                    quantile_mae=quantile_mae(boundaries[method], true_boundaries),
                    feedback_residual=feedback_residual(boundaries[method], observations),
                    beats_stale_qerr=method_qerr < stale_qerr,
                ))
            done += 1
            if done % 50 == 0:
                print(f"Processed {done}/{total} cases")

    summary = aggregate(rows, hybrid_choices)
    write_outputs(args.output_dir, rows, summary, hybrid_choices, write_rows=args.write_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOD drift realism suite")
    parser.add_argument("--model-path", type=Path,
                        default=_REPO_DIR / "experiments" / "results" / "synthetic_paper_suite_rerun_20260529" / "models" / "oasis_k16.json")
    parser.add_argument("--output-dir", type=Path,
                        default=_REPO_DIR / "experiments" / "results" / "ood_drift_realism_20260529")
    parser.add_argument("--patterns", nargs="+",
                        default=[pattern.value for pattern in PATTERN_ORDER],
                        choices=[pattern.value for pattern in PATTERN_ORDER])
    parser.add_argument("--cases-per-pattern", type=int, default=128)
    parser.add_argument("--initial-rows", type=int, default=5000)
    parser.add_argument("--q-mods", type=int, default=10)
    parser.add_argument("--observations", type=int, default=16)
    parser.add_argument("--num-buckets", type=int, default=10)
    parser.add_argument("--metric-points", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--write-rows", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
