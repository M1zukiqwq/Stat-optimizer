#!/usr/bin/env python3
"""Tiny OASIS efficiency smoke test.

This script intentionally reuses cached paper-suite data and model checkpoints.
It does not retrain OASIS or rerun any full experiment. The smoke compares the
deployed full OASIS projection against lower-IPF-iteration projection variants,
while reporting both accuracy and feedback-consistency residuals.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_PIPELINE_DIR = _REPO_DIR / "cdf_kll_ml_pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from histogram_math import clamp01, evaluate_piecewise_cdf, project_monotonic
from json_histogram_parser import load_feedback_sample
from mlp_histogram_model_v2 import MlpHistogramModelV2
from modern_baselines import correct_isomer
from tensorizer import tensorize_sample


def boundaries_from_quantiles(quantiles: Sequence[float]) -> List[float]:
    values = [clamp01(float(value)) for value in quantiles]
    return [0.0] + project_monotonic(values) + [1.0]


def cdf_levels(boundaries: Sequence[float]) -> List[float]:
    buckets = max(len(boundaries) - 1, 1)
    return [index / buckets for index in range(len(boundaries))]


def estimate_selectivity(boundaries: Sequence[float], predicate: Dict[str, object]) -> float:
    levels = cdf_levels(boundaries)
    pred_type = str(predicate["predicate_type"])
    value = float(predicate["value"])

    if pred_type in {"<", "<="}:
        return max(evaluate_piecewise_cdf(boundaries, levels, value), 1e-9)
    if pred_type in {">", ">="}:
        return max(1.0 - evaluate_piecewise_cdf(boundaries, levels, value), 1e-9)
    if pred_type == "BETWEEN":
        upper = float(predicate.get("value_upper", value))
        lo, hi = sorted((value, upper))
        return max(
            evaluate_piecewise_cdf(boundaries, levels, hi)
            - evaluate_piecewise_cdf(boundaries, levels, lo),
            1e-9,
        )

    width = 0.01
    return max(
        evaluate_piecewise_cdf(boundaries, levels, min(1.0, value + width))
        - evaluate_piecewise_cdf(boundaries, levels, max(0.0, value - width)),
        1e-9,
    )


def qerr(estimate: float, truth: float) -> float:
    estimate = max(float(estimate), 1e-9)
    truth = max(float(truth), 1e-9)
    return max(estimate / truth, truth / estimate)


def geomean(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    return math.exp(sum(math.log(max(float(value), 1e-12)) for value in values) / len(values))


def pct_improvement(base: float, value: float) -> float:
    return (base - value) / max(base, 1e-12) * 100.0


def observations_to_dicts(sample, max_observations: int) -> List[dict]:
    selected = sample.observations[-max_observations:]
    result: List[dict] = []
    for obs in selected:
        item = {
            "predicate_type": obs.predicate_type,
            "value": obs.value,
            "estimated_sel": obs.estimated_selectivity,
            "actual_sel": obs.actual_selectivity,
        }
        if obs.value_upper is not None:
            item["value_upper"] = obs.value_upper
        result.append(item)
    return result


def feedback_residuals(boundaries: Sequence[float], observations: Sequence[dict]) -> Tuple[float, float]:
    if not observations:
        return float("inf"), float("inf")
    errors = []
    for obs in observations:
        pred = {
            "predicate_type": obs["predicate_type"],
            "value": obs["value"],
            "value_upper": obs.get("value_upper"),
        }
        errors.append(abs(estimate_selectivity(boundaries, pred) - float(obs["actual_sel"])))
    return sum(errors) / len(errors), max(errors)


def project_boundaries(
    prior_boundaries: Sequence[float],
    observations: Sequence[dict],
    num_buckets: int,
    max_iter: int,
    tol: float,
) -> List[float]:
    corrected = correct_isomer(
        float(prior_boundaries[0]),
        float(prior_boundaries[-1]),
        list(prior_boundaries[1:-1]),
        list(observations),
        num_buckets=num_buckets,
        max_iter=max_iter,
        tol=tol,
    )
    return boundaries_from_quantiles(corrected)


def sample_paths(data_root: Path, q_values: Sequence[int], max_cases_per_q: int) -> List[Tuple[int, Path]]:
    paths = []
    for q_mods in q_values:
        q_dir = data_root / f"test_q{q_mods}"
        q_paths = sorted(q_dir.glob("*.json"))
        if max_cases_per_q > 0:
            q_paths = q_paths[:max_cases_per_q]
        paths.extend((q_mods, path) for path in q_paths)
    return paths


def metric_points(seed: int, count: int) -> List[float]:
    rng = random.Random(seed)
    return [rng.uniform(0.01, 0.99) for _ in range(count)]


def write_outputs(output_dir: Path, rows: Sequence[dict], summary: Sequence[dict], verdict: Sequence[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rows.json").open("w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(list(summary), handle, indent=2)
    with (output_dir / "verdict.json").open("w", encoding="utf-8") as handle:
        json.dump(list(verdict), handle, indent=2)

    if rows:
        with (output_dir / "rows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if summary:
        with (output_dir / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    lines = [
        "OASIS efficiency smoke",
        "=" * 48,
        "Pass rule: a fast projection variant should reduce total per-case time",
        "by the requested threshold while keeping Q-error and feedback residual",
        "within the requested relative tolerances against baseline full OASIS.",
        "",
        "Method                 n   QE(gm)  FeedMean  FeedMax  Stage1ms  Projms  Totalms  QEImp",
        "-" * 92,
    ]
    stale_qerr = next((row["qerror_gm"] for row in summary if row["method"] == "stale"), None)
    for row in summary:
        qe_imp = pct_improvement(stale_qerr, row["qerror_gm"]) if stale_qerr is not None else 0.0
        lines.append(
            f"{row['method']:<20s} {row['n_cases']:3d}  "
            f"{row['qerror_gm']:6.3f}  {row['feedback_residual_mean']:8.5f}  "
            f"{row['feedback_residual_max']:7.5f}  {row['stage1_ms_mean']:8.3f}  "
            f"{row['projection_ms_mean']:6.3f}  {row['total_ms_mean']:7.3f}  {qe_imp:6.1f}%"
        )
    lines.append("")
    lines.append("Verdict:")
    for row in verdict:
        lines.append(
            f"  {row['method']}: pass={row['pass']} "
            f"speedup={row['runtime_reduction_pct']:.1f}% "
            f"qerr_delta={row['qerror_relative_delta_pct']:+.2f}% "
            f"residual_delta={row['feedback_residual_relative_delta_pct']:+.2f}%"
        )

    text = "\n".join(lines)
    (output_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


def run_smoke(args: argparse.Namespace) -> None:
    model = MlpHistogramModelV2.load(str(args.model_path))
    paths = sample_paths(args.data_root, args.q_values, args.max_cases_per_q)
    if not paths:
        raise FileNotFoundError(f"No samples found under {args.data_root}")

    variant_iters = sorted(set(args.fast_projection_iters))
    rows: List[dict] = []

    for sample_index, (q_mods, path) in enumerate(paths):
        sample = load_feedback_sample(str(path))
        if sample.corrected_quantile_values is None:
            continue

        observations = observations_to_dicts(sample, args.max_observations)
        stale = boundaries_from_quantiles(sample.prior.quantile_values)
        fresh = boundaries_from_quantiles(sample.corrected_quantile_values)

        start_stage1 = time.perf_counter()
        record = tensorize_sample(sample, max_observations=args.max_observations, teacher_fn=None, use_time_decay=False)
        oasis_prediction = model.predict([record.feature_tensor])[0]
        oasis = boundaries_from_quantiles(oasis_prediction)
        stage1_ms = (time.perf_counter() - start_stage1) * 1000.0

        method_boundaries: Dict[str, List[float]] = {
            "stale": stale,
            "fresh": fresh,
            "oasis_no_proj": oasis,
        }
        projection_times = {"stale": 0.0, "fresh": 0.0, "oasis_no_proj": 0.0}
        stage1_times = {"stale": 0.0, "fresh": 0.0, "oasis_no_proj": stage1_ms}

        start = time.perf_counter()
        method_boundaries["isomer"] = project_boundaries(stale, observations, args.num_buckets, args.baseline_projection_iters, args.projection_tol)
        projection_times["isomer"] = (time.perf_counter() - start) * 1000.0
        stage1_times["isomer"] = 0.0

        start = time.perf_counter()
        method_boundaries["oasis_full"] = project_boundaries(oasis, observations, args.num_buckets, args.baseline_projection_iters, args.projection_tol)
        projection_times["oasis_full"] = (time.perf_counter() - start) * 1000.0
        stage1_times["oasis_full"] = stage1_ms

        for iter_count in variant_iters:
            method = f"oasis_fast_iter_{iter_count}"
            start = time.perf_counter()
            method_boundaries[method] = project_boundaries(oasis, observations, args.num_buckets, iter_count, args.projection_tol)
            projection_times[method] = (time.perf_counter() - start) * 1000.0
            stage1_times[method] = stage1_ms

        points = metric_points(args.seed + q_mods * 10_000 + sample_index, args.q_points)
        for method, boundaries in method_boundaries.items():
            qerrors = [
                qerr(estimate_selectivity(boundaries, {"predicate_type": "<=", "value": point}), estimate_selectivity(fresh, {"predicate_type": "<=", "value": point}))
                for point in points
            ]
            residual_mean, residual_max = feedback_residuals(boundaries, observations)
            rows.append(
                {
                    "q_mods": q_mods,
                    "case_id": path.stem,
                    "method": method,
                    "qerror": geomean(qerrors),
                    "feedback_residual_mean": residual_mean,
                    "feedback_residual_max": residual_max,
                    "stage1_ms": stage1_times[method],
                    "projection_ms": projection_times[method],
                    "total_ms": stage1_times[method] + projection_times[method],
                }
            )

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    method_order = ["stale", "isomer", "oasis_no_proj", "oasis_full"] + [f"oasis_fast_iter_{value}" for value in variant_iters] + ["fresh"]
    summary = []
    for method in method_order:
        method_rows = grouped.get(method, [])
        if not method_rows:
            continue
        summary.append(
            {
                "method": method,
                "n_cases": len(method_rows),
                "qerror_gm": geomean([row["qerror"] for row in method_rows]),
                "feedback_residual_mean": sum(row["feedback_residual_mean"] for row in method_rows) / len(method_rows),
                "feedback_residual_max": max(row["feedback_residual_max"] for row in method_rows),
                "stage1_ms_mean": sum(row["stage1_ms"] for row in method_rows) / len(method_rows),
                "projection_ms_mean": sum(row["projection_ms"] for row in method_rows) / len(method_rows),
                "total_ms_mean": sum(row["total_ms"] for row in method_rows) / len(method_rows),
            }
        )

    by_method = {row["method"]: row for row in summary}
    baseline = by_method["oasis_full"]
    verdict = []
    for iter_count in variant_iters:
        method = f"oasis_fast_iter_{iter_count}"
        candidate = by_method[method]
        runtime_reduction = pct_improvement(baseline["total_ms_mean"], candidate["total_ms_mean"])
        qerr_delta = (candidate["qerror_gm"] - baseline["qerror_gm"]) / max(baseline["qerror_gm"], 1e-12) * 100.0
        residual_delta = (
            (candidate["feedback_residual_mean"] - baseline["feedback_residual_mean"])
            / max(baseline["feedback_residual_mean"], 1e-12)
            * 100.0
        )
        passed = (
            runtime_reduction >= args.min_runtime_reduction_pct
            and qerr_delta <= args.max_qerror_regression_pct
            and residual_delta <= args.max_residual_regression_pct
        )
        verdict.append(
            {
                "method": method,
                "pass": passed,
                "runtime_reduction_pct": runtime_reduction,
                "qerror_relative_delta_pct": qerr_delta,
                "feedback_residual_relative_delta_pct": residual_delta,
                "criteria": {
                    "min_runtime_reduction_pct": args.min_runtime_reduction_pct,
                    "max_qerror_regression_pct": args.max_qerror_regression_pct,
                    "max_residual_regression_pct": args.max_residual_regression_pct,
                },
            }
        )

    write_outputs(args.output_dir, rows, summary, verdict)


def parse_args() -> argparse.Namespace:
    root = _REPO_DIR / "experiments" / "results" / "synthetic_paper_suite_rerun_20260529"
    parser = argparse.ArgumentParser(description="Tiny OASIS efficiency smoke test over cached data")
    parser.add_argument("--data-root", type=Path, default=root / "compound_data")
    parser.add_argument("--model-path", type=Path, default=root / "models" / "oasis_k16.json")
    parser.add_argument("--output-dir", type=Path, default=_REPO_DIR / "experiments" / "results" / "oasis_efficiency_smoke_20260531")
    parser.add_argument("--q-values", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--max-cases-per-q", type=int, default=8)
    parser.add_argument("--q-points", type=int, default=24)
    parser.add_argument("--num-buckets", type=int, default=10)
    parser.add_argument("--max-observations", type=int, default=16)
    parser.add_argument("--baseline-projection-iters", type=int, default=200)
    parser.add_argument("--fast-projection-iters", type=int, nargs="+", default=[25, 50, 100])
    parser.add_argument("--projection-tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--min-runtime-reduction-pct", type=float, default=15.0)
    parser.add_argument("--max-qerror-regression-pct", type=float, default=2.0)
    parser.add_argument("--max-residual-regression-pct", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    run_smoke(parse_args())


if __name__ == "__main__":
    main()
