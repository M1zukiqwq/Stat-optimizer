from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

from histogram_math import clamp01, evaluate_piecewise_cdf, project_monotonic
from kll_codec import default_quantile_levels, encode_simulated_kll

PREDICATES = ["<", "<=", ">", ">=", "=", "BETWEEN"]



def _build_piecewise(boundaries: List[float]) -> Tuple[List[float], List[float]]:
    bucket_count = len(boundaries) - 1
    cdf_x = list(boundaries)
    cdf_p = [index / bucket_count for index in range(bucket_count + 1)]
    return cdf_x, cdf_p


def _sample_boundaries(bucket_count: int, rng: random.Random, noise_scale: float) -> List[float]:
    if bucket_count < 2:
        raise ValueError("bucket_count must be >= 2")

    base = [index / bucket_count for index in range(bucket_count + 1)]
    interior = [clamp01(base[index] + rng.uniform(-noise_scale, noise_scale)) for index in range(1, bucket_count)]
    interior = project_monotonic(interior)
    return [0.0] + interior + [1.0]


def _draw_observation(
    rng: random.Random,
    prior_x: List[float],
    prior_p: List[float],
    true_x: List[float],
    true_p: List[float],
    ts: datetime,
    null_fraction: float = 0.0,
) -> dict:
    non_null_frac = max(1.0 - null_fraction, 1e-6)

    predicate = rng.choice(PREDICATES)
    value = rng.uniform(0.0, 1.0)
    value_upper = None

    if predicate == "BETWEEN":
        second = rng.uniform(0.0, 1.0)
        value, value_upper = sorted((value, second))
        est_overall = clamp01(evaluate_piecewise_cdf(prior_x, prior_p, value_upper) - evaluate_piecewise_cdf(prior_x, prior_p, value))
        act_overall = clamp01(evaluate_piecewise_cdf(true_x, true_p, value_upper) - evaluate_piecewise_cdf(true_x, true_p, value))
    elif predicate in {"<", "<="}:
        est_overall = clamp01(evaluate_piecewise_cdf(prior_x, prior_p, value))
        act_overall = clamp01(evaluate_piecewise_cdf(true_x, true_p, value))
    elif predicate in {">", ">="}:
        est_overall = clamp01(1.0 - evaluate_piecewise_cdf(prior_x, prior_p, value))
        act_overall = clamp01(1.0 - evaluate_piecewise_cdf(true_x, true_p, value))
    else:
        width = 0.01
        left = max(0.0, value - width)
        right = min(1.0, value + width)
        est_overall = clamp01(evaluate_piecewise_cdf(prior_x, prior_p, right) - evaluate_piecewise_cdf(prior_x, prior_p, left))
        act_overall = clamp01(evaluate_piecewise_cdf(true_x, true_p, right) - evaluate_piecewise_cdf(true_x, true_p, left))

    # cdf 算出的是在非空行中的选择率（因为 evaluate_piecewise_cdf 只分布在 [0, 1] 有效值中）。
    # Presto 真实的反馈是整体 selection = (非空行条件选择率) * non_null_frac。
    # 这样 Teacher 的 _effective_actual 除以 (1 - null_fraction) 就能正确还原。
    est = clamp01(est_overall * non_null_frac)
    act = clamp01(act_overall * non_null_frac)

    observation = {
        "predicate_type": predicate,
        "value": round(value, 6),
        "estimated_sel": round(est, 6),
        "actual_sel": round(act, 6),
        "timestamp": ts.isoformat().replace("+00:00", "Z"),
    }
    if value_upper is not None:
        observation["value_upper"] = round(value_upper, 6)
    return observation


def build_case(case_index: int, rng: random.Random, bucket_count: int, sketch_k: int) -> dict:
    null_fraction = rng.uniform(0.0, 0.12)

    true_boundaries = _sample_boundaries(bucket_count=bucket_count, rng=rng, noise_scale=0.08)
    prior_boundaries = _sample_boundaries(bucket_count=bucket_count, rng=rng, noise_scale=0.04)

    prior_x, prior_p = _build_piecewise(prior_boundaries)
    true_x, true_p = _build_piecewise(true_boundaries)

    observation_count = rng.randint(8, 24)
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=case_index)
    observations = []
    for obs_index in range(observation_count):
        ts = base_time + timedelta(hours=obs_index)
        observations.append(
            _draw_observation(rng, prior_x, prior_p, true_x, true_p, ts, null_fraction=null_fraction)
        )

    quantile_levels = default_quantile_levels(bucket_count - 1)
    prior_quantiles = prior_boundaries[1:-1]
    true_quantiles = true_boundaries[1:-1]

    prior_sketch_base64 = encode_simulated_kll(
        quantile_levels=quantile_levels,
        quantile_values=prior_quantiles,
        min_value=0.0,
        max_value=1.0,
        value_type="double",
        sketch_k=sketch_k,
    )

    return {
        "prior_kll": {
            "type": "double",
            "k": sketch_k,
            "min": 0.0,
            "max": 1.0,
            "null_fraction": round(null_fraction, 6),
            "quantile_levels": [round(level, 6) for level in quantile_levels],
            "quantile_values": [round(value, 6) for value in prior_quantiles],
            "bucket_boundaries": [round(value, 6) for value in prior_boundaries],
            "sketch_bytes_base64": prior_sketch_base64,
        },
        "observations": observations,
        "corrected_kll": {
            "type": "double",
            "k": sketch_k,
            "quantile_levels": [round(level, 6) for level in quantile_levels],
            "quantile_values": [round(value, 6) for value in true_quantiles],
            "bucket_boundaries": [round(value, 6) for value in true_boundaries],
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic KLL correction training dataset")
    parser.add_argument("--output-dir", default="training_data", help="Output directory for generated JSON files")
    parser.add_argument("--k", type=int, default=32, help="Generate k training JSON files")
    parser.add_argument("--num-buckets", type=int, default=10, help="Quantile grid points derive from this bucket count")
    parser.add_argument("--sketch-k", type=int, default=1024, help="K parameter stored in generated KLL metadata")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    for case_index in range(args.k):
        payload = build_case(case_index, rng, bucket_count=args.num_buckets, sketch_k=args.sketch_k)
        output_file = output_dir / f"synthetic_case_{case_index:04d}.json"
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Generated {args.k} files into {output_dir.resolve()}")


if __name__ == "__main__":
    main()
