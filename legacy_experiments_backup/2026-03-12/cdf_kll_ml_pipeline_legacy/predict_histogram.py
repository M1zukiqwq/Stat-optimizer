from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from cdf_teacher import estimate_observation_coverage
from histogram_math import clamp, project_monotonic
from json_histogram_parser import load_feedback_sample
from kll_codec import encode_simulated_kll, quantiles_to_bucket_boundaries
from ridge_histogram_model import RidgeMultiOutputRegressor
from tensorizer import tensorize_sample



def _denormalize(values: List[float], lower: float, upper: float) -> List[float]:
    value_range = max(upper - lower, 1e-12)
    return [clamp(lower + value * value_range, lower, upper) for value in values]


def _to_markdown_output(payload: Dict[str, object]) -> str:
    levels = payload["quantile_levels"]
    values = payload["corrected_quantile_values"]

    return "\n".join(
        [
            "# Predicted Corrected KLL",
            "",
            "## CorrectedQuantiles",
            f"quantile_levels: {levels}",
            f"quantile_values: {values}",
        ]
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict corrected KLL quantiles from JSON input")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument(
        "--model",
        default="artifacts/kll_ridge_model.json",
        help="Trained model JSON path",
    )
    parser.add_argument(
        "--max-observations",
        type=int,
        help="Override observation window; defaults to model metadata or 16",
    )
    parser.add_argument("--output-json", help="Optional output JSON path")
    parser.add_argument("--output-md", help="Optional output markdown path")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    sample = load_feedback_sample(args.input)
    model_metadata = RidgeMultiOutputRegressor.load_metadata(args.model)
    max_observations = args.max_observations or int(model_metadata.get("max_observations", 16))

    fallback_reason: Optional[str] = None
    if len(sample.observations) < 3:
        fallback_reason = "insufficient observations (<3); returned prior quantiles"

    if fallback_reason is None:
        model = RidgeMultiOutputRegressor.load(args.model)
        tensor_record = tensorize_sample(sample, max_observations=max_observations, teacher_fn=None)
        predicted = model.predict([tensor_record.feature_tensor])[0]
        predicted = project_monotonic(predicted)

        coverage = estimate_observation_coverage(sample)
        if coverage < 0.2:
            blend = coverage / 0.2
            prior_norm = [
                (value - sample.prior.min_value) / max(sample.prior.value_range, 1e-12)
                for value in sample.prior.quantile_values
            ]
            predicted = [
                (1.0 - blend) * prior + blend * pred
                for prior, pred in zip(prior_norm, predicted)
            ]

        corrected_values = _denormalize(predicted, sample.prior.min_value, sample.prior.max_value)
    else:
        corrected_values = list(sample.prior.quantile_values)

    corrected_boundaries = quantiles_to_bucket_boundaries(
        sample.prior.min_value,
        sample.prior.max_value,
        corrected_values,
    )

    corrected_kll_base64 = encode_simulated_kll(
        quantile_levels=sample.prior.quantile_levels,
        quantile_values=corrected_values,
        min_value=sample.prior.min_value,
        max_value=sample.prior.max_value,
        value_type=sample.prior.value_type,
        sketch_k=sample.prior.sketch_k,
    )

    payload = {
        "input_file": str(Path(args.input).resolve()),
        "model_file": str(Path(args.model).resolve()),
        "quantile_levels": sample.prior.quantile_levels,
        "corrected_quantile_values": corrected_values,
        "corrected_kll": {
            "type": sample.prior.value_type,
            "k": sample.prior.sketch_k,
            "quantile_levels": sample.prior.quantile_levels,
            "quantile_values": corrected_values,
            "bucket_boundaries": corrected_boundaries,
            "sketch_bytes_base64": corrected_kll_base64,
        },
        "corrected_histogram": {
            "bucket_boundaries": corrected_boundaries,
        },
        "fallback_reason": fallback_reason,
        "observation_count": len(sample.observations),
    }

    output_json = json.dumps(payload, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(output_json, encoding="utf-8")
    else:
        print(output_json)

    if args.output_md:
        Path(args.output_md).write_text(_to_markdown_output(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
