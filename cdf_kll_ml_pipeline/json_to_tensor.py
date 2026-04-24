from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdf_teacher import correct_quantiles
from json_histogram_parser import load_feedback_sample
from tensorizer import tensorize_sample


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert KLL feedback JSON into model tensors")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", help="Optional output JSON file")
    parser.add_argument(
        "--max-observations",
        type=int,
        default=16,
        help="Observation window size for tensorization",
    )
    parser.add_argument(
        "--no-teacher-target",
        action="store_true",
        help="Do not generate target tensor from teacher when corrected quantiles are missing",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    sample = load_feedback_sample(args.input)
    teacher_fn = None if args.no_teacher_target else correct_quantiles
    tensor_record = tensorize_sample(
        sample,
        max_observations=args.max_observations,
        teacher_fn=teacher_fn,
    )

    payload = {
        "input_file": str(Path(args.input).resolve()),
        "feature_tensor": tensor_record.feature_tensor,
        "observation_tensor": tensor_record.observation_tensor,
        "mask_tensor": tensor_record.mask_tensor,
        "target_tensor": tensor_record.target_tensor,
        "metadata": tensor_record.metadata,
    }

    output_text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
