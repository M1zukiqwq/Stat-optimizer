from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List, Tuple

from cdf_teacher import correct_quantiles
from json_histogram_parser import load_feedback_sample
from ridge_histogram_model import RidgeMultiOutputRegressor
from tensorizer import tensorize_sample


def _mean_absolute_error(predictions: List[List[float]], targets: List[List[float]]) -> float:
    total = 0.0
    count = 0
    for pred_row, target_row in zip(predictions, targets):
        for pred, target in zip(pred_row, target_row):
            total += abs(pred - target)
            count += 1
    return total / max(count, 1)


def _load_training_set(
    pattern: str,
    max_observations: int,
    use_time_decay: bool = True,
) -> Tuple[List[List[float]], List[List[float]], List[str], List[float]]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError(f"no JSON files matched pattern: {pattern}")

    features: List[List[float]] = []
    targets: List[List[float]] = []
    quantile_levels: List[float] = []

    for path in files:
        sample = load_feedback_sample(path)
        if not quantile_levels:
            quantile_levels = list(sample.prior.quantile_levels)
        elif sample.prior.quantile_levels != quantile_levels:
            raise ValueError(
                "all training samples must use same quantile_levels "
                "(assumption: same logical column and same KLL projection grid)"
            )

        tensor_record = tensorize_sample(
            sample,
            max_observations=max_observations,
            teacher_fn=correct_quantiles,
            use_time_decay=use_time_decay,
        )
        if tensor_record.target_tensor is None:
            continue

        features.append(tensor_record.feature_tensor)
        targets.append(tensor_record.target_tensor)

    if not features:
        raise ValueError("no trainable samples produced from input JSON files")

    return features, targets, files, quantile_levels


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ridge model for KLL quantile correction")
    parser.add_argument(
        "--train-glob",
        default="training_data/*.json",
        help="Glob pattern for JSON training samples",
    )
    parser.add_argument(
        "--output-model",
        default="artifacts/kll_ridge_model.json",
        help="Path to output model JSON",
    )
    parser.add_argument(
        "--max-observations",
        type=int,
        default=16,
        help="Observation window size for tensorization",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization coefficient",
    )
    parser.add_argument(
        "--no-time-decay",
        action="store_true",
        help="Exclude the time-decay feature from observation vectors (timestamp ablation)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    use_time_decay = not args.no_time_decay

    features, targets, files, quantile_levels = _load_training_set(
        pattern=args.train_glob,
        max_observations=args.max_observations,
        use_time_decay=use_time_decay,
    )

    model = RidgeMultiOutputRegressor(alpha=args.alpha)
    model.fit(features, targets)

    predictions = model.predict(features)
    train_mae = _mean_absolute_error(predictions, targets)

    metadata = {
        "train_glob": args.train_glob,
        "train_files": [str(Path(path).resolve()) for path in files],
        "max_observations": args.max_observations,
        "feature_dim": len(features[0]),
        "output_dim": len(targets[0]),
        "quantile_levels": quantile_levels,
        "target_semantics": "normalized_kll_quantile_values",
        "train_mae": train_mae,
        "use_time_decay": use_time_decay,
    }
    model.save(args.output_model, metadata=metadata)

    print(f"Training samples: {len(features)}")
    print(f"Feature dim: {len(features[0])}")
    print(f"Output dim: {len(targets[0])}")
    print(f"Train MAE (normalized quantiles): {train_mae:.6f}")
    print(f"Saved model: {Path(args.output_model).resolve()}")


if __name__ == "__main__":
    main()
