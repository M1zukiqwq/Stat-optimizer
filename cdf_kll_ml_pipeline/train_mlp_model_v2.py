"""train_mlp_model_v2.py — train the enhanced OASIS model (v2) with multi-head attention."""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List, Tuple

from cdf_teacher import correct_quantiles
from json_histogram_parser import load_feedback_sample
from mlp_histogram_model_v2 import MlpHistogramModelV2
from tensorizer import tensorize_sample, OBSERVATION_FEATURE_DIM_NO_TS


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

        tensor_record = tensorize_sample(
            sample,
            max_observations=max_observations,
            teacher_fn=correct_quantiles,
            use_time_decay=False,
        )
        if tensor_record.target_tensor is None:
            continue

        features.append(tensor_record.feature_tensor)
        targets.append(tensor_record.target_tensor)

    if not features:
        raise ValueError("no trainable samples produced from input JSON files")

    return features, targets, files, quantile_levels


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train enhanced OASIS model (v2)")
    parser.add_argument(
        "--train-glob", default="training_data/*.json",
        help="Glob pattern for JSON training samples",
    )
    parser.add_argument(
        "--output-model", default="artifacts/kll_mlp_model_v2.json",
        help="Path to output model JSON",
    )
    parser.add_argument("--max-observations", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=3, help="Number of attention heads")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 128, 64, 64])
    parser.add_argument("--prior-encoder-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 regularisation")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    features, targets, files, quantile_levels = _load_training_set(
        pattern=args.train_glob,
        max_observations=args.max_observations,
    )

    obs_dim = OBSERVATION_FEATURE_DIM_NO_TS   # 12
    prior_dim = len(targets[0])

    model = MlpHistogramModelV2(
        obs_dim=obs_dim,
        prior_dim=prior_dim,
        meta_dim=3,
        max_observations=args.max_observations,
        num_heads=args.num_heads,
        hidden_dims=tuple(args.hidden_dims),
        prior_encoder_dim=args.prior_encoder_dim,
        alpha=args.alpha,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"Training enhanced OASIS v2 ({args.num_heads} heads, {args.hidden_dims}) on {len(features)} samples …")
    model.fit(features, targets)

    predictions = model.predict(features)
    train_mae = _mean_absolute_error(predictions, targets)

    metadata = {
        "train_glob": args.train_glob,
        "train_files": [str(Path(p).resolve()) for p in files],
        "max_observations": args.max_observations,
        "feature_dim": len(features[0]),
        "output_dim": len(targets[0]),
        "quantile_levels": quantile_levels,
        "num_heads": args.num_heads,
        "hidden_dims": args.hidden_dims,
        "prior_encoder_dim": args.prior_encoder_dim,
        "train_mae": train_mae,
        "use_time_decay": False,
    }
    model.save(args.output_model, metadata=metadata)

    print(f"Training samples: {len(features)}")
    print(f"Feature dim: {len(features[0])}   Output dim: {len(targets[0])}")
    print(f"Train MAE (normalized quantiles): {train_mae:.6f}")
    print(f"Saved model: {Path(args.output_model).resolve()}")


if __name__ == "__main__":
    main()
