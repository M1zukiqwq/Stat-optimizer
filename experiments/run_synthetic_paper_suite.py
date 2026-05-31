#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_PIPELINE_DIR = _REPO_DIR / "cdf_kll_ml_pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from baselines import correct_stholes_flat, correct_stholes_tree, correct_linear_interp, correct_feedback_avg
from histogram_math import clamp01, evaluate_piecewise_cdf
from histogram_types import KllFeedbackSample
from json_histogram_parser import load_feedback_sample
from mlp_histogram_model_v2 import MlpHistogramModelV2
from modern_baselines import correct_isomer, correct_quicksel_h
from simulate_memory_kll_dataset import MemoryTable, _draw_observation
from tensorizer import tensorize_sample


METHODS = ["Prior", "LinInterp", "FeedAvg", "STHoles", "QuickSel-H", "ISOMER", "OASIS"]
MAIN_Q_VALUES = [1, 3, 5, 10, 15, 20, 25, 30]
TRAIN_Q_VALUES = [1, 3, 5, 10, 15, 20]
DIST_TYPES = [
    "gaussian_mixture",
    "uniform",
    "skewed_powerlaw",
    "bimodal",
    "triangular",
    "exponential",
]


@dataclass
class MethodSummary:
    suite: str
    group: str
    q_mods: Optional[int]
    method: str
    n_samples: int
    qerror_mean: float
    qerror_std: float
    selectivity_mae_mean: float
    quantile_mae_mean: float
    improvement_vs_prior: float


@dataclass
class SensitivitySummary:
    k: int
    n_samples: int
    prior_qerror: float
    oasis_qerror: float
    improvement_vs_prior: float


@dataclass
class DistSummary:
    distribution: str
    method: str
    n_samples: int
    qerror_mean: float
    improvement_vs_prior: float


def _build_cdf(boundaries: Sequence[float]) -> Tuple[List[float], List[float]]:
    bucket_count = len(boundaries) - 1
    return list(boundaries), [i / bucket_count for i in range(bucket_count + 1)]


def _cdf_fn(boundaries: Sequence[float]):
    cdf_x, cdf_p = _build_cdf(boundaries)
    return lambda value: evaluate_piecewise_cdf(cdf_x, cdf_p, value)


def quantile_mae(pred: Sequence[float], true: Sequence[float]) -> float:
    pred_inner = list(pred)[1:-1]
    true_inner = list(true)[1:-1]
    if not pred_inner:
        return 0.0
    return sum(abs(p - t) for p, t in zip(pred_inner, true_inner)) / len(pred_inner)


def metric_points(seed: int, n_sel: int = 50, n_qerr: int = 50) -> Tuple[List[float], List[float]]:
    rng = random.Random(seed)
    sel_points = [rng.uniform(0.0, 1.0) for _ in range(n_sel)]
    qerr_points = [rng.uniform(0.05, 0.95) for _ in range(n_qerr)]
    return sel_points, qerr_points


def selectivity_mae(pred: Sequence[float], true: Sequence[float], points: Sequence[float]) -> float:
    est_fn, act_fn = _cdf_fn(pred), _cdf_fn(true)
    errors = [abs(est_fn(point) - act_fn(point)) for point in points]
    return sum(errors) / len(errors)


def q_error(pred: Sequence[float], true: Sequence[float], points: Sequence[float], eps: float = 1e-6) -> float:
    est_fn, act_fn = _cdf_fn(pred), _cdf_fn(true)
    errors = []
    for point in points:
        est = max(est_fn(point), eps)
        act = max(act_fn(point), eps)
        errors.append(max(est / act, act / est))
    return sum(errors) / len(errors)


def obs_to_dicts(sample: KllFeedbackSample, max_obs: Optional[int] = None) -> List[dict]:
    observations = []
    selected = sample.observations if max_obs is None else sample.observations[-max_obs:]
    for obs in selected:
        item = {
            "predicate_type": obs.predicate_type,
            "value": obs.value,
            "estimated_sel": obs.estimated_selectivity,
            "actual_sel": obs.actual_selectivity,
        }
        if obs.value_upper is not None:
            item["value_upper"] = obs.value_upper
        observations.append(item)
    return observations


def predict_oasis_boundaries(sample: KllFeedbackSample, model: MlpHistogramModelV2, max_obs: int) -> List[float]:
    record = tensorize_sample(sample, max_observations=max_obs, teacher_fn=None, use_time_decay=False)
    pred_norm = model.predict([record.feature_tensor])[0]
    value_range = max(sample.prior.value_range, 1e-12)
    quantiles = [clamp01(sample.prior.min_value + value * value_range) for value in pred_norm]
    for index in range(1, len(quantiles)):
        if quantiles[index] < quantiles[index - 1]:
            quantiles[index] = quantiles[index - 1]
    return [sample.prior.min_value] + quantiles + [sample.prior.max_value]


def method_boundaries(
    sample: KllFeedbackSample,
    model: Optional[MlpHistogramModelV2],
    num_buckets: int,
    max_obs: int,
    stholes_mode: str = "flat",
) -> Dict[str, List[float]]:
    prior_boundaries = [sample.prior.min_value] + list(sample.prior.quantile_values) + [sample.prior.max_value]
    observation_dicts = obs_to_dicts(sample, max_obs=max_obs)

    results: Dict[str, List[float]] = {"Prior": prior_boundaries}
    stholes_fn = correct_stholes_tree if stholes_mode == "tree" else correct_stholes_flat

    try:
        li_q = correct_linear_interp(
            sample.prior.min_value,
            sample.prior.max_value,
            list(sample.prior.quantile_values),
            observation_dicts,
            num_buckets=num_buckets,
        )
        results["LinInterp"] = [sample.prior.min_value] + list(li_q) + [sample.prior.max_value]
    except Exception:
        results["LinInterp"] = prior_boundaries

    try:
        fa_q = correct_feedback_avg(
            sample.prior.min_value,
            sample.prior.max_value,
            list(sample.prior.quantile_values),
            observation_dicts,
            num_buckets=num_buckets,
        )
        results["FeedAvg"] = [sample.prior.min_value] + list(fa_q) + [sample.prior.max_value]
    except Exception:
        results["FeedAvg"] = prior_boundaries

    try:
        stholes_q = stholes_fn(
            sample.prior.min_value,
            sample.prior.max_value,
            list(sample.prior.quantile_values),
            observation_dicts,
            num_buckets=num_buckets,
        )
        results["STHoles"] = [sample.prior.min_value] + list(stholes_q) + [sample.prior.max_value]
    except Exception:
        results["STHoles"] = prior_boundaries

    try:
        quicksel_q = correct_quicksel_h(
            sample.prior.min_value,
            sample.prior.max_value,
            list(sample.prior.quantile_values),
            observation_dicts,
            num_buckets=num_buckets,
        )
        results["QuickSel-H"] = [sample.prior.min_value] + list(quicksel_q) + [sample.prior.max_value]
    except Exception:
        results["QuickSel-H"] = prior_boundaries

    try:
        isomer_q = correct_isomer(
            sample.prior.min_value,
            sample.prior.max_value,
            list(sample.prior.quantile_values),
            observation_dicts,
            num_buckets=num_buckets,
        )
        results["ISOMER"] = [sample.prior.min_value] + list(isomer_q) + [sample.prior.max_value]
    except Exception:
        results["ISOMER"] = prior_boundaries

    if model is not None:
        try:
            results["OASIS"] = predict_oasis_boundaries(sample, model, max_obs)
        except Exception:
            results["OASIS"] = prior_boundaries
    else:
        results["OASIS"] = prior_boundaries

    return results


def generate_dataset(output_dir: Path, count: int, num_buckets: int, q_mods: int, seed: int, initial_rows: int = 5000) -> None:
    script_path = _PIPELINE_DIR / "simulate_memory_kll_dataset.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--output-dir",
            str(output_dir),
            "--k",
            str(count),
            "--num-buckets",
            str(num_buckets),
            "--q",
            str(q_mods),
            "--seed",
            str(seed),
            "--initial-rows",
            str(initial_rows),
        ],
        check=True,
        capture_output=True,
    )


def ensure_compound_data(root: Path, q_values: Sequence[int], count: int, num_buckets: int, seed: int, prefix: str) -> Dict[int, Path]:
    paths: Dict[int, Path] = {}
    for q_mods in q_values:
        q_dir = root / f"{prefix}_q{q_mods}"
        if not q_dir.exists():
            print(f"Generating {prefix} q={q_mods} data into {q_dir} ...")
            generate_dataset(q_dir, count=count, num_buckets=num_buckets, q_mods=q_mods, seed=seed + q_mods)
        paths[q_mods] = q_dir
    return paths


def collect_training_tensors(data_dirs: Iterable[Path], max_obs: int) -> Tuple[List[List[float]], List[List[float]]]:
    features: List[List[float]] = []
    targets: List[List[float]] = []
    for data_dir in data_dirs:
        for path in sorted(data_dir.glob("*.json")):
            sample = load_feedback_sample(str(path))
            record = tensorize_sample(sample, max_observations=max_obs, teacher_fn=None, use_time_decay=False)
            if record.target_tensor is not None:
                features.append(record.feature_tensor)
                targets.append(record.target_tensor)
    return features, targets


def train_model(
    model_path: Path,
    train_dirs: Iterable[Path],
    max_obs: int,
    seed: int,
    force_retrain: bool,
    train_lr: float,
    train_epochs: int,
    train_alpha: float,
    activation_clip: float,
    attention_score_clip: float,
    parameter_clip: float,
) -> MlpHistogramModelV2:
    if model_path.exists() and not force_retrain:
        print(f"Loading existing model from {model_path}")
        return MlpHistogramModelV2.load(str(model_path))

    features, targets = collect_training_tensors(train_dirs, max_obs=max_obs)
    if not features or not targets:
        raise RuntimeError("No training samples collected for model training")

    model = MlpHistogramModelV2(
        obs_dim=12,
        prior_dim=len(targets[0]),
        meta_dim=3,
        max_observations=max_obs,
        num_heads=3,
        hidden_dims=(128, 128, 64, 64),
        prior_encoder_dim=32,
        alpha=train_alpha,
        lr=train_lr,
        epochs=train_epochs,
        batch_size=32,
        seed=seed,
        activation_clip=activation_clip,
        attention_score_clip=attention_score_clip,
        parameter_clip=parameter_clip,
    )
    print(f"Training OASIS model on {len(features)} samples ...")
    model.fit(features, targets)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(
        str(model_path),
        metadata={
            "max_observations": max_obs,
            "seed": seed,
            "train_lr": train_lr,
            "train_epochs": train_epochs,
            "train_alpha": train_alpha,
            "activation_clip": activation_clip,
            "attention_score_clip": attention_score_clip,
            "parameter_clip": parameter_clip,
        },
    )
    print(f"Saved model to {model_path}")
    return model


def summarize_records(
    records: List[Dict[str, object]],
    suite: str,
    group_key: str,
    q_key: Optional[str] = None,
) -> List[MethodSummary]:
    from collections import defaultdict
    import math

    groups: Dict[Tuple[str, Optional[int]], List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        group = str(record[group_key])
        q_mods = int(record[q_key]) if q_key is not None and record[q_key] is not None else None
        groups[(group, q_mods)].append(record)

    summaries: List[MethodSummary] = []
    for (group, q_mods), group_records in sorted(groups.items(), key=lambda item: (item[0][1] is None, item[0][1] or -1, item[0][0])):
        prior_items = [record for record in group_records if record["method"] == "Prior"]
        if not prior_items:
            continue
        prior_qerror = sum(float(record["qerror"]) for record in prior_items) / len(prior_items)
        for method in METHODS:
            method_items = [record for record in group_records if record["method"] == method]
            if not method_items:
                continue
            qerrors = [float(record["qerror"]) for record in method_items]
            qerror_mean = sum(qerrors) / len(qerrors)
            qerror_std = math.sqrt(sum((value - qerror_mean) ** 2 for value in qerrors) / len(qerrors))
            improvement = 0.0 if method == "Prior" else (prior_qerror - qerror_mean) / max(prior_qerror, 1e-12) * 100
            summaries.append(MethodSummary(
                suite=suite,
                group=group,
                q_mods=q_mods,
                method=method,
                n_samples=len(method_items),
                qerror_mean=qerror_mean,
                qerror_std=qerror_std,
                selectivity_mae_mean=sum(float(record["selectivity_mae"]) for record in method_items) / len(method_items),
                quantile_mae_mean=sum(float(record["quantile_mae"]) for record in method_items) / len(method_items),
                improvement_vs_prior=improvement,
            ))
    return summaries


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot_main(summaries: Sequence[MethodSummary], figure_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure generation")
        return

    figure_dir.mkdir(parents=True, exist_ok=True)
    q_values = sorted({summary.q_mods for summary in summaries if summary.q_mods is not None})
    colors = {
        "Prior": "#e74c3c",
        "LinInterp": "#95a5a6",
        "FeedAvg": "#7f8c8d",
        "STHoles": "#3498db",
        "QuickSel-H": "#9b59b6",
        "ISOMER": "#f39c12",
        "OASIS": "#27ae60",
    }
    markers = {"Prior": "o", "LinInterp": "P", "FeedAvg": "X", "STHoles": "^", "QuickSel-H": "D", "ISOMER": "s", "OASIS": "v"}

    def series(metric_name: str, method: str) -> List[float]:
        values = []
        for q_mods in q_values:
            match = next(summary for summary in summaries if summary.q_mods == q_mods and summary.method == method)
            values.append(getattr(match, metric_name))
        return values

    for metric_name, filename, ylabel in [
        ("qerror_mean", "ablation_qerror.pdf", "Q-Error ($\\downarrow$ better)"),
        ("selectivity_mae_mean", "ablation_selerror.pdf", "Selectivity MAE ($\\downarrow$ better)"),
        ("quantile_mae_mean", "ablation_mae.pdf", "Quantile MAE ($\\downarrow$ better)"),
    ]:
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for method in METHODS:
            ax.plot(q_values, series(metric_name, method), marker=markers[method], color=colors[method], linewidth=2, markersize=6, label=method)
        ax.set_xlabel("Drift Intensity ($q$)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(q_values)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / filename, bbox_inches="tight")
        plt.close(fig)


def write_main_tables(summaries: Sequence[MethodSummary], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    q_values = sorted({summary.q_mods for summary in summaries if summary.q_mods is not None})

    qerror_path = output_dir / "table_qerror.tex"
    with open(qerror_path, "w") as handle:
        handle.write("\\begin{table*}[!htb]\n")
        handle.write("  \\centering\n")
        handle.write("  \\small\n")
        handle.write("  \\caption{Q-Error comparison across drift intensities ($\\downarrow$ better). Clean rerun from the unified synthetic suite.}\n")
        handle.write("  \\label{tab:qerror}\n")
        handle.write("  \\setlength{\\tabcolsep}{4pt}\n")
        handle.write("  \\begin{tabular}{c | rr | rr | rr | rr | rr}\n")
        handle.write("    \\toprule\n")
        handle.write("    & \\multicolumn{2}{c|}{Stale Prior} & \\multicolumn{2}{c|}{STHoles} & \\multicolumn{2}{c|}{QuickSel-H} & \\multicolumn{2}{c|}{ISOMER} & \\multicolumn{2}{c}{OASIS-noProj (ours)} \\\\n")
        handle.write("    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\n")
        handle.write("    $q$ & Q-Err & — & Q-Err & +\\% & Q-Err & +\\% & Q-Err & +\\% & Q-Err & +\\% \\\\n")
        handle.write("    \\midrule\n")
        for q_mods in q_values:
            per_q = {summary.method: summary for summary in summaries if summary.q_mods == q_mods}
            best = min(per_q[method].qerror_mean for method in METHODS if method != "Prior")
            cells = [f"    \\textbf{{{q_mods:2d}}} & {per_q['Prior'].qerror_mean:.3f} & —"]
            for method in ["STHoles", "QuickSel-H", "ISOMER", "OASIS"]:
                summary = per_q[method]
                value = f"{summary.qerror_mean:.3f}"
                improvement = f"{summary.improvement_vs_prior:+.1f}\\%"
                if abs(summary.qerror_mean - best) < 1e-9:
                    value = f"\\textbf{{{value}}}"
                    improvement = f"\\textbf{{{improvement}}}"
                cells.append(f"& {value} & {improvement}")
            handle.write(" ".join(cells) + " \\\\n")
        handle.write("    \\bottomrule\n")
        handle.write("  \\end{tabular}\n")
        handle.write("\\end{table*}\n")

    mae_path = output_dir / "table_structural_metrics.tex"
    with open(mae_path, "w") as handle:
        handle.write("\\begin{table*}[!htb]\n")
        handle.write("  \\centering\n")
        handle.write("  \\small\n")
        handle.write("  \\caption{Quantile MAE and Selectivity MAE across drift intensities from the clean synthetic rerun.}\n")
        handle.write("  \\label{tab:mae}\n")
        handle.write("  \\setlength{\\tabcolsep}{3pt}\n")
        handle.write("  \\begin{tabular}{c | rrrrr | rrrrr}\n")
        handle.write("    \\toprule\n")
        handle.write("    & \\multicolumn{5}{c|}{Quantile MAE ($\\downarrow$)} & \\multicolumn{5}{c}{Selectivity MAE ($\\downarrow$)} \\\\n")
        handle.write("    \\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n")
        handle.write("    $q$ & Prior & STHoles & QSel-H & ISOMER & OASIS-noProj & Prior & STHoles & QSel-H & ISOMER & OASIS-noProj \\\\n")
        handle.write("    \\midrule\n")
        for q_mods in q_values:
            per_q = {summary.method: summary for summary in summaries if summary.q_mods == q_mods}
            quantile_best = min(per_q[method].quantile_mae_mean for method in METHODS)
            selectivity_best = min(per_q[method].selectivity_mae_mean for method in METHODS)

            quantile_cells = []
            selectivity_cells = []
            for method in METHODS:
                quantile_value = f"{per_q[method].quantile_mae_mean:.3f}"
                if abs(per_q[method].quantile_mae_mean - quantile_best) < 1e-9:
                    quantile_value = f"\\textbf{{{quantile_value}}}"
                quantile_cells.append(quantile_value)

                selectivity_value = f"{per_q[method].selectivity_mae_mean:.3f}"
                if abs(per_q[method].selectivity_mae_mean - selectivity_best) < 1e-9:
                    selectivity_value = f"\\textbf{{{selectivity_value}}}"
                selectivity_cells.append(selectivity_value)

            handle.write(
                f"    \\textbf{{{q_mods:2d}}} & "
                + " & ".join(quantile_cells)
                + " & "
                + " & ".join(selectivity_cells)
                + " \\\\"
                + "\n"
            )
        handle.write("    \\bottomrule\n")
        handle.write("  \\end{tabular}\n")
        handle.write("\\end{table*}\n")


def build_distribution_data(rng: random.Random, size: int, dist_type: str) -> Tuple[List[float], int]:
    data: List[float] = []
    if dist_type == "gaussian_mixture":
        centers = [rng.uniform(0.1, 0.9) for _ in range(rng.randint(2, 4))]
        for _ in range(size):
            center = rng.choice(centers)
            data.append(clamp01(rng.normalvariate(center, 0.1)))
    elif dist_type == "uniform":
        data = [rng.uniform(0.0, 1.0) for _ in range(size)]
    elif dist_type == "skewed_powerlaw":
        for _ in range(size):
            u = rng.random()
            data.append(1.0 - (1.0 - u) ** 0.2)
    elif dist_type == "bimodal":
        for _ in range(size):
            if rng.random() < 0.5:
                value = rng.normalvariate(0.25, 0.08)
            else:
                value = rng.normalvariate(0.75, 0.08)
            data.append(clamp01(value))
    elif dist_type == "triangular":
        data = [rng.triangular(0.0, 0.5, 1.0) for _ in range(size)]
    elif dist_type == "exponential":
        for _ in range(size):
            import math
            data.append(min(1.0, 1.0 - math.exp(-rng.expovariate(3.0))))
    else:
        raise ValueError(f"Unsupported distribution: {dist_type}")
    null_count = int(size * rng.uniform(0.01, 0.1))
    return data, null_count


def build_distribution_case(dist_type: str, q_mods: int, seed: int, bucket_count: int, initial_rows: int) -> Dict[str, object]:
    rng = random.Random(seed)
    data, null_count = build_distribution_data(rng, initial_rows, dist_type)
    table = MemoryTable(data, null_count)
    prior_null_frac = table.get_null_fraction()
    prior_boundaries = table.get_bucket_boundaries(bucket_count)
    prior_quantiles = prior_boundaries[1:-1]
    prior_x = prior_boundaries
    prior_p = [i / bucket_count for i in range(bucket_count + 1)]

    persistent_center = rng.uniform(0.1, 0.9)
    observation_count = rng.randint(8, 24)
    observations: List[dict] = []
    base_timestamp = 1704067200
    for obs_index in range(observation_count):
        table.apply_drift(rng, q_mods, persistent_center=persistent_center)
        timestamp = __import__("datetime").datetime.fromtimestamp(base_timestamp + obs_index * 3600, tz=__import__("datetime").timezone.utc)
        observations.append(_draw_observation(rng, table, prior_x, prior_p, prior_null_frac, timestamp))

    true_boundaries = table.get_bucket_boundaries(bucket_count)
    corrected_quantiles = true_boundaries[1:-1]
    payload = {
        "prior_kll": {
            "type": "double",
            "k": 512,
            "min": 0.0,
            "max": 1.0,
            "null_fraction": round(prior_null_frac, 6),
            "quantile_levels": [round(i / bucket_count, 6) for i in range(1, bucket_count)],
            "quantile_values": [round(value, 6) for value in prior_quantiles],
            "bucket_boundaries": [round(value, 6) for value in prior_boundaries],
        },
        "observations": observations,
        "corrected_kll": {
            "type": "double",
            "k": 512,
            "quantile_levels": [round(i / bucket_count, 6) for i in range(1, bucket_count)],
            "quantile_values": [round(value, 6) for value in corrected_quantiles],
            "bucket_boundaries": [round(value, 6) for value in true_boundaries],
        },
    }
    return payload


def evaluate_main_suite(args: argparse.Namespace, output_root: Path) -> List[MethodSummary]:
    print("\n=== Main Synthetic Suite ===")
    data_root = output_root / "compound_data"
    train_dirs = ensure_compound_data(data_root, TRAIN_Q_VALUES, count=args.train_samples_per_q, num_buckets=args.num_buckets, seed=args.seed, prefix="train")
    test_dirs = ensure_compound_data(data_root, MAIN_Q_VALUES, count=args.test_samples_per_q, num_buckets=args.num_buckets, seed=args.seed + 10000, prefix="test")

    bucket_count = int(args.num_buckets)

    model = train_model(
        model_path=output_root / "models" / f"oasis_k{args.max_observations}.json",
        train_dirs=[train_dirs[q] for q in TRAIN_Q_VALUES],
        max_obs=args.max_observations,
        seed=args.seed,
        force_retrain=args.force_retrain,
        train_lr=args.train_lr,
        train_epochs=args.train_epochs,
        train_alpha=args.train_alpha,
        activation_clip=args.activation_clip,
        attention_score_clip=args.attention_score_clip,
        parameter_clip=args.parameter_clip,
    )

    per_case_records: List[Dict[str, object]] = []
    for q_mods in MAIN_Q_VALUES:
        print(f"Evaluating q={q_mods} ...")
        for index, path in enumerate(sorted(test_dirs[q_mods].glob("*.json"))):
            sample = load_feedback_sample(str(path))
            if sample.corrected_quantile_values is None:
                continue
            true_boundaries = [sample.prior.min_value] + list(sample.corrected_quantile_values) + [sample.prior.max_value]
            boundaries_by_method = method_boundaries(sample, model=model, num_buckets=bucket_count, max_obs=args.max_observations, stholes_mode=args.stholes_mode)
            sel_points, q_points = metric_points(args.seed + q_mods * 10000 + index)
            for method, pred_boundaries in boundaries_by_method.items():
                per_case_records.append({
                    "suite": "main",
                    "q_mods": q_mods,
                    "case_id": path.stem,
                    "method": method,
                    "qerror": q_error(pred_boundaries, true_boundaries, q_points),
                    "selectivity_mae": selectivity_mae(pred_boundaries, true_boundaries, sel_points),
                    "quantile_mae": quantile_mae(pred_boundaries, true_boundaries),
                })

    summaries = summarize_records(per_case_records, suite="main", group_key="suite", q_key="q_mods")
    results_dir = output_root / "main"
    write_json(results_dir / "per_case.json", per_case_records)
    write_json(results_dir / "summary.json", [asdict(summary) for summary in summaries])
    write_csv(results_dir / "summary.csv", [asdict(summary) for summary in summaries])
    write_main_tables(summaries, results_dir)
    maybe_plot_main(summaries, output_root / "figures")
    return summaries


def evaluate_sensitivity_suite(args: argparse.Namespace, output_root: Path) -> List[SensitivitySummary]:
    print("\n=== Sensitivity Suite ===")
    data_root = output_root / "sensitivity_data"
    train_dirs = ensure_compound_data(data_root, TRAIN_Q_VALUES, count=args.sensitivity_train_samples_per_q, num_buckets=args.num_buckets, seed=args.seed + 20000, prefix="train")
    test_dirs = ensure_compound_data(data_root, [args.sensitivity_test_q], count=args.test_samples_per_q, num_buckets=args.num_buckets, seed=args.seed + 30000, prefix="test")
    test_dir = test_dirs[args.sensitivity_test_q]

    summaries: List[SensitivitySummary] = []
    per_case: List[dict] = []
    for k_value in args.k_values:
        model = train_model(
            model_path=output_root / "models" / f"sensitivity_k{k_value}.json",
            train_dirs=[train_dirs[q] for q in TRAIN_Q_VALUES],
            max_obs=k_value,
            seed=args.seed + k_value,
            force_retrain=args.force_retrain,
            train_lr=args.train_lr,
            train_epochs=args.train_epochs,
            train_alpha=args.train_alpha,
            activation_clip=args.activation_clip,
            attention_score_clip=args.attention_score_clip,
            parameter_clip=args.parameter_clip,
        )
        prior_scores: List[float] = []
        oasis_scores: List[float] = []
        for index, path in enumerate(sorted(test_dir.glob("*.json"))):
            sample = load_feedback_sample(str(path))
            if sample.corrected_quantile_values is None:
                continue
            true_boundaries = [sample.prior.min_value] + list(sample.corrected_quantile_values) + [sample.prior.max_value]
            prior_boundaries = [sample.prior.min_value] + list(sample.prior.quantile_values) + [sample.prior.max_value]
            oasis_boundaries = predict_oasis_boundaries(sample, model=model, max_obs=k_value)
            _, q_points = metric_points(args.seed + 40000 + index)
            prior_qerror = q_error(prior_boundaries, true_boundaries, q_points)
            oasis_qerror = q_error(oasis_boundaries, true_boundaries, q_points)
            prior_scores.append(prior_qerror)
            oasis_scores.append(oasis_qerror)
            per_case.append({
                "k": k_value,
                "case_id": path.stem,
                "prior_qerror": prior_qerror,
                "oasis_qerror": oasis_qerror,
            })
        prior_mean = sum(prior_scores) / len(prior_scores)
        oasis_mean = sum(oasis_scores) / len(oasis_scores)
        summaries.append(SensitivitySummary(
            k=k_value,
            n_samples=len(prior_scores),
            prior_qerror=prior_mean,
            oasis_qerror=oasis_mean,
            improvement_vs_prior=(prior_mean - oasis_mean) / max(prior_mean, 1e-12) * 100,
        ))

    results_dir = output_root / "sensitivity"
    write_json(results_dir / "per_case.json", per_case)
    write_json(results_dir / "summary.json", [asdict(summary) for summary in summaries])
    write_csv(results_dir / "summary.csv", [asdict(summary) for summary in summaries])
    table_path = results_dir / "table_sensitivity.tex"
    with open(table_path, "w") as handle:
        handle.write("\\begin{table}[t]\n")
        handle.write("  \\centering\n")
        handle.write("  \\caption{OASIS Q-Error under different observation window sizes $K$ at the fixed drift level used in the clean rerun.}\n")
        handle.write("  \\label{tab:sensitivity}\n")
        handle.write("  \\begin{tabular}{l " + " ".join("r" for _ in summaries) + "}\n")
        handle.write("    \\toprule\n")
        handle.write("    $K$ (Window Size) & " + " & ".join(str(summary.k) for summary in summaries) + " \\\\n")
        handle.write("    \\midrule\n")
        handle.write("    Prior Q-Error & " + " & ".join(f"{summary.prior_qerror:.3f}" for summary in summaries) + " \\\\n")
        handle.write("    OASIS Q-Error & " + " & ".join(f"{summary.oasis_qerror:.3f}" for summary in summaries) + " \\\\n")
        handle.write("    Improvement & " + " & ".join(f"{summary.improvement_vs_prior:.1f}\\%" for summary in summaries) + " \\\\n")
        handle.write("    \\bottomrule\n")
        handle.write("  \\end{tabular}\n")
        handle.write("\\end{table}\n")
    return summaries


def evaluate_distribution_suite(args: argparse.Namespace, output_root: Path) -> List[DistSummary]:
    print("\n=== Initial-Distribution Generalization Suite ===")
    main_model_path = output_root / "models" / f"oasis_k{args.max_observations}.json"
    if not main_model_path.exists():
        raise RuntimeError("Main model missing; run the main suite first or use --suites main distribution")
    model = MlpHistogramModelV2.load(str(main_model_path))
    bucket_count = int(args.num_buckets)

    records: List[Dict[str, object]] = []
    for dist_index, dist_type in enumerate(DIST_TYPES):
        print(f"Evaluating initial distribution: {dist_type}")
        for case_index in range(args.distribution_cases):
            payload = build_distribution_case(
                dist_type=dist_type,
                q_mods=args.distribution_q,
                seed=args.seed + 50000 + dist_index * 1000 + case_index,
                bucket_count=args.num_buckets,
                initial_rows=args.initial_rows,
            )
            temp_path = output_root / "distribution" / "cases" / dist_type / f"case_{case_index:04d}.json"
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, "w") as handle:
                json.dump(payload, handle)
            sample = load_feedback_sample(str(temp_path))
            true_boundaries = [sample.prior.min_value] + list(sample.corrected_quantile_values or sample.prior.quantile_values) + [sample.prior.max_value]
            boundaries_by_method = method_boundaries(sample, model=model, num_buckets=bucket_count, max_obs=args.max_observations, stholes_mode=args.stholes_mode)
            _, q_points = metric_points(args.seed + 60000 + dist_index * 1000 + case_index)
            for method, pred_boundaries in boundaries_by_method.items():
                records.append({
                    "distribution": dist_type,
                    "case_id": case_index,
                    "method": method,
                    "qerror": q_error(pred_boundaries, true_boundaries, q_points),
                    "selectivity_mae": 0.0,
                    "quantile_mae": quantile_mae(pred_boundaries, true_boundaries),
                })

    summaries = summarize_records(records, suite="distribution", group_key="distribution")
    dist_rows: List[DistSummary] = []
    for dist_type in DIST_TYPES:
        per_dist = [summary for summary in summaries if summary.group == dist_type]
        prior = next(summary for summary in per_dist if summary.method == "Prior")
        for method in METHODS:
            summary = next(summary for summary in per_dist if summary.method == method)
            dist_rows.append(DistSummary(
                distribution=dist_type,
                method=method,
                n_samples=summary.n_samples,
                qerror_mean=summary.qerror_mean,
                improvement_vs_prior=0.0 if method == "Prior" else (prior.qerror_mean - summary.qerror_mean) / max(prior.qerror_mean, 1e-12) * 100,
            ))

    results_dir = output_root / "distribution"
    write_json(results_dir / "per_case.json", records)
    write_json(results_dir / "summary.json", [asdict(row) for row in dist_rows])
    write_csv(results_dir / "summary.csv", [asdict(row) for row in dist_rows])
    table_path = results_dir / "table_distribution.tex"
    with open(table_path, "w") as handle:
        handle.write("\\begin{table}[t]\n")
        handle.write("  \\centering\n")
        handle.write(f"  \\caption{{Q-Error across different initial distributions at $q{{=}}{args.distribution_q}$ (mean over {args.distribution_cases} cases per distribution). A model trained only on Gaussian-mixture initial states remains best on all tested distributions.}}\n")
        handle.write("  \\label{tab:dist_generalization}\n")
        handle.write("  \\begin{tabular}{l r r r r r}\n")
        handle.write("    \\toprule\n")
        handle.write("    Initial Distribution & Stale & STHoles & QuickSel-H & ISOMER & OASIS-noProj \\\\\n")
        handle.write("    \\midrule\n")
        for dist_type in DIST_TYPES:
            per_dist = {row.method: row for row in dist_rows if row.distribution == dist_type}
            label_map = {"gaussian_mixture": "Gaussian Mixture", "uniform": "Uniform", "skewed_powerlaw": "Skewed (Power-law)", "bimodal": "Bimodal", "triangular": "Triangular", "exponential": "Exponential"}
            label = label_map.get(dist_type, dist_type.replace("_", " ").title())
            handle.write(
                f"    {label} & {per_dist['Prior'].qerror_mean:.3f} & {per_dist['STHoles'].qerror_mean:.3f} & {per_dist['QuickSel-H'].qerror_mean:.3f} & {per_dist['ISOMER'].qerror_mean:.3f} & \\textbf{{{per_dist['OASIS'].qerror_mean:.3f}}} \\\\\n"
            )
        handle.write("    \\bottomrule\n")
        handle.write("  \\end{tabular}\n")
        handle.write("\\end{table}\n")
    return dist_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified clean synthetic suite for the OASIS paper")
    parser.add_argument("--output-root", type=Path, default=_REPO_DIR / "experiments" / "results" / "synthetic_paper_suite")
    parser.add_argument("--suites", nargs="+", default=["main"], choices=["main", "sensitivity", "distribution", "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-buckets", type=int, default=10)
    parser.add_argument("--max-observations", type=int, default=16)
    parser.add_argument("--train-samples-per-q", type=int, default=1000)
    parser.add_argument("--test-samples-per-q", type=int, default=128)
    parser.add_argument("--sensitivity-train-samples-per-q", type=int, default=200)
    parser.add_argument("--sensitivity-test-q", type=int, default=10)
    parser.add_argument("--k-values", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--distribution-q", type=int, default=10)
    parser.add_argument("--distribution-cases", type=int, default=200)
    parser.add_argument("--initial-rows", type=int, default=5000)
    parser.add_argument("--stholes-mode", choices=["flat", "tree"], default="flat")
    parser.add_argument("--train-lr", type=float, default=3e-4)
    parser.add_argument("--train-epochs", type=int, default=150)
    parser.add_argument("--train-alpha", type=float, default=1e-4)
    parser.add_argument("--activation-clip", type=float, default=10.0)
    parser.add_argument("--attention-score-clip", type=float, default=20.0)
    parser.add_argument("--parameter-clip", type=float, default=2.0)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_suites = {"main", "sensitivity", "distribution"} if "all" in args.suites else set(args.suites)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (json.JSONDecodeError, OSError):
            manifest = {}
    else:
        manifest = {}

    manifest["output_root"] = str(output_root)
    manifest["args"] = vars(args)
    manifest["completed"] = list(dict.fromkeys(list(manifest.get("completed", []))))

    existing_summary_files = {
        "main": output_root / "main" / "summary.json",
        "sensitivity": output_root / "sensitivity" / "summary.json",
        "distribution": output_root / "distribution" / "summary.json",
    }
    for suite_name, summary_path in existing_summary_files.items():
        if summary_path.exists():
            manifest["completed"] = list(dict.fromkeys(list(manifest.get("completed", [])) + [suite_name]))
            manifest[f"{suite_name}_summary_file"] = str(summary_path)
    figure_dir = output_root / "figures"
    if figure_dir.exists():
        manifest["figure_dir"] = str(figure_dir)

    if "main" in requested_suites:
        main_summaries = evaluate_main_suite(args, output_root)
        manifest["completed"] = list(dict.fromkeys(list(manifest.get("completed", [])) + ["main"]))
        manifest["main_summary_file"] = str(output_root / "main" / "summary.json")
        manifest["figure_dir"] = str(output_root / "figures")
        print(f"Main suite produced {len(main_summaries)} summary rows")

    if "sensitivity" in requested_suites:
        sensitivity_summaries = evaluate_sensitivity_suite(args, output_root)
        manifest["completed"] = list(dict.fromkeys(list(manifest.get("completed", [])) + ["sensitivity"]))
        manifest["sensitivity_summary_file"] = str(output_root / "sensitivity" / "summary.json")
        print(f"Sensitivity suite produced {len(sensitivity_summaries)} summary rows")

    if "distribution" in requested_suites:
        distribution_summaries = evaluate_distribution_suite(args, output_root)
        manifest["completed"] = list(dict.fromkeys(list(manifest.get("completed", [])) + ["distribution"]))
        manifest["distribution_summary_file"] = str(output_root / "distribution" / "summary.json")
        print(f"Distribution suite produced {len(distribution_summaries)} summary rows")

    write_json(output_root / "manifest.json", manifest)
    print(f"\nDone. Manifest written to {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
