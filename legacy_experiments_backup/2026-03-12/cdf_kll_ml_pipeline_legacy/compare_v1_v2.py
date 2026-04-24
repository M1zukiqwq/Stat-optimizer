"""
compare_v1_v2.py
================
Compare OASIS v1 (single-head attention) vs v2 (multi-head attention) on heavy drift scenarios.

Usage:
    python3 compare_v1_v2.py --train-q-values 10 20 --k-train 1000 --q-values 10 15 20 25 30
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from cdf_teacher import correct_quantiles
from histogram_math import clamp01, evaluate_piecewise_cdf
from histogram_types import KllFeedbackSample
from json_histogram_parser import load_feedback_sample
from mlp_histogram_model import MlpHistogramModel
from mlp_histogram_model_v2 import MlpHistogramModelV2
from tensorizer import tensorize_sample


@dataclass
class ComparisonResult:
    q_mods: int
    n: int

    qerr_prior: float
    qerr_teacher: float
    qerr_v1: float
    qerr_v2: float

    mae_prior: float
    mae_teacher: float
    mae_v1: float
    mae_v2: float

    sel_err_prior: float
    sel_err_teacher: float
    sel_err_v1: float
    sel_err_v2: float


def _build_cdf_from_boundaries(boundaries: List[float]):
    b = len(boundaries) - 1
    cdf_x = list(boundaries)
    cdf_p = [i / b for i in range(b + 1)]
    def fn(v: float) -> float:
        return evaluate_piecewise_cdf(cdf_x, cdf_p, v)
    return fn


def quantile_mae(pred_boundaries: List[float], true_boundaries: List[float]) -> float:
    inner_pred = pred_boundaries[1:-1]
    inner_true = true_boundaries[1:-1]
    if not inner_pred:
        return 0.0
    return sum(abs(p - t) for p, t in zip(inner_pred, inner_true)) / len(inner_pred)


def selectivity_error(pred_boundaries: List[float], true_boundaries: List[float], rng: random.Random, n_probes: int = 50) -> float:
    est_fn = _build_cdf_from_boundaries(pred_boundaries)
    act_fn = _build_cdf_from_boundaries(true_boundaries)
    total = 0.0
    for _ in range(n_probes):
        v = rng.uniform(0.0, 1.0)
        total += abs(est_fn(v) - act_fn(v))
    return total / n_probes


def q_error(pred_boundaries: List[float], true_boundaries: List[float], rng: random.Random, n_probes: int = 50, epsilon: float = 1e-6) -> float:
    est_fn = _build_cdf_from_boundaries(pred_boundaries)
    act_fn = _build_cdf_from_boundaries(true_boundaries)
    total = 0.0
    for _ in range(n_probes):
        v = rng.uniform(0.05, 0.95)
        est = max(est_fn(v), epsilon)
        act = max(act_fn(v), epsilon)
        total += max(est / act, act / est)
    return total / n_probes


def evaluate_model(sample: KllFeedbackSample, true_boundaries: List[float], model, max_observations: int, eval_rng: random.Random):
    """Evaluate a single model (v1 or v2) on a sample."""
    if model is None:
        prior_boundaries = [sample.prior.min_value] + list(sample.prior.quantile_values) + [sample.prior.max_value]
        return (
            quantile_mae(prior_boundaries, true_boundaries),
            selectivity_error(prior_boundaries, true_boundaries, eval_rng),
            q_error(prior_boundaries, true_boundaries, eval_rng),
        )

    try:
        record = tensorize_sample(sample, max_observations=max_observations, teacher_fn=None, use_time_decay=False)
        predicted_norm = model.predict([record.feature_tensor])[0]
        val_range = max(sample.prior.value_range, 1e-12)
        model_quantiles = [
            clamp01(sample.prior.min_value + v * val_range)
            for v in predicted_norm
        ]
        # Isotonic projection
        for idx in range(1, len(model_quantiles)):
            if model_quantiles[idx] < model_quantiles[idx - 1]:
                model_quantiles[idx] = model_quantiles[idx - 1]
        bounds = [sample.prior.min_value] + model_quantiles + [sample.prior.max_value]
        return (
            quantile_mae(bounds, true_boundaries),
            selectivity_error(bounds, true_boundaries, eval_rng),
            q_error(bounds, true_boundaries, eval_rng),
        )
    except Exception as e:
        print(f"  Warning: model evaluation failed: {e}")
        prior_boundaries = [sample.prior.min_value] + list(sample.prior.quantile_values) + [sample.prior.max_value]
        return (
            quantile_mae(prior_boundaries, true_boundaries),
            selectivity_error(prior_boundaries, true_boundaries, eval_rng),
            q_error(prior_boundaries, true_boundaries, eval_rng),
        )


def generate_data(output_dir: Path, k: int, num_buckets: int, q_mods: int, seed: int) -> None:
    cmd = [
        sys.executable,
        "simulate_memory_kll_dataset.py",
        "--output-dir", str(output_dir),
        "--k", str(k),
        "--num-buckets", str(num_buckets),
        "--q", str(q_mods),
        "--seed", str(seed),
        "--initial-rows", "5000",
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def train_v1(train_glob: str, model_path: Path, max_observations: int, epochs: int, lr: float):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "train_mlp_model.py",
        "--train-glob", train_glob,
        "--output-model", str(model_path),
        "--max-observations", str(max_observations),
        "--epochs", str(epochs),
        "--lr", str(lr),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return MlpHistogramModel.load(str(model_path))


def train_v2(train_glob: str, model_path: Path, max_observations: int, epochs: int, lr: float, num_heads: int = 3):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "train_mlp_model_v2.py",
        "--train-glob", train_glob,
        "--output-model", str(model_path),
        "--max-observations", str(max_observations),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--num-heads", str(num_heads),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return MlpHistogramModelV2.load(str(model_path))


def run_comparison(
    q_values: List[int],
    k_train: int,
    k_test: int,
    num_buckets: int,
    max_observations: int,
    work_dir: Path,
    seed: int,
    train_q_values: List[int],
    epochs: int = 200,
    lr: float = 3e-3,
) -> List[ComparisonResult]:
    eval_rng = random.Random(seed + 9999)
    results: List[ComparisonResult] = []

    # Generate training data
    q_tag = "_".join(str(q) for q in train_q_values)
    train_dirs_label = f"train_q{q_tag}_k{k_train}"

    train_json_files: List[str] = []
    for tq in train_q_values:
        tdir = work_dir / f"train_q{tq}_k{k_train}"
        if not tdir.exists() or not any(tdir.glob("*.json")):
            print(f"  [训练数据] 生成 q={tq}, k={k_train} ...")
            generate_data(tdir, k=k_train, num_buckets=num_buckets, q_mods=tq, seed=seed + tq)
        train_json_files += [str(p) for p in tdir.glob("*.json")]

    # Merge training data
    merged_dir = work_dir / f"train_merged_{train_dirs_label}"
    if not merged_dir.exists():
        merged_dir.mkdir(parents=True)
        import shutil
        for src in train_json_files:
            dst = merged_dir / Path(src).name
            if not dst.exists():
                shutil.copy2(src, dst)
        print(f"  [训练数据] 合并完成: {len(train_json_files)} 条样本")

    train_glob = str(merged_dir / "*.json")

    # Train v1
    v1_model_path = work_dir / "artifacts" / f"v1_{train_dirs_label}.json"
    if not v1_model_path.exists():
        print(f"  [训练 v1] 单头注意力 MLP (128→64) ...")
        v1_model = train_v1(train_glob, v1_model_path, max_observations, epochs, lr)
    else:
        print(f"  [复用 v1] {v1_model_path.name}")
        v1_model = MlpHistogramModel.load(str(v1_model_path))

    # Train v2
    v2_model_path = work_dir / "artifacts" / f"v2_{train_dirs_label}.json"
    if not v2_model_path.exists():
        print(f"  [训练 v2] 多头注意力 MLP (3 heads, 128→128→64→64) ...")
        v2_model = train_v2(train_glob, v2_model_path, max_observations, epochs, lr, num_heads=3)
    else:
        print(f"  [复用 v2] {v2_model_path.name}")
        v2_model = MlpHistogramModelV2.load(str(v2_model_path))

    # Evaluate on each q
    for q in q_values:
        print(f"\n========== q={q} (漂移强度) ==========")
        test_dir = work_dir / f"test_q{q}"

        print(f"  生成 {k_test} 条测试样本 ...")
        generate_data(test_dir, k=k_test, num_buckets=num_buckets, q_mods=q, seed=seed + 10000 + q)

        mae_p_list, mae_t_list, mae_v1_list, mae_v2_list = [], [], [], []
        sel_p_list, sel_t_list, sel_v1_list, sel_v2_list = [], [], [], []
        qerr_p_list, qerr_t_list, qerr_v1_list, qerr_v2_list = [], [], [], []

        test_files = sorted(test_dir.glob("*.json"))
        for fpath in test_files:
            data = json.loads(fpath.read_text())
            true_boundaries = data["corrected_kll"]["bucket_boundaries"]
            sample = load_feedback_sample(str(fpath))

            # Prior
            prior_boundaries = [sample.prior.min_value] + list(sample.prior.quantile_values) + [sample.prior.max_value]
            mae_p = quantile_mae(prior_boundaries, true_boundaries)
            sel_p = selectivity_error(prior_boundaries, true_boundaries, eval_rng)
            qerr_p = q_error(prior_boundaries, true_boundaries, eval_rng)

            # Teacher
            teacher_quantiles = correct_quantiles(sample)
            teacher_boundaries = [sample.prior.min_value] + teacher_quantiles + [sample.prior.max_value]
            mae_t = quantile_mae(teacher_boundaries, true_boundaries)
            sel_t = selectivity_error(teacher_boundaries, true_boundaries, eval_rng)
            qerr_t = q_error(teacher_boundaries, true_boundaries, eval_rng)

            # v1
            mae_v1, sel_v1, qerr_v1 = evaluate_model(sample, true_boundaries, v1_model, max_observations, eval_rng)

            # v2
            mae_v2, sel_v2, qerr_v2 = evaluate_model(sample, true_boundaries, v2_model, max_observations, eval_rng)

            mae_p_list.append(mae_p); mae_t_list.append(mae_t); mae_v1_list.append(mae_v1); mae_v2_list.append(mae_v2)
            sel_p_list.append(sel_p); sel_t_list.append(sel_t); sel_v1_list.append(sel_v1); sel_v2_list.append(sel_v2)
            qerr_p_list.append(qerr_p); qerr_t_list.append(qerr_t); qerr_v1_list.append(qerr_v1); qerr_v2_list.append(qerr_v2)

        n = len(test_files)
        def avg(lst): return sum(lst) / max(n, 1)

        result = ComparisonResult(
            q_mods=q, n=n,
            qerr_prior=avg(qerr_p_list), qerr_teacher=avg(qerr_t_list), qerr_v1=avg(qerr_v1_list), qerr_v2=avg(qerr_v2_list),
            mae_prior=avg(mae_p_list), mae_teacher=avg(mae_t_list), mae_v1=avg(mae_v1_list), mae_v2=avg(mae_v2_list),
            sel_err_prior=avg(sel_p_list), sel_err_teacher=avg(sel_t_list), sel_err_v1=avg(sel_v1_list), sel_err_v2=avg(sel_v2_list),
        )
        results.append(result)

        # Print results
        print(f"  样本数: {n}")
        print(f"  {'方法':<15} {'Q-Error':<10} {'MAE':<10} {'Sel-Error':<10}")
        print(f"  {'Prior':<15} {result.qerr_prior:<10.4f} {result.mae_prior:<10.5f} {result.sel_err_prior:<10.5f}")
        print(f"  {'Teacher':<15} {result.qerr_teacher:<10.4f} {result.mae_teacher:<10.5f} {result.sel_err_teacher:<10.5f}")
        print(f"  {'OASIS v1':<15} {result.qerr_v1:<10.4f} {result.mae_v1:<10.5f} {result.sel_err_v1:<10.5f}")
        print(f"  {'OASIS v2':<15} {result.qerr_v2:<10.4f} {result.mae_v2:<10.5f} {result.sel_err_v2:<10.5f}")

        def pct_improve(base, new):
            if base < 1e-9: return 0.0
            return (base - new) / base * 100

        print(f"\n  Q-Error 降低 vs Prior:")
        print(f"    v1: {pct_improve(result.qerr_prior, result.qerr_v1):.1f}%")
        print(f"    v2: {pct_improve(result.qerr_prior, result.qerr_v2):.1f}%")
        print(f"  v2 vs v1 提升: {pct_improve(result.qerr_v1, result.qerr_v2):.1f}%")

    return results


def save_comparison_csv(results: List[ComparisonResult], output_path: Path) -> None:
    lines = [
        "q_mods,n,qerr_prior,qerr_teacher,qerr_v1,qerr_v2,mae_prior,mae_teacher,mae_v1,mae_v2,sel_err_prior,sel_err_teacher,sel_err_v1,sel_err_v2"
    ]
    for r in results:
        lines.append(
            f"{r.q_mods},{r.n},"
            f"{r.qerr_prior:.6f},{r.qerr_teacher:.6f},{r.qerr_v1:.6f},{r.qerr_v2:.6f},"
            f"{r.mae_prior:.6f},{r.mae_teacher:.6f},{r.mae_v1:.6f},{r.mae_v2:.6f},"
            f"{r.sel_err_prior:.6f},{r.sel_err_teacher:.6f},{r.sel_err_v1:.6f},{r.sel_err_v2:.6f}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结果已保存到 {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare OASIS v1 vs v2")
    parser.add_argument("--q-values", nargs="+", type=int, default=[10, 15, 20, 25, 30], help="Test drift intensities")
    parser.add_argument("--train-q-values", nargs="+", type=int, default=[10, 20], help="Training drift intensities")
    parser.add_argument("--k-train", type=int, default=1000, help="Training samples per q")
    parser.add_argument("--k-test", type=int, default=128, help="Test samples per q")
    parser.add_argument("--num-buckets", type=int, default=10)
    parser.add_argument("--max-observations", type=int, default=16)
    parser.add_argument("--work-dir", type=Path, default=Path("comparison_work"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OASIS v1 vs v2 对比实验")
    print(f"训练集: q={args.train_q_values}, k={args.k_train}")
    print(f"测试集: q={args.q_values}, k={args.k_test}")
    print(f"工作目录: {work_dir.resolve()}")
    print("=" * 60)

    results = run_comparison(
        q_values=args.q_values,
        k_train=args.k_train,
        k_test=args.k_test,
        num_buckets=args.num_buckets,
        max_observations=args.max_observations,
        work_dir=work_dir,
        seed=args.seed,
        train_q_values=args.train_q_values,
        epochs=args.epochs,
        lr=args.lr,
    )

    csv_path = work_dir / "comparison_results.csv"
    save_comparison_csv(results, csv_path)

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'q':>4} {'QErr_Prior':>11} {'QErr_Teacher':>13} {'QErr_v1':>10} {'QErr_v2':>10} {'v2 vs v1':>10}")
    print("-" * 80)
    for r in results:
        improve = (r.qerr_v1 - r.qerr_v2) / r.qerr_v1 * 100 if r.qerr_v1 > 1e-9 else 0.0
        print(f"{r.q_mods:>4} {r.qerr_prior:>11.4f} {r.qerr_teacher:>13.4f} {r.qerr_v1:>10.4f} {r.qerr_v2:>10.4f} {improve:>9.1f}%")


if __name__ == "__main__":
    main()
