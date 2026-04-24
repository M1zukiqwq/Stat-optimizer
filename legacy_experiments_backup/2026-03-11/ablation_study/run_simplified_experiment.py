#!/usr/bin/env python3
"""
简化实验：仅对比 Prior vs OASIS
删除 Teacher 和其他变体
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _SCRIPT_DIR.parent / "cdf_kll_ml_pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from histogram_math import clamp01, evaluate_piecewise_cdf
from histogram_types import KllFeedbackSample
from json_histogram_parser import load_feedback_sample
from mlp_histogram_model_v2 import MlpHistogramModelV2
from tensorizer import tensorize_sample


@dataclass
class ExperimentResult:
    q_mods: int
    model_name: str
    case_id: str
    obs_count: int
    qerror: float
    sel_error: float
    quantile_mae: float


@dataclass
class GroupSummary:
    q_mods: int
    model_name: str
    n_samples: int
    qerror_mean: float
    qerror_std: float
    sel_error_mean: float
    quantile_mae_mean: float
    improvement_vs_prior: float


def _build_cdf(boundaries: List[float]) -> Tuple[List[float], List[float]]:
    b = len(boundaries) - 1
    return list(boundaries), [i / b for i in range(b + 1)]


def _cdf_fn(boundaries: List[float]):
    cdf_x, cdf_p = _build_cdf(boundaries)
    return lambda v: evaluate_piecewise_cdf(cdf_x, cdf_p, v)


def quantile_mae(pred: List[float], true: List[float]) -> float:
    inner_pred, inner_true = pred[1:-1], true[1:-1]
    if not inner_pred:
        return 0.0
    return sum(abs(p - t) for p, t in zip(inner_pred, inner_true)) / len(inner_pred)


def selectivity_error(pred: List[float], true: List[float], rng: random.Random, n: int = 50) -> float:
    est_fn, act_fn = _cdf_fn(pred), _cdf_fn(true)
    errors = []
    for _ in range(n):
        v = rng.uniform(0, 1)
        errors.append(abs(est_fn(v) - act_fn(v)))
    return sum(errors) / len(errors)


def q_error(pred: List[float], true: List[float], rng: random.Random, n: int = 50, eps: float = 1e-6) -> float:
    est_fn, act_fn = _cdf_fn(pred), _cdf_fn(true)
    errors = []
    for _ in range(n):
        v = rng.uniform(0.05, 0.95)
        est = max(est_fn(v), eps)
        act = max(act_fn(v), eps)
        errors.append(max(est / act, act / est))
    return sum(errors) / len(errors)


def generate_dataset(output_dir: Path, k: int, num_buckets: int, q_mods: int, seed: int) -> None:
    script_path = _PIPELINE_DIR / "simulate_memory_kll_dataset.py"
    subprocess.run([
        sys.executable, str(script_path),
        "--output-dir", str(output_dir),
        "--k", str(k),
        "--num-buckets", str(num_buckets),
        "--q", str(q_mods),
        "--seed", str(seed),
        "--initial-rows", "5000",
    ], check=True, capture_output=True)


def train_oasis_model(
    train_q_values: List[int],
    k_per_q: int,
    num_buckets: int,
    max_obs: int,
    work_dir: Path,
    seed: int,
) -> MlpHistogramModelV2:
    print(f"\n  [训练 OASIS] q={train_q_values}, k={k_per_q} per q")

    train_dir = work_dir / "train_oasis"
    train_dir.mkdir(parents=True, exist_ok=True)

    for q in train_q_values:
        q_dir = train_dir / f"q{q}"
        if not q_dir.exists():
            print(f"    生成 q={q} 训练数据...")
            generate_dataset(q_dir, k=k_per_q, num_buckets=num_buckets, q_mods=q, seed=seed + q)

    features, targets = [], []
    for q in train_q_values:
        q_dir = train_dir / f"q{q}"
        for fpath in sorted(q_dir.glob("*.json")):
            sample = load_feedback_sample(str(fpath))
            record = tensorize_sample(sample, max_observations=max_obs, teacher_fn=None, use_time_decay=False)
            if record.target_tensor is not None:
                features.append(record.feature_tensor)
                targets.append(record.target_tensor)

    print(f"    训练样本总数: {len(features)}")

    obs_dim = 12
    prior_dim = len(targets[0])
    model = MlpHistogramModelV2(
        obs_dim=obs_dim,
        prior_dim=prior_dim,
        meta_dim=3,
        max_observations=max_obs,
        num_heads=3,
        hidden_dims=(128, 128, 64, 64),
        prior_encoder_dim=32,
        alpha=1e-4,
        lr=3e-3,
        epochs=200,
        batch_size=32,
        seed=seed,
    )
    model.fit(features, targets)

    model_path = work_dir / "models" / "oasis_model.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path), metadata={
        "train_q_values": train_q_values,
        "k_per_q": k_per_q,
        "max_observations": max_obs,
    })
    print(f"    模型已保存: {model_path.name}")

    return model


def evaluate_sample(
    sample: KllFeedbackSample,
    true_boundaries: List[float],
    max_obs: int,
    rng: random.Random,
    model: MlpHistogramModelV2,
) -> dict:
    prior_b = [sample.prior.min_value] + list(sample.prior.quantile_values) + [sample.prior.max_value]

    results = {}

    # 1. Stale Prior
    results["Prior"] = {
        "qerror": q_error(prior_b, true_boundaries, rng),
        "sel_error": selectivity_error(prior_b, true_boundaries, rng),
        "quantile_mae": quantile_mae(prior_b, true_boundaries),
    }

    # 2. OASIS
    try:
        record = tensorize_sample(sample, max_observations=max_obs, teacher_fn=None, use_time_decay=False)
        pred_norm = model.predict([record.feature_tensor])[0]
        vr = max(sample.prior.value_range, 1e-12)
        model_q = [clamp01(sample.prior.min_value + v * vr) for v in pred_norm]
        for i in range(1, len(model_q)):
            if model_q[i] < model_q[i - 1]:
                model_q[i] = model_q[i - 1]
        model_b = [sample.prior.min_value] + model_q + [sample.prior.max_value]
        results["OASIS"] = {
            "qerror": q_error(model_b, true_boundaries, rng),
            "sel_error": selectivity_error(model_b, true_boundaries, rng),
            "quantile_mae": quantile_mae(model_b, true_boundaries),
        }
    except Exception as e:
        print(f"      警告: OASIS 预测失败: {e}")
        results["OASIS"] = results["Prior"]

    return results


def run_evaluation(
    test_q_values: List[int],
    k_test: int,
    num_buckets: int,
    max_obs: int,
    work_dir: Path,
    model: MlpHistogramModelV2,
    seed: int,
) -> List[ExperimentResult]:
    eval_rng = random.Random(seed + 9999)
    all_results = []

    for q in test_q_values:
        print(f"\n  ── 测试 q={q} ──")
        test_dir = work_dir / f"test_q{q}"
        if not test_dir.exists():
            print(f"    生成测试数据...")
            generate_dataset(test_dir, k=k_test, num_buckets=num_buckets, q_mods=q, seed=seed + 1000 + q)

        for fpath in sorted(test_dir.glob("*.json")):
            data = json.loads(fpath.read_text())
            sample = load_feedback_sample(str(fpath))
            true_b = data["corrected_kll"]["bucket_boundaries"]

            metrics = evaluate_sample(sample, true_b, max_obs, eval_rng, model)

            for model_name, m in metrics.items():
                all_results.append(ExperimentResult(
                    q_mods=q,
                    model_name=model_name,
                    case_id=fpath.stem,
                    obs_count=len(data["observations"]),
                    qerror=m["qerror"],
                    sel_error=m["sel_error"],
                    quantile_mae=m["quantile_mae"],
                ))

    return all_results


def summarize_results(results: List[ExperimentResult]) -> List[GroupSummary]:
    from collections import defaultdict
    import math

    groups = defaultdict(list)
    for r in results:
        groups[(r.q_mods, r.model_name)].append(r)

    summaries = []
    prior_qerrors = {}

    for (q, model_name), items in groups.items():
        if model_name == "Prior":
            prior_qerrors[q] = sum(r.qerror for r in items) / len(items)

    for (q, model_name), items in sorted(groups.items()):
        n = len(items)
        qerrors = [r.qerror for r in items]
        mean_qerr = sum(qerrors) / n
        std_qerr = math.sqrt(sum((x - mean_qerr) ** 2 for x in qerrors) / n) if n > 1 else 0.0

        prior_qerr = prior_qerrors.get(q, mean_qerr)
        improvement = (prior_qerr - mean_qerr) / prior_qerr * 100 if prior_qerr > 0 else 0.0

        summaries.append(GroupSummary(
            q_mods=q,
            model_name=model_name,
            n_samples=n,
            qerror_mean=mean_qerr,
            qerror_std=std_qerr,
            sel_error_mean=sum(r.sel_error for r in items) / n,
            quantile_mae_mean=sum(r.quantile_mae for r in items) / n,
            improvement_vs_prior=improvement,
        ))

    return summaries


def save_results(summaries: List[GroupSummary], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(s) for s in summaries]
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\n  结果已保存: {output_path}")


def print_summary_table(summaries: List[GroupSummary]) -> None:
    print("\n" + "=" * 80)
    print(f"  {'q':>3}  {'Model':<10}  {'Q-Error':>10}  {'±std':>8}  {'vs Prior':>10}  {'Sel-Err':>10}")
    print("  " + "-" * 76)

    current_q = None
    for s in summaries:
        if s.q_mods != current_q:
            if current_q is not None:
                print("  " + "-" * 76)
            current_q = s.q_mods

        improvement_str = f"{s.improvement_vs_prior:+.1f}%" if s.model_name != "Prior" else "—"
        print(f"  {s.q_mods:>3}  {s.model_name:<10}  {s.qerror_mean:>10.4f}  {s.qerror_std:>8.4f}  "
              f"{improvement_str:>10}  {s.sel_error_mean:>10.4f}")
    print("=" * 80)


def generate_latex_table(summaries: List[GroupSummary], output_path: Path) -> None:
    lines = [
        "% 简化实验结果表格（Prior vs OASIS）",
        "\\begin{table}[!htb]",
        "  \\centering",
        "  \\caption{Q-Error comparison: OASIS vs Stale Prior across drift intensities.",
        "           OASIS is trained on $q \\in \\{1,3,5,10,15,20\\}$ (1000 samples per $q$, 6000 total).}",
        "  \\label{tab:qerror}",
        "  \\setlength{\\tabcolsep}{5pt}",
        "  \\begin{tabular}{c rr rr}",
        "    \\toprule",
        "    & \\multicolumn{2}{c}{Stale Prior}",
        "    & \\multicolumn{2}{c}{OASIS (ours)} \\\\",
        "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "    $q$ & Q-Error & — & Q-Error & Improvement \\\\",
        "    \\midrule",
    ]

    prior_data = {}
    oasis_data = {}
    for s in summaries:
        if s.model_name == "Prior":
            prior_data[s.q_mods] = s.qerror_mean
        elif s.model_name == "OASIS":
            oasis_data[s.q_mods] = s.qerror_mean

    for q in sorted(prior_data.keys()):
        prior_qerr = prior_data[q]
        oasis_qerr = oasis_data[q]
        improvement = (prior_qerr - oasis_qerr) / prior_qerr * 100
        lines.append(f"    \\textbf{{{q:2d}}} & {prior_qerr:.3f} & — & \\textbf{{{oasis_qerr:.3f}}} & +{improvement:.1f}\\% \\\\")

    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"  LaTeX 表格已保存: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="简化实验：Prior vs OASIS")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    parser.add_argument("--work-dir", type=Path, default=Path("work_simplified"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        k_train, k_test = 200, 50
        test_q_values = [1, 5, 10, 20]
    else:
        k_train, k_test = 1000, 128
        test_q_values = [1, 3, 5, 10, 15, 20, 25, 30]

    num_buckets = 10
    max_obs = 16
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  简化实验：Prior vs OASIS")
    print(f"  工作目录: {work_dir.resolve()}")
    print(f"  训练集: {k_train} 样本/q   测试集: {k_test} 样本/q")
    print(f"  测试 q 值: {test_q_values}")
    print("=" * 80)

    # 训练OASIS模型
    print("\n[阶段 1/3] 训练OASIS模型")
    print("-" * 80)
    model = train_oasis_model(
        train_q_values=[1, 3, 5, 10, 15, 20],
        k_per_q=k_train,
        num_buckets=num_buckets,
        max_obs=max_obs,
        work_dir=work_dir,
        seed=args.seed,
    )

    # 评估
    print("\n[阶段 2/3] 评估模型")
    print("-" * 80)
    results = run_evaluation(
        test_q_values=test_q_values,
        k_test=k_test,
        num_buckets=num_buckets,
        max_obs=max_obs,
        work_dir=work_dir,
        model=model,
        seed=args.seed,
    )

    # 汇总
    print("\n[阶段 3/3] 汇总结果")
    print("-" * 80)
    summaries = summarize_results(results)
    print_summary_table(summaries)

    # 保存
    output_path = work_dir / "results" / "simplified_results.json"
    save_results(summaries, output_path)

    latex_path = work_dir / "results" / "table_simplified.tex"
    generate_latex_table(summaries, latex_path)

    print("\n✓ 实验完成！")


if __name__ == "__main__":
    main()
