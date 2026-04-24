#!/usr/bin/env python3
"""
综合实验脚本：训练多个模型并进行完整消融实验
=================================================

实验设计：
1. 训练三个模型变体：
   - Model A: 训练于 q={10,20} (论文当前配置)
   - Model B: 训练于 q={1,3,5} (低漂移优化)
   - Model C: 训练于 q={1,3,5,10,15,20} (全范围)

2. 在 q={1,3,5,10,15,20,25,30} 上测试所有模型

3. 对比四种方法：
   - Stale Prior (baseline)
   - Analytical Baseline (Teacher)
   - STHoles (classical self-tuning)
   - OASIS v2 (三个训练变体)

用法：
    python3 run_comprehensive_experiments.py
    python3 run_comprehensive_experiments.py --quick  # 快速测试
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

# 添加 pipeline 路径
_SCRIPT_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _SCRIPT_DIR.parent / "cdf_kll_ml_pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from cdf_teacher import correct_quantiles
from histogram_math import clamp01, evaluate_piecewise_cdf
from histogram_types import KllFeedbackSample
from json_histogram_parser import load_feedback_sample
from mlp_histogram_model_v2 import MlpHistogramModelV2
from tensorizer import tensorize_sample

# STHoles 实现
try:
    from baselines import STHolesCorrector
except ImportError:
    STHolesCorrector = None


@dataclass
class ExperimentResult:
    """单个测试样本的结果"""
    q_mods: int
    model_name: str
    case_id: str
    obs_count: int
    qerror: float
    sel_error: float
    quantile_mae: float


@dataclass
class GroupSummary:
    """按 (q, model) 分组的汇总统计"""
    q_mods: int
    model_name: str
    n_samples: int
    qerror_mean: float
    qerror_std: float
    sel_error_mean: float
    quantile_mae_mean: float
    improvement_vs_prior: float  # 相对于 Prior 的改进百分比


# ============================================================================
# 指标计算
# ============================================================================

def _build_cdf(boundaries: List[float]) -> Tuple[List[float], List[float]]:
    b = len(boundaries) - 1
    return list(boundaries), [i / b for i in range(b + 1)]


def _cdf_fn(boundaries: List[float]):
    cdf_x, cdf_p = _build_cdf(boundaries)
    return lambda v: evaluate_piecewise_cdf(cdf_x, cdf_p, v)


def quantile_mae(pred: List[float], true: List[float]) -> float:
    """分位点平均绝对误差（仅内部分位点）"""
    inner_pred, inner_true = pred[1:-1], true[1:-1]
    if not inner_pred:
        return 0.0
    return sum(abs(p - t) for p, t in zip(inner_pred, inner_true)) / len(inner_pred)


def selectivity_error(pred: List[float], true: List[float], rng: random.Random, n: int = 50) -> float:
    """选择率平均绝对误差"""
    est_fn, act_fn = _cdf_fn(pred), _cdf_fn(true)
    errors = []
    for _ in range(n):
        v = rng.uniform(0, 1)
        errors.append(abs(est_fn(v) - act_fn(v)))
    return sum(errors) / len(errors)


def q_error(pred: List[float], true: List[float], rng: random.Random, n: int = 50, eps: float = 1e-6) -> float:
    """Q-Error: max(est/act, act/est)"""
    est_fn, act_fn = _cdf_fn(pred), _cdf_fn(true)
    errors = []
    for _ in range(n):
        v = rng.uniform(0.05, 0.95)
        est = max(est_fn(v), eps)
        act = max(act_fn(v), eps)
        errors.append(max(est / act, act / est))
    return sum(errors) / len(errors)


# ============================================================================
# 数据生成
# ============================================================================

def generate_dataset(output_dir: Path, k: int, num_buckets: int, q_mods: int, seed: int) -> None:
    """生成合成数据集"""
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


# ============================================================================
# 模型训练
# ============================================================================

def train_oasis_model(
    train_q_values: List[int],
    k_per_q: int,
    num_buckets: int,
    max_obs: int,
    work_dir: Path,
    model_name: str,
    seed: int,
) -> MlpHistogramModelV2:
    """训练 OASIS v2 模型"""
    print(f"\n  [训练 {model_name}] q={train_q_values}, k={k_per_q} per q")

    # 生成训练数据
    train_dir = work_dir / f"train_{model_name}"
    train_dir.mkdir(parents=True, exist_ok=True)

    for q in train_q_values:
        q_dir = train_dir / f"q{q}"
        if not q_dir.exists():
            print(f"    生成 q={q} 训练数据...")
            generate_dataset(q_dir, k=k_per_q, num_buckets=num_buckets, q_mods=q, seed=seed + q)

    # 加载所有训练样本
    features, targets = [], []
    for q in train_q_values:
        q_dir = train_dir / f"q{q}"
        for fpath in sorted(q_dir.glob("*.json")):
            sample = load_feedback_sample(str(fpath))
            record = tensorize_sample(sample, max_observations=max_obs, teacher_fn=correct_quantiles, use_time_decay=False)
            if record.target_tensor is not None:
                features.append(record.feature_tensor)
                targets.append(record.target_tensor)

    print(f"    训练样本总数: {len(features)}")

    # 训练模型
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

    # 保存模型
    model_path = work_dir / "models" / f"{model_name}.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path), metadata={
        "model_name": model_name,
        "train_q_values": train_q_values,
        "k_per_q": k_per_q,
        "max_observations": max_obs,
    })
    print(f"    模型已保存: {model_path.name}")

    return model


# ============================================================================
# 评估
# ============================================================================

def evaluate_sample(
    sample: KllFeedbackSample,
    true_boundaries: List[float],
    max_obs: int,
    rng: random.Random,
    models: dict,
) -> dict:
    """评估单个样本，返回所有方法的指标"""
    prior_b = [sample.prior.min_value] + list(sample.prior.quantile_values) + [sample.prior.max_value]

    results = {}

    # 1. Stale Prior
    results["Prior"] = {
        "qerror": q_error(prior_b, true_boundaries, rng),
        "sel_error": selectivity_error(prior_b, true_boundaries, rng),
        "quantile_mae": quantile_mae(prior_b, true_boundaries),
    }

    # 2. Analytical Baseline (Teacher)
    teacher_q = correct_quantiles(sample)
    teacher_b = [sample.prior.min_value] + teacher_q + [sample.prior.max_value]
    results["Teacher"] = {
        "qerror": q_error(teacher_b, true_boundaries, rng),
        "sel_error": selectivity_error(teacher_b, true_boundaries, rng),
        "quantile_mae": quantile_mae(teacher_b, true_boundaries),
    }

    # 3. STHoles
    if STHolesCorrector is not None:
        try:
            stholes = STHolesCorrector(num_buckets=len(sample.prior.quantile_values) + 1)
            stholes_q = stholes.correct(sample)
            stholes_b = [sample.prior.min_value] + stholes_q + [sample.prior.max_value]
            results["STHoles"] = {
                "qerror": q_error(stholes_b, true_boundaries, rng),
                "sel_error": selectivity_error(stholes_b, true_boundaries, rng),
                "quantile_mae": quantile_mae(stholes_b, true_boundaries),
            }
        except Exception:
            results["STHoles"] = results["Prior"]  # fallback

    # 4. OASIS 模型变体
    for model_name, model in models.items():
        try:
            record = tensorize_sample(sample, max_observations=max_obs, teacher_fn=None, use_time_decay=False)
            pred_norm = model.predict([record.feature_tensor])[0]
            vr = max(sample.prior.value_range, 1e-12)
            model_q = [clamp01(sample.prior.min_value + v * vr) for v in pred_norm]
            # 单调性修正
            for i in range(1, len(model_q)):
                if model_q[i] < model_q[i - 1]:
                    model_q[i] = model_q[i - 1]
            model_b = [sample.prior.min_value] + model_q + [sample.prior.max_value]
            results[model_name] = {
                "qerror": q_error(model_b, true_boundaries, rng),
                "sel_error": selectivity_error(model_b, true_boundaries, rng),
                "quantile_mae": quantile_mae(model_b, true_boundaries),
            }
        except Exception as e:
            print(f"      警告: {model_name} 预测失败: {e}")
            results[model_name] = results["Prior"]  # fallback

    return results


def run_evaluation(
    test_q_values: List[int],
    k_test: int,
    num_buckets: int,
    max_obs: int,
    work_dir: Path,
    models: dict,
    seed: int,
) -> List[ExperimentResult]:
    """在所有测试 q 值上评估所有模型"""
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

            metrics = evaluate_sample(sample, true_b, max_obs, eval_rng, models)

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


# ============================================================================
# 汇总与输出
# ============================================================================

def summarize_results(results: List[ExperimentResult]) -> List[GroupSummary]:
    """按 (q, model) 分组汇总"""
    from collections import defaultdict
    import math

    groups = defaultdict(list)
    for r in results:
        groups[(r.q_mods, r.model_name)].append(r)

    summaries = []
    prior_qerrors = {}  # 存储每个 q 的 Prior Q-Error

    # 先收集 Prior 的 Q-Error
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
    """保存结果为 JSON"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(s) for s in summaries]
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\n  结果已保存: {output_path}")


def print_summary_table(summaries: List[GroupSummary]) -> None:
    """打印汇总表格"""
    print("\n" + "=" * 100)
    print(f"  {'q':>3}  {'Model':<20}  {'Q-Error':>10}  {'±std':>8}  {'vs Prior':>10}  {'Sel-Err':>10}  {'Q-MAE':>10}")
    print("  " + "-" * 96)

    current_q = None
    for s in summaries:
        if s.q_mods != current_q:
            if current_q is not None:
                print("  " + "-" * 96)
            current_q = s.q_mods

        improvement_str = f"{s.improvement_vs_prior:+.1f}%" if s.model_name != "Prior" else "—"
        print(f"  {s.q_mods:>3}  {s.model_name:<20}  {s.qerror_mean:>10.4f}  {s.qerror_std:>8.4f}  "
              f"{improvement_str:>10}  {s.sel_error_mean:>10.4f}  {s.quantile_mae_mean:>10.4f}")
    print("=" * 100)


# ============================================================================
# 主流程
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="综合实验：多模型训练与评估")
    parser.add_argument("--quick", action="store_true", help="快速测试模式（小数据集）")
    parser.add_argument("--work-dir", type=Path, default=Path("work_comprehensive"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        k_train, k_test = 200, 50
        test_q_values = [1, 3, 5, 10, 15, 20]
    else:
        k_train, k_test = 1000, 128
        test_q_values = [1, 3, 5, 10, 15, 20, 25, 30]

    num_buckets = 10
    max_obs = 16
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("  OASIS 综合实验：多训练配置对比")
    print(f"  工作目录: {work_dir.resolve()}")
    print(f"  训练集: {k_train} 样本/q   测试集: {k_test} 样本/q")
    print(f"  测试 q 值: {test_q_values}")
    print("=" * 100)

    # 训练三个模型变体
    models = {}

    print("\n[阶段 1/3] 训练模型变体")
    print("-" * 100)

    # Model A: 论文当前配置 (q=10,20)
    models["OASIS_A_q10_20"] = train_oasis_model(
        train_q_values=[10, 20],
        k_per_q=k_train,
        num_buckets=num_buckets,
        max_obs=max_obs,
        work_dir=work_dir,
        model_name="OASIS_A_q10_20",
        seed=args.seed,
    )

    # Model B: 低漂移优化 (q=1,3,5)
    models["OASIS_B_q1_3_5"] = train_oasis_model(
        train_q_values=[1, 3, 5],
        k_per_q=k_train,
        num_buckets=num_buckets,
        max_obs=max_obs,
        work_dir=work_dir,
        model_name="OASIS_B_q1_3_5",
        seed=args.seed + 100,
    )

    # Model C: 全范围 (q=1,3,5,10,15,20)
    models["OASIS_C_q1_to_20"] = train_oasis_model(
        train_q_values=[1, 3, 5, 10, 15, 20],
        k_per_q=k_train,  # 每个 q 1000 样本
        num_buckets=num_buckets,
        max_obs=max_obs,
        work_dir=work_dir,
        model_name="OASIS_C_q1_to_20",
        seed=args.seed + 200,
    )

    # 评估
    print("\n[阶段 2/3] 评估所有模型")
    print("-" * 100)

    results = run_evaluation(
        test_q_values=test_q_values,
        k_test=k_test,
        num_buckets=num_buckets,
        max_obs=max_obs,
        work_dir=work_dir,
        models=models,
        seed=args.seed,
    )

    # 汇总
    print("\n[阶段 3/3] 汇总结果")
    print("-" * 100)

    summaries = summarize_results(results)
    print_summary_table(summaries)

    # 保存
    output_path = work_dir / "results" / "comprehensive_results.json"
    save_results(summaries, output_path)

    # 生成 LaTeX 表格
    generate_latex_table(summaries, work_dir / "results" / "table_comprehensive.tex")

    print("\n✓ 实验完成！")


def generate_latex_table(summaries: List[GroupSummary], output_path: Path) -> None:
    """生成 LaTeX 表格"""
    lines = [
        "% 综合实验结果表格",
        "\\begin{table}[!htb]",
        "  \\centering",
        "  \\caption{Q-Error comparison across training configurations and drift intensities.}",
        "  \\label{tab:comprehensive}",
        "  \\begin{tabular}{c lrrr}",
        "    \\toprule",
        "    $q$ & Method & Q-Error & vs Prior & Sel-Error \\\\",
        "    \\midrule",
    ]

    current_q = None
    for s in summaries:
        if s.q_mods != current_q:
            if current_q is not None:
                lines.append("    \\midrule")
            current_q = s.q_mods

        improvement_str = f"{s.improvement_vs_prior:+.1f}\\%" if s.model_name != "Prior" else "—"
        lines.append(f"    {s.q_mods} & {s.model_name:<20} & {s.qerror_mean:.4f} & {improvement_str:>10} & {s.sel_error_mean:.4f} \\\\")

    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"  LaTeX 表格已保存: {output_path}")


if __name__ == "__main__":
    main()
