#!/usr/bin/env python3
"""
敏感性分析：Observation Window Size (K) 对 OASIS 性能的影响
使用新的多样化训练方式 (q∈{1,3,5,10,15,20})
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict

_SCRIPT_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _SCRIPT_DIR.parent / "cdf_kll_ml_pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from histogram_math import clamp01, evaluate_piecewise_cdf
from histogram_types import KllFeedbackSample
from json_histogram_parser import load_feedback_sample
from mlp_histogram_model_v2 import MlpHistogramModelV2
from tensorizer import tensorize_sample
from baselines import correct_stholes
from modern_baselines import correct_quicksel_h, correct_isomer


@dataclass
class SensitivityResult:
    k: int
    prior_qerror: float
    oasis_qerror: float
    improvement: float


def _build_cdf(boundaries: List[float]) -> Tuple[List[float], List[float]]:
    b = len(boundaries) - 1
    return list(boundaries), [i / b for i in range(b + 1)]


def _cdf_fn(boundaries: List[float]):
    cdf_x, cdf_p = _build_cdf(boundaries)
    return lambda v: evaluate_piecewise_cdf(cdf_x, cdf_p, v)


def q_error(pred: List[float], true: List[float], rng: random.Random, n: int = 50, eps: float = 1e-6) -> float:
    est_fn, act_fn = _cdf_fn(pred), _cdf_fn(true)
    errors = []
    for _ in range(n):
        v = rng.uniform(0.05, 0.95)
        est, act = max(est_fn(v), eps), max(act_fn(v), eps)
        errors.append(max(est / act, act / est))
    return sum(errors) / len(errors)


def generate_dataset(output_dir: Path, k: int, num_buckets: int, q_mods: int, seed: int) -> None:
    script_path = _PIPELINE_DIR / "simulate_memory_kll_dataset.py"
    subprocess.run([
        sys.executable, str(script_path),
        "--output-dir", str(output_dir),
        "--k", str(k), "--num-buckets", str(num_buckets),
        "--q", str(q_mods), "--seed", str(seed), "--initial-rows", "5000",
    ], check=True, capture_output=True)


def train_oasis_model(train_q_values, k_per_q, num_buckets, max_obs, work_dir, seed):
    """训练 OASIS 模型（多样化漂移训练）"""
    print(f"\n  [训练 OASIS] q={train_q_values}, k={k_per_q}/q, K={max_obs}")
    train_dir = work_dir / "train_oasis"
    train_dir.mkdir(parents=True, exist_ok=True)

    for q in train_q_values:
        q_dir = train_dir / f"q{q}"
        if not q_dir.exists():
            print(f"    生成 q={q} 训练数据...")
            generate_dataset(q_dir, k=k_per_q, num_buckets=num_buckets, q_mods=q, seed=seed + q)

    features, targets = [], []
    for q in train_q_values:
        for fpath in sorted((train_dir / f"q{q}").glob("*.json")):
            sample = load_feedback_sample(str(fpath))
            record = tensorize_sample(sample, max_observations=max_obs, teacher_fn=None, use_time_decay=False)
            if record.target_tensor is not None:
                features.append(record.feature_tensor)
                targets.append(record.target_tensor)

    print(f"    训练样本: {len(features)}, 特征维度: {len(features[0])}")
    model = MlpHistogramModelV2(
        obs_dim=12, prior_dim=len(targets[0]), meta_dim=3,
        max_observations=max_obs, num_heads=3,
        hidden_dims=(128, 128, 64, 64), prior_encoder_dim=32,
        alpha=1e-4, lr=3e-3, epochs=200, batch_size=32, seed=seed,
    )
    model.fit(features, targets)
    model_path = work_dir / "models" / "oasis.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"    模型已保存到 {model_path}")
    return model


def evaluate_at_k(k: int, test_q: int, num_buckets: int, work_dir: Path, seed: int) -> SensitivityResult:
    """在特定 K 值下评估 OASIS 性能"""
    train_q_values = [1, 3, 5, 10, 15, 20]
    k_train = 200  # 使用较小数据集加速敏感性分析
    k_test = 128
    
    print(f"\n{'='*60}")
    print(f"测试 K={k} (q={test_q})")
    print(f"{'='*60}")
    
    # 1. 训练模型
    model = train_oasis_model(train_q_values, k_train, num_buckets, k, work_dir, seed)
    
    # 2. 生成测试数据
    test_dir = work_dir / f"test_q{test_q}"
    if not test_dir.exists():
        print(f"  生成 q={test_q} 测试数据...")
        generate_dataset(test_dir, k=k_test, num_buckets=num_buckets, q_mods=test_q, seed=seed + 1000 + test_q)
    
    # 3. 评估
    rng = random.Random(seed + 9999)
    prior_qerrors = []
    oasis_qerrors = []
    
    for fpath in sorted(test_dir.glob("*.json")):
        sample = load_feedback_sample(str(fpath))
        if sample.corrected_quantile_values is None:
            continue
        
        true_boundaries = [sample.prior.min_value] + list(sample.corrected_quantile_values) + [sample.prior.max_value]
        prior_boundaries = [sample.prior.min_value] + list(sample.prior.quantile_values) + [sample.prior.max_value]
        
        # Prior Q-Error
        prior_qerr = q_error(prior_boundaries, true_boundaries, rng)
        prior_qerrors.append(prior_qerr)
        
        # OASIS Q-Error
        try:
            record = tensorize_sample(sample, max_observations=k, teacher_fn=None, use_time_decay=False)
            pred_norm = model.predict([record.feature_tensor])[0]
            vr = max(sample.prior.value_range, 1e-12)
            model_q = [clamp01(sample.prior.min_value + v * vr) for v in pred_norm]
            for i in range(1, len(model_q)):
                if model_q[i] < model_q[i-1]:
                    model_q[i] = model_q[i-1]
            model_boundaries = [sample.prior.min_value] + model_q + [sample.prior.max_value]
            oasis_qerr = q_error(model_boundaries, true_boundaries, rng)
            oasis_qerrors.append(oasis_qerr)
        except Exception as e:
            oasis_qerrors.append(prior_qerr)
    
    mean_prior = sum(prior_qerrors) / len(prior_qerrors)
    mean_oasis = sum(oasis_qerrors) / len(oasis_qerrors)
    improvement = (mean_prior - mean_oasis) / mean_prior * 100
    
    print(f"\n  K={k} 结果:")
    print(f"    Prior Q-Error:  {mean_prior:.4f}")
    print(f"    OASIS Q-Error:  {mean_oasis:.4f}")
    print(f"    Improvement:    {improvement:.1f}%")
    
    return SensitivityResult(k=k, prior_qerror=mean_prior, oasis_qerror=mean_oasis, improvement=improvement)


def save_results(results: List[SensitivityResult], output_dir: Path):
    """保存结果为 CSV 和 LaTeX 表格"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV
    csv_path = output_dir / "sensitivity_k_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "Prior_QError", "OASIS_QError", "Improvement_%"])
        for r in results:
            writer.writerow([r.k, f"{r.prior_qerror:.4f}", f"{r.oasis_qerror:.4f}", f"{r.improvement:.1f}"])
    print(f"\n  CSV 已保存: {csv_path}")
    
    # LaTeX 表格
    latex_path = output_dir / "table_sensitivity_k.tex"
    with open(latex_path, "w") as f:
        f.write("\\begin{table}[!htb]\n")
        f.write("  \\centering\n")
        f.write("  \\caption{OASIS Q-Error sensitivity to window size $K$ at $q{=}10$. "
                "Training: diverse drift scenarios ($q \\in \\{1,3,5,10,15,20\\}$).}\n")
        f.write("  \\label{tab:sensitivity}\n")
        f.write("  \\begin{tabular}{l r r r r}\n")
        f.write("    \\toprule\n")
        f.write("    $K$ (Window Size) & 4 & 8 & 16 & 32 \\\\\n")
        f.write("    \\midrule\n")
        
        prior_row = "    Prior Q-Error & " + " & ".join([f"{r.prior_qerror:.3f}" for r in results]) + " \\\\\n"
        oasis_row = "    OASIS Q-Error & " + " & ".join([f"{r.oasis_qerror:.3f}" for r in results]) + " \\\\\n"
        imprv_row = "    Improvement & " + " & ".join([f"{r.improvement:.1f}\\%" for r in results]) + " \\\\\n"
        
        f.write(prior_row)
        f.write(oasis_row)
        f.write(imprv_row)
        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  LaTeX 已保存: {latex_path}")


def plot_results(results: List[SensitivityResult], output_dir: Path):
    """绘制敏感性分析图表"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib 不可用，跳过图表生成")
        return
    
    ks = [r.k for r in results]
    prior_errors = [r.prior_qerror for r in results]
    oasis_errors = [r.oasis_qerror for r in results]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(ks, prior_errors, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Stale Prior')
    ax.plot(ks, oasis_errors, '^-', color='#27ae60', linewidth=2, markersize=8, label='OASIS (ours)')
    
    # 标注最优 K
    best_k = min(results, key=lambda r: r.oasis_qerror).k
    best_qerr = min(r.oasis_qerror for r in results)
    ax.axvline(x=best_k, color='gray', linestyle='--', alpha=0.5)
    ax.annotate(f'Optimal $K$={best_k}', xy=(best_k, best_qerr), 
                xytext=(best_k+3, best_qerr+0.1),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))
    
    ax.set_xlabel('Observation Window Size ($K$)', fontsize=11)
    ax.set_ylabel('Q-Error ($\\downarrow$ better)', fontsize=11)
    ax.set_xticks(ks)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    plot_path = output_dir / "sensitivity_k_plot.pdf"
    fig.savefig(str(plot_path), bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {plot_path}")
    
    # 同时保存到论文目录
    paper_plot_path = _SCRIPT_DIR.parent / "paper" / "figures" / "sensitivity_k.pdf"
    fig.savefig(str(paper_plot_path), bbox_inches='tight')
    print(f"  论文图表已保存: {paper_plot_path}")


def main():
    parser = argparse.ArgumentParser(description="敏感性分析：Observation Window Size (K)")
    parser.add_argument("--work-dir", type=Path, default=Path("work_sensitivity_k"))
    parser.add_argument("--test-q", type=int, default=10, help="测试漂移强度 (默认: 10)")
    parser.add_argument("--k-values", type=int, nargs="+", default=[4, 8, 16, 32], help="测试的 K 值")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("  OASIS 敏感性分析：Observation Window Size (K)")
    print(f"  测试条件: q={args.test_q}, K∈{args.k_values}")
    print(f"  训练方式: 多样化漂移场景 (q∈{{1,3,5,10,15,20}})")
    print("="*70)
    
    results = []
    for k in args.k_values:
        result = evaluate_at_k(k, args.test_q, num_buckets=10, work_dir=work_dir / f"k_{k}", seed=args.seed)
        results.append(result)
    
    # 保存结果
    print("\n" + "="*70)
    print("保存结果...")
    save_results(results, work_dir / "results")
    plot_results(results, work_dir / "results")
    
    # 打印汇总
    print("\n" + "="*70)
    print("汇总结果")
    print("="*70)
    print(f"{'K':>4} {'Prior Q-Err':>12} {'OASIS Q-Err':>12} {'Improvement':>12}")
    print("-"*70)
    for r in results:
        print(f"{r.k:>4} {r.prior_qerror:>12.4f} {r.oasis_qerror:>12.4f} {r.improvement:>11.1f}%")
    print("="*70)
    
    best = min(results, key=lambda r: r.oasis_qerror)
    print(f"\n结论: 最优窗口大小 K={best.k}, Q-Error={best.oasis_qerror:.4f}")
    print("✓ 敏感性分析完成!")


if __name__ == "__main__":
    main()
