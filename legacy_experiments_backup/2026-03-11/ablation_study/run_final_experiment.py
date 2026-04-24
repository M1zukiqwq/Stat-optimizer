#!/usr/bin/env python3
"""
最终实验：Prior vs STHoles vs QuickSel-H vs ISOMER vs OASIS
============================================================
一次性完成所有实验、生成CSV、更新图表和LaTeX表格。
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
    return sum(errors) / n


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
    print(f"\n  [训练 OASIS] q={train_q_values}, k={k_per_q}/q")
    model_path = work_dir / "models" / "oasis_model.json"
    if model_path.exists():
        print(f"    找到现有模型 {model_path}，直接加载...")
        return MlpHistogramModelV2.load(str(model_path))

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

    print(f"    训练样本: {len(features)}")
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
    print(f"    模型已保存")
    return model


def _obs_to_dicts(sample: KllFeedbackSample) -> List[dict]:
    """将sample的observations转换为dict列表"""
    obs_list = []
    for o in sample.observations:
        d = {
            "predicate_type": o.predicate_type,
            "value": o.value,
            "actual_sel": o.actual_selectivity,
            "estimated_sel": o.estimated_selectivity,
        }
        if o.value_upper is not None:
            d["value_upper"] = o.value_upper
        obs_list.append(d)
    return obs_list


def evaluate_sample(sample, true_boundaries, max_obs, rng, model, num_buckets):
    prior_b = [sample.prior.min_value] + list(sample.prior.quantile_values) + [sample.prior.max_value]
    obs_dicts = _obs_to_dicts(sample)
    results = {}

    # 1. Prior
    results["Prior"] = {
        "qerror": q_error(prior_b, true_boundaries, rng),
        "sel_error": selectivity_error(prior_b, true_boundaries, rng),
        "quantile_mae": quantile_mae(prior_b, true_boundaries),
    }

    # 2. STHoles
    try:
        sth_q = correct_stholes(
            sample.prior.min_value, sample.prior.max_value,
            list(sample.prior.quantile_values), obs_dicts, num_buckets=num_buckets)
        sth_b = [sample.prior.min_value] + list(sth_q) + [sample.prior.max_value]
        results["STHoles"] = {
            "qerror": q_error(sth_b, true_boundaries, rng),
            "sel_error": selectivity_error(sth_b, true_boundaries, rng),
            "quantile_mae": quantile_mae(sth_b, true_boundaries),
        }
    except Exception:
        results["STHoles"] = results["Prior"].copy()

    # 3. QuickSel-H
    try:
        qs_q = correct_quicksel_h(
            sample.prior.min_value, sample.prior.max_value,
            list(sample.prior.quantile_values), obs_dicts, num_buckets=num_buckets)
        qs_b = [sample.prior.min_value] + list(qs_q) + [sample.prior.max_value]
        results["QuickSel-H"] = {
            "qerror": q_error(qs_b, true_boundaries, rng),
            "sel_error": selectivity_error(qs_b, true_boundaries, rng),
            "quantile_mae": quantile_mae(qs_b, true_boundaries),
        }
    except Exception:
        results["QuickSel-H"] = results["Prior"].copy()

    # 4. ISOMER
    try:
        iso_q = correct_isomer(
            sample.prior.min_value, sample.prior.max_value,
            list(sample.prior.quantile_values), obs_dicts, num_buckets=num_buckets)
        iso_b = [sample.prior.min_value] + list(iso_q) + [sample.prior.max_value]
        results["ISOMER"] = {
            "qerror": q_error(iso_b, true_boundaries, rng),
            "sel_error": selectivity_error(iso_b, true_boundaries, rng),
            "quantile_mae": quantile_mae(iso_b, true_boundaries),
        }
    except Exception:
        results["ISOMER"] = results["Prior"].copy()

    # 5. OASIS
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
    except Exception:
        results["OASIS"] = results["Prior"].copy()

    return results


def run_evaluation(test_q_values, k_test, num_buckets, max_obs, work_dir, model, seed):
    all_results = []
    rng = random.Random(seed + 9999)

    for q in test_q_values:
        test_dir = work_dir / f"test_q{q}"
        if not test_dir.exists():
            print(f"    生成 q={q} 测试数据...")
            generate_dataset(test_dir, k=k_test, num_buckets=num_buckets, q_mods=q, seed=seed + 1000 + q)

        # 按方法收集结果
        method_results: Dict[str, List[dict]] = {}
        for fpath in sorted(test_dir.glob("*.json")):
            sample = load_feedback_sample(str(fpath))
            if sample.corrected_quantile_values is None:
                continue
            true_b = [sample.prior.min_value] + list(sample.corrected_quantile_values) + [sample.prior.max_value]
            res = evaluate_sample(sample, true_b, max_obs, rng, model, num_buckets)
            for method, metrics in res.items():
                method_results.setdefault(method, []).append(metrics)

        # 汇总
        prior_qerr = 0
        for method, metrics_list in method_results.items():
            qerrs = [m["qerror"] for m in metrics_list]
            sels = [m["sel_error"] for m in metrics_list]
            maes = [m["quantile_mae"] for m in metrics_list]
            mean_qerr = sum(qerrs) / len(qerrs)
            if method == "Prior":
                prior_qerr = mean_qerr

        for method, metrics_list in method_results.items():
            qerrs = [m["qerror"] for m in metrics_list]
            sels = [m["sel_error"] for m in metrics_list]
            maes = [m["quantile_mae"] for m in metrics_list]
            mean_qerr = sum(qerrs) / len(qerrs)
            std_qerr = (sum((x - mean_qerr) ** 2 for x in qerrs) / len(qerrs)) ** 0.5
            improvement = (prior_qerr - mean_qerr) / prior_qerr * 100 if method != "Prior" else 0.0

            all_results.append(GroupSummary(
                q_mods=q, model_name=method, n_samples=len(metrics_list),
                qerror_mean=mean_qerr, qerror_std=std_qerr,
                sel_error_mean=sum(sels) / len(sels),
                quantile_mae_mean=sum(maes) / len(maes),
                improvement_vs_prior=improvement,
            ))

        print(f"    q={q}: {len(metrics_list)} samples evaluated")

    return all_results


def save_csv(results: List[GroupSummary], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["q_mods", "method", "n", "qerror_mean", "qerror_std",
                         "sel_error_mean", "quantile_mae_mean", "improvement_vs_prior"])
        for r in results:
            writer.writerow([r.q_mods, r.model_name, r.n_samples,
                             f"{r.qerror_mean:.4f}", f"{r.qerror_std:.4f}",
                             f"{r.sel_error_mean:.4f}", f"{r.quantile_mae_mean:.4f}",
                             f"{r.improvement_vs_prior:.1f}"])
    print(f"  CSV: {path}")


def save_json(results: List[GroupSummary], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"  JSON: {path}")


def generate_latex_table(results: List[GroupSummary], path: Path):
    """生成论文用LaTeX表格"""
    path.parent.mkdir(parents=True, exist_ok=True)
    q_values = sorted(set(r.q_mods for r in results))
    methods = ["Prior", "STHoles", "QuickSel-H", "ISOMER", "OASIS"]

    with open(path, "w") as f:
        f.write("\\begin{table}[!htb]\n")
        f.write("  \\centering\n")
        f.write("  \\caption{Q-Error comparison across drift intensities ($\\downarrow$ better).\n")
        f.write("           OASIS is trained on $q \\in \\{1,3,5,10,15,20\\}$ (1000 samples per $q$).\n")
        f.write("           Bold denotes best method. ``+\\%%'' shows improvement over Stale Prior.}\n")
        f.write("  \\label{tab:qerror}\n")
        f.write("  \\setlength{\\tabcolsep}{3pt}\n")
        f.write("  \\begin{tabular}{c rr rr rr rr}\n")
        f.write("    \\toprule\n")
        f.write("    & \\multicolumn{2}{c}{Stale Prior}\n")
        f.write("    & \\multicolumn{2}{c}{STHoles}\n")
        f.write("    & \\multicolumn{2}{c}{QuickSel-H}\n")
        f.write("    & \\multicolumn{2}{c}{OASIS (ours)} \\\\\n")
        f.write("    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n")
        f.write("    $q$ & Q-Err & — & Q-Err & +\\% & Q-Err & +\\% & Q-Err & +\\% \\\\\n")
        f.write("    \\midrule\n")

        for q in q_values:
            q_results = {r.model_name: r for r in results if r.q_mods == q}
            prior = q_results.get("Prior")
            sth = q_results.get("STHoles")
            qs = q_results.get("QuickSel-H")
            oasis = q_results.get("OASIS")

            if not all([prior, sth, qs, oasis]):
                continue

            # Find best non-prior
            best_qerr = min(sth.qerror_mean, qs.qerror_mean, oasis.qerror_mean)

            def fmt(r, is_best):
                if is_best:
                    return f"\\textbf{{{r.qerror_mean:.3f}}} & \\textbf{{{r.improvement_vs_prior:+.1f}\\%}}"
                return f"{r.qerror_mean:.3f} & {r.improvement_vs_prior:+.1f}\\%"

            f.write(f"    \\textbf{{{q:2d}}} & {prior.qerror_mean:.3f} & — "
                    f"& {fmt(sth, abs(sth.qerror_mean - best_qerr) < 0.001)} "
                    f"& {fmt(qs, abs(qs.qerror_mean - best_qerr) < 0.001)} "
                    f"& {fmt(oasis, abs(oasis.qerror_mean - best_qerr) < 0.001)} \\\\\n")

        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  LaTeX: {path}")


def generate_plots(results: List[GroupSummary], output_dir: Path):
    """生成论文用图表"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib不可用，跳过图表生成")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    q_values = sorted(set(r.q_mods for r in results))
    methods = ["Prior", "STHoles", "QuickSel-H", "ISOMER", "OASIS"]
    colors = {
        "Prior": "#e74c3c",
        "STHoles": "#3498db",
        "QuickSel-H": "#9b59b6",
        "ISOMER": "#f39c12",
        "OASIS": "#27ae60",
    }
    markers = {"Prior": "o", "STHoles": "^", "QuickSel-H": "D", "ISOMER": "s", "OASIS": "v"}

    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 9})

    # 提取数据
    data = {}
    for m in methods:
        data[m] = {"qerror": [], "sel": [], "mae": []}
        for q in q_values:
            r = next((r for r in results if r.q_mods == q and r.model_name == m), None)
            if r:
                data[m]["qerror"].append(r.qerror_mean)
                data[m]["sel"].append(r.sel_error_mean)
                data[m]["mae"].append(r.quantile_mae_mean)

    # 图1: Q-Error
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for m in methods:
        if data[m]["qerror"]:
            ax.plot(q_values, data[m]["qerror"], marker=markers[m], color=colors[m],
                    label=m, linewidth=2, markersize=6)
    ax.set_xlabel('Drift Intensity ($q$)')
    ax.set_ylabel('Q-Error ($\\downarrow$ better)')
    ax.set_xticks(q_values)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'ablation_qerror.pdf', bbox_inches='tight')
    plt.close()

    # 图2: Selectivity Error
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for m in methods:
        if data[m]["sel"]:
            ax.plot(q_values, data[m]["sel"], marker=markers[m], color=colors[m],
                    label=m, linewidth=2, markersize=6)
    ax.set_xlabel('Drift Intensity ($q$)')
    ax.set_ylabel('Selectivity MAE ($\\downarrow$ better)')
    ax.set_xticks(q_values)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'ablation_selerror.pdf', bbox_inches='tight')
    plt.close()

    # 图3: Quantile MAE
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for m in methods:
        if data[m]["mae"]:
            ax.plot(q_values, data[m]["mae"], marker=markers[m], color=colors[m],
                    label=m, linewidth=2, markersize=6)
    ax.set_xlabel('Drift Intensity ($q$)')
    ax.set_ylabel('Quantile MAE ($\\downarrow$ better)')
    ax.set_xticks(q_values)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'ablation_mae.pdf', bbox_inches='tight')
    plt.close()

    print(f"  图表已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="最终实验：多方法对比")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=Path("work_final"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        k_train, k_test = 200, 50
        test_q_values = [1, 5, 10, 20]
    else:
        k_train, k_test = 1000, 128
        test_q_values = [1, 3, 5, 10, 15, 20, 25, 30]

    num_buckets, max_obs = 10, 16
    train_q_values = [1, 3, 5, 10, 15, 20]
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("  OASIS 最终实验：Prior vs STHoles vs QuickSel-H vs ISOMER vs OASIS")
    print(f"  训练: q={train_q_values}, {k_train} 样本/q")
    print(f"  测试: q={test_q_values}, {k_test} 样本/q")
    print("=" * 100)

    # 阶段1: 训练
    print("\n[阶段 1/4] 训练 OASIS 模型")
    print("-" * 100)
    model = train_oasis_model(train_q_values, k_train, num_buckets, max_obs, work_dir, args.seed)

    # 阶段2: 评估
    print("\n[阶段 2/4] 评估所有方法")
    print("-" * 100)
    results = run_evaluation(test_q_values, k_test, num_buckets, max_obs, work_dir, model, args.seed)

    # 阶段3: 保存结果
    print("\n[阶段 3/4] 保存结果")
    print("-" * 100)
    results_dir = work_dir / "results"
    save_csv(results, results_dir / "final_results.csv")
    save_json(results, results_dir / "final_results.json")
    generate_latex_table(results, results_dir / "table_qerror.tex")

    # 阶段4: 生成图表（直接输出到论文目录）
    print("\n[阶段 4/4] 生成图表")
    print("-" * 100)
    paper_fig_dir = _SCRIPT_DIR.parent / "paper" / "figures"
    generate_plots(results, paper_fig_dir)
    generate_plots(results, results_dir)  # 也保存一份到结果目录

    # 打印汇总
    print("\n" + "=" * 100)
    q_values = sorted(set(r.q_mods for r in results))
    methods = ["Prior", "STHoles", "QuickSel-H", "ISOMER", "OASIS"]
    header = f"{'q':>4}"
    for m in methods:
        header += f"  {m:>12}"
    print(header)
    print("-" * 100)
    for q in q_values:
        line = f"{q:>4}"
        for m in methods:
            r = next((r for r in results if r.q_mods == q and r.model_name == m), None)
            if r:
                imp = f"({r.improvement_vs_prior:+.1f}%)" if m != "Prior" else ""
                line += f"  {r.qerror_mean:>7.3f}{imp:>8}"
            else:
                line += f"  {'N/A':>12}"
        print(line)
    print("=" * 100)
    print("\n✓ 所有实验完成！")


if __name__ == "__main__":
    main()
