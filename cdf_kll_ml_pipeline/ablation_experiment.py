"""
消融实验：量化直方图校正效果

实验设计：
  - 自变量：数据漂移程度（--q 参数，即每次观测间的修改批次数）
  - 三种方法对比：
      A) prior        —— 完全用旧统计（不校正）
      B) teacher      —— 使用 cdf_teacher（保序回归，无监督校正）
      C) mlp_model    —— 使用训练好的 Attention-pooled MLP 模型
  - 因变量：
      Q-error         —— max(est/act, act/est)，越低越好
      MAE             —— 分位点的绝对误差，越低越好
      Sel-error       —— 选择率绝对误差（模拟实际查询估算质量）

用法：
    cd presto-cdf-simulation/cdf_kll_ml_pipeline

    # 完整消融实验（自动生成数据、训练、评估、出图）
    python3 ablation_experiment.py

    # 指定漂移梯度
    python3 ablation_experiment.py --q-values 1 5 10 20 40

    # 不生成图（无 matplotlib 环境）
    python3 ablation_experiment.py --no-plot
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# 公共工具 —— 直接复用工程内已有模块
# ---------------------------------------------------------------------------
from cdf_teacher import correct_quantiles
from histogram_math import clamp01, evaluate_piecewise_cdf
from histogram_types import KllFeedbackSample
from json_histogram_parser import load_feedback_sample
from mlp_histogram_model import MlpHistogramModel
from tensorizer import tensorize_sample, OBSERVATION_FEATURE_DIM_NO_TS
from stgrid import correct_stgrid
from baselines import correct_stholes, correct_qm

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    q_mods: int
    case_id: str
    obs_count: int

    mae_prior: float
    mae_teacher: float
    mae_mlp: float          # Attention-pooled MLP
    mae_stgrid: float
    mae_stholes: float
    mae_qm: float

    sel_err_prior: float
    sel_err_teacher: float
    sel_err_mlp: float
    sel_err_stgrid: float
    sel_err_stholes: float
    sel_err_qm: float

    qerr_prior: float
    qerr_teacher: float
    qerr_mlp: float
    qerr_stgrid: float
    qerr_stholes: float
    qerr_qm: float


@dataclass
class GroupSummary:
    q_mods: int
    n: int

    mae_prior: float
    mae_teacher: float
    mae_mlp: float
    mae_stgrid: float
    mae_stholes: float
    mae_qm: float

    sel_err_prior: float
    sel_err_teacher: float
    sel_err_mlp: float
    sel_err_stgrid: float
    sel_err_stholes: float
    sel_err_qm: float

    qerr_prior: float
    qerr_teacher: float
    qerr_mlp: float
    qerr_stgrid: float
    qerr_stholes: float
    qerr_qm: float


# ---------------------------------------------------------------------------
# 指标计算工具
# ---------------------------------------------------------------------------

def _build_cdf_from_boundaries(boundaries: List[float]) -> Tuple[List[float], List[float]]:
    """等深直方图边界 → (cdf_x, cdf_p)"""
    b = len(boundaries) - 1
    return list(boundaries), [i / b for i in range(b + 1)]


def _cdf_fn(boundaries: List[float]):
    cdf_x, cdf_p = _build_cdf_from_boundaries(boundaries)
    def fn(v: float) -> float:
        return evaluate_piecewise_cdf(cdf_x, cdf_p, v)
    return fn


def quantile_mae(pred_boundaries: List[float], true_boundaries: List[float]) -> float:
    """内部分位点（去掉首尾 0/1）的平均绝对误差。"""
    inner_pred = pred_boundaries[1:-1]
    inner_true = true_boundaries[1:-1]
    if not inner_pred:
        return 0.0
    return sum(abs(p - t) for p, t in zip(inner_pred, inner_true)) / len(inner_pred)


def selectivity_error(
    pred_boundaries: List[float],
    true_boundaries: List[float],
    rng: random.Random,
    n_probes: int = 50,
) -> float:
    """
    随机采样 n_probes 个 < 谓词，计算估算选择率与真实选择率的平均绝对误差。
    """
    est_fn = _cdf_fn(pred_boundaries)
    act_fn = _cdf_fn(true_boundaries)
    total = 0.0
    for _ in range(n_probes):
        v = rng.uniform(0.0, 1.0)
        est = est_fn(v)
        act = act_fn(v)
        total += abs(est - act)
    return total / n_probes


def q_error(
    pred_boundaries: List[float],
    true_boundaries: List[float],
    rng: random.Random,
    n_probes: int = 50,
    epsilon: float = 1e-6,
) -> float:
    """
    mean Q-error on < predicates.
    Q-error = max(est/act, act/est)
    """
    est_fn = _cdf_fn(pred_boundaries)
    act_fn = _cdf_fn(true_boundaries)
    total = 0.0
    for _ in range(n_probes):
        v = rng.uniform(0.05, 0.95)
        est = max(est_fn(v), epsilon)
        act = max(act_fn(v), epsilon)
        total += max(est / act, act / est)
    return total / n_probes


# ---------------------------------------------------------------------------
# 单个样本评估（三种方法）
# ---------------------------------------------------------------------------

def evaluate_case(
    sample: KllFeedbackSample,
    true_boundaries: List[float],
    mlp_model: Optional[MlpHistogramModel],
    max_observations: int,
    eval_rng: random.Random,
):
    prior_boundaries = [sample.prior.min_value] + list(sample.prior.quantile_values) + [sample.prior.max_value]

    mae_p = quantile_mae(prior_boundaries, true_boundaries)
    sel_p = selectivity_error(prior_boundaries, true_boundaries, eval_rng)
    qerr_p = q_error(prior_boundaries, true_boundaries, eval_rng)

    teacher_quantiles = correct_quantiles(sample)
    teacher_boundaries = [sample.prior.min_value] + teacher_quantiles + [sample.prior.max_value]
    mae_t = quantile_mae(teacher_boundaries, true_boundaries)
    sel_t = selectivity_error(teacher_boundaries, true_boundaries, eval_rng)
    qerr_t = q_error(teacher_boundaries, true_boundaries, eval_rng)

    stgrid_quantiles = correct_stgrid(
        prior_min=sample.prior.min_value,
        prior_max=sample.prior.max_value,
        prior_quantiles=list(sample.prior.quantile_values),
        observations=[
            {"predicate_type": o.predicate_type, "value": o.value, "value_upper": o.value_upper, "actual_sel": o.actual_selectivity}
            for o in sample.observations[:max_observations]
        ],
        num_buckets=len(sample.prior.quantile_values) + 1,
        lr=0.5
    )
    stgrid_boundaries = [sample.prior.min_value] + stgrid_quantiles + [sample.prior.max_value]
    mae_s = quantile_mae(stgrid_boundaries, true_boundaries)
    sel_s = selectivity_error(stgrid_boundaries, true_boundaries, eval_rng)
    qerr_s = q_error(stgrid_boundaries, true_boundaries, eval_rng)

    common_obs = [
        {"predicate_type": o.predicate_type, "value": o.value, "value_upper": o.value_upper, "actual_sel": o.actual_selectivity}
        for o in sample.observations[:max_observations]
    ]

    stholes_quantiles = correct_stholes(
        sample.prior.min_value, sample.prior.max_value, list(sample.prior.quantile_values), 
        common_obs, num_buckets=len(sample.prior.quantile_values) + 1
    )
    b_holes = [sample.prior.min_value] + stholes_quantiles + [sample.prior.max_value]
    mae_sh = quantile_mae(b_holes, true_boundaries)
    sel_sh = selectivity_error(b_holes, true_boundaries, eval_rng)
    qerr_sh = q_error(b_holes, true_boundaries, eval_rng)

    qm_quantiles = correct_qm(
        sample.prior.min_value, sample.prior.max_value, list(sample.prior.quantile_values), 
        common_obs
    )
    b_qm = [sample.prior.min_value] + qm_quantiles + [sample.prior.max_value]
    mae_qm = quantile_mae(b_qm, true_boundaries)
    sel_qm = selectivity_error(b_qm, true_boundaries, eval_rng)
    qerr_qm = q_error(b_qm, true_boundaries, eval_rng)

    def _run_model(
        m,
        use_time_decay: bool,
    ):
        if m is None:
            return mae_p, sel_p, qerr_p
        try:
            record = tensorize_sample(sample, max_observations=max_observations,
                                      teacher_fn=None, use_time_decay=use_time_decay)
            predicted_norm = m.predict([record.feature_tensor])[0]
            val_range = max(sample.prior.value_range, 1e-12)
            model_quantiles = [
                clamp01(sample.prior.min_value + v * val_range)
                for v in predicted_norm
            ]
            for idx in range(1, len(model_quantiles)):
                if model_quantiles[idx] < model_quantiles[idx - 1]:
                    model_quantiles[idx] = model_quantiles[idx - 1]
            bounds = [sample.prior.min_value] + model_quantiles + [sample.prior.max_value]
            return (
                quantile_mae(bounds, true_boundaries),
                selectivity_error(bounds, true_boundaries, eval_rng),
                q_error(bounds, true_boundaries, eval_rng),
            )
        except Exception:
            return mae_p, sel_p, qerr_p

    mae_mlp, sel_mlp, qerr_mlp = _run_model(mlp_model, use_time_decay=False)

    return (mae_p, mae_t, mae_mlp, mae_s, mae_sh, mae_qm,
            sel_p, sel_t, sel_mlp, sel_s, sel_sh, sel_qm,
            qerr_p, qerr_t, qerr_mlp, qerr_s, qerr_sh, qerr_qm)


# ---------------------------------------------------------------------------
# 数据生成（调用已有脚本）
# ---------------------------------------------------------------------------

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


def train_mlp(
    train_glob: str,
    model_path: Path,
    max_observations: int = 16,
    epochs: int = 200,
    lr: float = 3e-3,
) -> MlpHistogramModel:
    """MLP 模型训练。"""
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


# ---------------------------------------------------------------------------
# 主实验流程
# ---------------------------------------------------------------------------

def run_ablation(
    q_values: List[int],
    k_train: int,
    k_test: int,
    num_buckets: int,
    max_observations: int,
    work_dir: Path,
    seed: int,
    train_q_values: Optional[List[int]] = None,
    mlp_epochs: int = 200,
) -> List[GroupSummary]:
    eval_rng = random.Random(seed + 9999)
    summaries: List[GroupSummary] = []

    # 训练数据生成
    if train_q_values is None:
        train_q_values = [5]  # default: single q=5

    q_tag = "_".join(str(q) for q in train_q_values)
    train_dirs_label = f"train_q{q_tag}_k{k_train}"
    mlp_model_path = work_dir / "artifacts" / f"mlp_{train_dirs_label}.json"

    # 生成各 q 的训练数据
    train_json_files: List[str] = []
    for tq in train_q_values:
        tdir = work_dir / f"train_q{tq}_k{k_train}"
        if not tdir.exists() or not any(tdir.glob("*.json")):
            print(f"  [训练] 生成训练数据 (q={tq}, k={k_train}) ...")
            generate_data(tdir, k=k_train, num_buckets=num_buckets, q_mods=tq, seed=seed + tq)
        train_json_files += [str(p) for p in tdir.glob("*.json")]

    # 将多个 q 的训练文件通过 symlink 合并到一个目录
    merged_dir = work_dir / f"train_merged_{train_dirs_label}"
    if not merged_dir.exists():
        merged_dir.mkdir(parents=True)
        import shutil
        for src in train_json_files:
            dst = merged_dir / Path(src).name
            if not dst.exists():
                shutil.copy2(src, dst)
        print(f"  [训练] 合并完成: {len(train_json_files)} 条训练样本 (q={train_q_values})")

    train_glob = str(merged_dir / "*.json")
    total_train = len(list(merged_dir.glob("*.json")))

    # Attention-pooled MLP
    if not mlp_model_path.exists():
        print(f"  [训练] 训练 MLP ({total_train} 条样本, 128→64, epochs={mlp_epochs}) ...")
        mlp_model = train_mlp(train_glob, mlp_model_path, max_observations, epochs=mlp_epochs)
    else:
        print(f"  [复用] {mlp_model_path.name}")
        mlp_model = MlpHistogramModel.load(str(mlp_model_path))

    for q in q_values:
        print(f"\n========== q={q} (漂移强度) ==========")
        test_dir = work_dir / f"test_q{q}"

        # 生成测试集
        print(f"  生成 {k_test} 条测试样本 ...")
        generate_data(test_dir, k=k_test, num_buckets=num_buckets, q_mods=q, seed=seed + 10000 + q)

        results: List[CaseResult] = []
        test_files = sorted(test_dir.glob("*.json"))

        for fpath in test_files:
            data = json.loads(fpath.read_text())

            # ground truth
            true_boundaries: List[float] = data["corrected_kll"]["bucket_boundaries"]

            # 加载为 KllFeedbackSample
            sample: KllFeedbackSample = load_feedback_sample(str(fpath))

            (mae_p, mae_t, mae_mlp, mae_stgrid, mae_stholes, mae_qm,
             sel_p, sel_t, sel_mlp, sel_stgrid, sel_stholes, sel_qm,
             qerr_p, qerr_t, qerr_mlp, qerr_stgrid, qerr_stholes, qerr_qm) = evaluate_case(
                sample, true_boundaries, mlp_model, max_observations, eval_rng
            )

            results.append(CaseResult(
                q_mods=q,
                case_id=fpath.stem,
                obs_count=len(data["observations"]),
                mae_prior=mae_p, mae_teacher=mae_t, mae_mlp=mae_mlp, mae_stgrid=mae_stgrid, 
                mae_stholes=mae_stholes, mae_qm=mae_qm,
                sel_err_prior=sel_p, sel_err_teacher=sel_t, sel_err_mlp=sel_mlp, sel_err_stgrid=sel_stgrid,
                sel_err_stholes=sel_stholes, sel_err_qm=sel_qm,
                qerr_prior=qerr_p, qerr_teacher=qerr_t, qerr_mlp=qerr_mlp, qerr_stgrid=qerr_stgrid,
                qerr_stholes=qerr_stholes, qerr_qm=qerr_qm,
            ))

        n = len(results)
        def avg(lst): return sum(lst) / max(n, 1)

        summary = GroupSummary(
            q_mods=q, n=n,
            mae_prior=avg([r.mae_prior for r in results]),
            mae_teacher=avg([r.mae_teacher for r in results]),
            mae_mlp=avg([r.mae_mlp for r in results]),
            mae_stgrid=avg([r.mae_stgrid for r in results]),
            mae_stholes=avg([r.mae_stholes for r in results]),
            mae_qm=avg([r.mae_qm for r in results]),

            sel_err_prior=avg([r.sel_err_prior for r in results]),
            sel_err_teacher=avg([r.sel_err_teacher for r in results]),
            sel_err_mlp=avg([r.sel_err_mlp for r in results]),
            sel_err_stgrid=avg([r.sel_err_stgrid for r in results]),
            sel_err_stholes=avg([r.sel_err_stholes for r in results]),
            sel_err_qm=avg([r.sel_err_qm for r in results]),

            qerr_prior=avg([r.qerr_prior for r in results]),
            qerr_teacher=avg([r.qerr_teacher for r in results]),
            qerr_mlp=avg([r.qerr_mlp for r in results]),
            qerr_stgrid=avg([r.qerr_stgrid for r in results]),
            qerr_stholes=avg([r.qerr_stholes for r in results]),
            qerr_qm=avg([r.qerr_qm for r in results]),
        )
        summaries.append(summary)

        # 即时打印（4列）
        print(f"  样本数: {n}")
        print(f"  {'方法':<18} {'MAE(分位点)':<14} {'Sel-Error':<12} {'Q-Error':<10}")
        print(f"  {'Prior':<18} {summary.mae_prior:<14.5f} {summary.sel_err_prior:<12.5f} {summary.qerr_prior:<10.4f}")
        print(f"  {'Teacher':<18} {summary.mae_teacher:<14.5f} {summary.sel_err_teacher:<12.5f} {summary.qerr_teacher:<10.4f}")
        print(f"  {'STGrid':<18} {summary.mae_stgrid:<14.5f} {summary.sel_err_stgrid:<12.5f} {summary.qerr_stgrid:<10.4f}")
        print(f"  {'STHoles':<18} {summary.mae_stholes:<14.5f} {summary.sel_err_stholes:<12.5f} {summary.qerr_stholes:<10.4f}")
        print(f"  {'OASIS MLP':<18} {summary.mae_mlp:<14.5f} {summary.sel_err_mlp:<12.5f} {summary.qerr_mlp:<10.4f}")

        # 提升率
        def pct_improve(base, new):
            if base < 1e-9: return 0.0
            return (base - new) / base * 100

        print(f"\n  Q-Error 降低 vs Prior:")
        print(f"    Teacher:       {pct_improve(summary.qerr_prior, summary.qerr_teacher):.1f}%")
        print(f"    OASIS MLP:    {pct_improve(summary.qerr_prior, summary.qerr_mlp):.1f}%")

    return summaries


# ---------------------------------------------------------------------------
# 结果输出（CSV + 可选图表）
# ---------------------------------------------------------------------------

def save_csv(summaries: List[GroupSummary], output_path: Path) -> None:
    lines = [
        "q_mods,n,mae_prior,mae_teacher,mae_mlp,mae_stgrid,mae_stholes,mae_qm,"
        "sel_err_prior,sel_err_teacher,sel_err_mlp,sel_err_stgrid,sel_err_stholes,sel_err_qm,"
        "qerr_prior,qerr_teacher,qerr_mlp,qerr_stgrid,qerr_stholes,qerr_qm"
    ]
    for s in summaries:
        lines.append(
            f"{s.q_mods},{s.n},"
            f"{s.mae_prior:.6f},{s.mae_teacher:.6f},{s.mae_mlp:.6f},{s.mae_stgrid:.6f},{s.mae_stholes:.6f},{s.mae_qm:.6f},"
            f"{s.sel_err_prior:.6f},{s.sel_err_teacher:.6f},{s.sel_err_mlp:.6f},{s.sel_err_stgrid:.6f},{s.sel_err_stholes:.6f},{s.sel_err_qm:.6f},"
            f"{s.qerr_prior:.6f},{s.qerr_teacher:.6f},{s.qerr_mlp:.6f},{s.qerr_stgrid:.6f},{s.qerr_stholes:.6f},{s.qerr_qm:.6f}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结果已保存到 {output_path}")


def plot_results(summaries: List[GroupSummary], output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图。可运行: pip install matplotlib")
        return

    q_vals = [s.q_mods for s in summaries]

    colors = {
        "Prior":       "#e74c3c",
        "Teacher":     "#f39c12",
        "STGrid":      "#27ae60",
        "STHoles":     "#3498db",
        "OASIS MLP":   "#8e44ad",
    }
    styles = {
        "Prior":       ("o-",  1.5),
        "Teacher":     ("s--", 1.5),
        "STGrid":      ("x-.", 1.5),
        "STHoles":     ("d-.", 1.5),
        "OASIS MLP":   ("^-",  2.5),
    }

    metrics = [
        ("Q-Error (↓ better)",
         [s.qerr_prior for s in summaries],
         [s.qerr_teacher for s in summaries],
         [s.qerr_stgrid for s in summaries],
         [s.qerr_stholes for s in summaries],
         [s.qerr_mlp for s in summaries],
         "ablation_qerror.pdf"),
        ("Selectivity Error (↓ better)",
         [s.sel_err_prior for s in summaries],
         [s.sel_err_teacher for s in summaries],
         [s.sel_err_stgrid for s in summaries],
         [s.sel_err_stholes for s in summaries],
         [s.sel_err_mlp for s in summaries],
         "ablation_selerror.pdf"),
        ("Quantile MAE (↓ better)",
         [s.mae_prior for s in summaries],
         [s.mae_teacher for s in summaries],
         [s.mae_stgrid for s in summaries],
         [s.mae_stholes for s in summaries],
         [s.mae_mlp for s in summaries],
         "ablation_mae.pdf"),
    ]

    for m_idx, (title, prior_vals, teacher_vals, stgrid_vals, stholes_vals, mlp_vals, filename) in enumerate(metrics):
        plt.figure(figsize=(5, 4))
        for name, vals in [
            ("Prior",       prior_vals),
            ("Teacher",     teacher_vals),
            ("STGrid",      stgrid_vals),
            ("STHoles",     stholes_vals),
            ("OASIS MLP",   mlp_vals),
        ]:
            fmt, lw = styles[name]
            plt.plot(q_vals, vals, fmt, color=colors[name], label=name, linewidth=lw)
        
        plt.xlabel("Drift Intensity ($q$)", fontsize=11)
        plt.ylabel(title, fontsize=11)
        plt.title(f"{title.split(' ')[0]} vs. Drift Intensity", fontsize=11)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_file = output_path.parent / filename
        plt.savefig(str(out_file), dpi=300, bbox_inches="tight")
        print(f"图表已单独保存到 {out_file}")
        plt.close()
    print(f"图表已保存到 {output_path}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="消融实验：直方图校正效果 vs. 漂移强度")
    parser.add_argument(
        "--q-values", nargs="+", type=int,
        default=[1, 3, 5, 10, 15, 20, 25, 30],
        help="要测试的漂移强度列表（q_mods 参数）",
    )
    parser.add_argument(
        "--train-q-values", nargs="+", type=int,
        default=[10, 20],
        help="训练数据的 q 列表，默认仅用 q=5（单一）。指定多个 q 则合并训练集",
    )
    parser.add_argument("--k-train", type=int, default=1000, help="每个 q 的训练样本数")
    parser.add_argument("--k-test", type=int, default=128, help="每个漂移强度的测试样本数")
    parser.add_argument("--num-buckets", type=int, default=10, help="直方图桶数")
    parser.add_argument("--max-observations", type=int, default=16, help="观测窗口大小")
    parser.add_argument("--work-dir", type=Path, default=Path("ablation_work"), help="工作目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--mlp-epochs", type=int, default=200, help="MLP 训练轮数")
    parser.add_argument("--no-plot", action="store_true", help="不生成图表")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("直方图校正消融实验")
    print(f"漂移梯度: {args.q_values}")
    print(f"训练集: {args.k_train} 条 | 测试集: {args.k_test} 条/组")
    print(f"工作目录: {work_dir.resolve()}")
    print("=" * 60)

    summaries = run_ablation(
        q_values=args.q_values,
        k_train=args.k_train,
        k_test=args.k_test,
        num_buckets=args.num_buckets,
        max_observations=args.max_observations,
        work_dir=work_dir,
        seed=args.seed,
        train_q_values=args.train_q_values,
        mlp_epochs=args.mlp_epochs,
    )

    # 保存 CSV
    csv_path = work_dir / "ablation_results.csv"
    save_csv(summaries, csv_path)

    # 打印最终汇总表（Prior / Teacher / STGrid / MLP）
    print("\n" + "=" * 80)
    print(f"{'q':>4} {'MAE_Prior':>10} {'MAE_Teacher':>12} {'MAE_STGrid':>11} {'MAE_MLP':>9} "
          f"{'QErr_Prior':>11} {'QErr_Teacher':>13} {'QErr_STGrid':>12} {'QErr_MLP':>10}")
    print("-" * 80)
    for s in summaries:
        print(f"{s.q_mods:>4} {s.mae_prior:>10.5f} {s.mae_teacher:>12.5f} "
              f"{s.mae_stgrid:>11.5f} {s.mae_mlp:>9.5f} "
              f"{s.qerr_prior:>11.4f} {s.qerr_teacher:>13.4f} "
              f"{s.qerr_stgrid:>12.4f} {s.qerr_mlp:>10.4f}")

    if not args.no_plot:
        plot_path = work_dir / "ablation_plot.png"
        plot_results(summaries, plot_path)


if __name__ == "__main__":
    main()
