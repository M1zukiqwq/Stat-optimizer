"""
直方图校正消融实验
==================

比较三种方法在不同数据漂移强度下的估算精度：
  A) Prior       —— 完全使用旧统计（不校正），作为 baseline
  B) Teacher     —— cdf_teacher 保序回归（无监督）
  C) Ridge Model —— 训练好的 Ridge 回归模型（有监督）

度量指标：
  - Q-error      : max(est/act, act/est)，学术标准，越低越好
  - Sel-Error    : 选择率平均绝对误差
  - MAE          : 分位点绝对误差（归一化空间）

用法（从 ablation_study/ 目录运行）：
    python3 ablation_experiment.py
    python3 ablation_experiment.py --q-values 1 5 10 20 40 --k-train 512 --k-test 128
    python3 ablation_experiment.py --no-plot   # 无 matplotlib 时使用
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# 自动解析 pipeline 路径，使本脚本可从 ablation_study/ 直接运行
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _SCRIPT_DIR.parent / "cdf_kll_ml_pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from cdf_teacher import correct_quantiles                       # noqa: E402
from histogram_math import clamp01, evaluate_piecewise_cdf      # noqa: E402
from histogram_types import KllFeedbackSample                   # noqa: E402
from json_histogram_parser import load_feedback_sample          # noqa: E402
from ridge_histogram_model import RidgeMultiOutputRegressor     # noqa: E402
from tensorizer import tensorize_sample                         # noqa: E402


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
    mae_model: float
    sel_err_prior: float
    sel_err_teacher: float
    sel_err_model: float
    qerr_prior: float
    qerr_teacher: float
    qerr_model: float


@dataclass
class GroupSummary:
    q_mods: int
    n: int
    mae_prior: float
    mae_teacher: float
    mae_model: float
    sel_err_prior: float
    sel_err_teacher: float
    sel_err_model: float
    qerr_prior: float
    qerr_teacher: float
    qerr_model: float


# ---------------------------------------------------------------------------
# 指标工具函数
# ---------------------------------------------------------------------------

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
    return sum(abs(est_fn(rng.uniform(0, 1)) - act_fn(rng.uniform(0, 1))) for _ in range(n)) / n


def q_error(pred: List[float], true: List[float], rng: random.Random, n: int = 50, eps: float = 1e-6) -> float:
    est_fn, act_fn = _cdf_fn(pred), _cdf_fn(true)
    total = 0.0
    for _ in range(n):
        v = rng.uniform(0.05, 0.95)
        est = max(est_fn(v), eps)
        act = max(act_fn(v), eps)
        total += max(est / act, act / est)
    return total / n


# ---------------------------------------------------------------------------
# 单样本评估
# ---------------------------------------------------------------------------

def evaluate_case(
    sample: KllFeedbackSample,
    true_boundaries: List[float],
    model: Optional[RidgeMultiOutputRegressor],
    max_obs: int,
    rng: random.Random,
) -> Tuple[float, ...]:
    prior_b = [sample.prior.min_value] + list(sample.prior.quantile_values) + [sample.prior.max_value]

    # A: Prior
    mae_p   = quantile_mae(prior_b, true_boundaries)
    sel_p   = selectivity_error(prior_b, true_boundaries, rng)
    qerr_p  = q_error(prior_b, true_boundaries, rng)

    # B: Teacher
    teacher_q = correct_quantiles(sample)
    teacher_b = [sample.prior.min_value] + teacher_q + [sample.prior.max_value]
    mae_t   = quantile_mae(teacher_b, true_boundaries)
    sel_t   = selectivity_error(teacher_b, true_boundaries, rng)
    qerr_t  = q_error(teacher_b, true_boundaries, rng)

    # C: Ridge Model
    if model is None:
        mae_m, sel_m, qerr_m = mae_p, sel_p, qerr_p
    else:
        try:
            record = tensorize_sample(sample, max_observations=max_obs, teacher_fn=None)
            pred_norm = model.predict([record.feature_tensor])[0]
            vr = max(sample.prior.value_range, 1e-12)
            model_q = [clamp01(sample.prior.min_value + v * vr) for v in pred_norm]
            for i in range(1, len(model_q)):
                if model_q[i] < model_q[i - 1]:
                    model_q[i] = model_q[i - 1]
            model_b = [sample.prior.min_value] + model_q + [sample.prior.max_value]
            mae_m   = quantile_mae(model_b, true_boundaries)
            sel_m   = selectivity_error(model_b, true_boundaries, rng)
            qerr_m  = q_error(model_b, true_boundaries, rng)
        except Exception:
            mae_m, sel_m, qerr_m = mae_p, sel_p, qerr_p

    return mae_p, mae_t, mae_m, sel_p, sel_t, sel_m, qerr_p, qerr_t, qerr_m


# ---------------------------------------------------------------------------
# 数据生成与模型训练（调用已有脚本）
# ---------------------------------------------------------------------------

def _run_pipeline_script(script_name: str, args_list: List[str]) -> None:
    script_path = _PIPELINE_DIR / script_name
    subprocess.run([sys.executable, str(script_path)] + args_list, check=True, capture_output=True)


def generate_data(output_dir: Path, k: int, num_buckets: int, q_mods: int, seed: int) -> None:
    _run_pipeline_script("simulate_memory_kll_dataset.py", [
        "--output-dir", str(output_dir),
        "--k", str(k),
        "--num-buckets", str(num_buckets),
        "--q", str(q_mods),
        "--seed", str(seed),
        "--initial-rows", "5000",
    ])


def train_model(train_dir: Path, model_path: Path, max_obs: int) -> RidgeMultiOutputRegressor:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    _run_pipeline_script("train_histogram_model.py", [
        "--train-glob", str(train_dir / "*.json"),
        "--output-model", str(model_path),
        "--max-observations", str(max_obs),
    ])
    return RidgeMultiOutputRegressor.load(str(model_path))


# ---------------------------------------------------------------------------
# 主实验循环
# ---------------------------------------------------------------------------

def run_ablation(
    q_values: List[int],
    k_train: int,
    k_test: int,
    num_buckets: int,
    max_obs: int,
    work_dir: Path,
    seed: int,
) -> List[GroupSummary]:
    eval_rng = random.Random(seed + 9999)
    summaries: List[GroupSummary] = []

    # 训练模型（用中等漂移 q=5 的数据）
    train_dir  = work_dir / "train_q5"
    model_path = work_dir / "artifacts" / "ablation_ridge_model.json"
    if not model_path.exists():
        print("  [训练] 生成训练数据 (q=5) ...")
        generate_data(train_dir, k=k_train, num_buckets=num_buckets, q_mods=5, seed=seed)
        print("  [训练] 训练 Ridge 模型 ...")
        model = train_model(train_dir, model_path, max_obs)
    else:
        print(f"  [训练] 复用已有模型: {model_path.name}")
        model = RidgeMultiOutputRegressor.load(str(model_path))

    for q in q_values:
        print(f"\n  ── q={q} ({'轻微' if q<=2 else '轻度' if q<=5 else '重度' if q<=15 else '极重'} 漂移) ──")
        test_dir = work_dir / f"test_q{q}"
        generate_data(test_dir, k=k_test, num_buckets=num_buckets, q_mods=q, seed=seed + q)

        results: List[CaseResult] = []
        for fpath in sorted(test_dir.glob("*.json")):
            data   = json.loads(fpath.read_text())
            sample = load_feedback_sample(str(fpath))
            true_b = data["corrected_kll"]["bucket_boundaries"]
            metrics = evaluate_case(sample, true_b, model, max_obs, eval_rng)
            results.append(CaseResult(q_mods=q, case_id=fpath.stem,
                                       obs_count=len(data["observations"]),
                                       mae_prior=metrics[0],   mae_teacher=metrics[1],   mae_model=metrics[2],
                                       sel_err_prior=metrics[3], sel_err_teacher=metrics[4], sel_err_model=metrics[5],
                                       qerr_prior=metrics[6],  qerr_teacher=metrics[7],  qerr_model=metrics[8]))

        n = len(results)
        avg = lambda lst: sum(lst) / max(n, 1)
        s = GroupSummary(
            q_mods=q, n=n,
            mae_prior=avg([r.mae_prior for r in results]),
            mae_teacher=avg([r.mae_teacher for r in results]),
            mae_model=avg([r.mae_model for r in results]),
            sel_err_prior=avg([r.sel_err_prior for r in results]),
            sel_err_teacher=avg([r.sel_err_teacher for r in results]),
            sel_err_model=avg([r.sel_err_model for r in results]),
            qerr_prior=avg([r.qerr_prior for r in results]),
            qerr_teacher=avg([r.qerr_teacher for r in results]),
            qerr_model=avg([r.qerr_model for r in results]),
        )
        summaries.append(s)

        def pct(base, new): return (base - new) / max(base, 1e-9) * 100

        print(f"  {'方法':<12} {'MAE':>10} {'Sel-Err':>10} {'Q-Error':>10}")
        print(f"  {'Prior':<12} {s.mae_prior:>10.5f} {s.sel_err_prior:>10.5f} {s.qerr_prior:>10.4f}")
        print(f"  {'Teacher':<12} {s.mae_teacher:>10.5f} {s.sel_err_teacher:>10.5f} {s.qerr_teacher:>10.4f}  [{pct(s.qerr_prior, s.qerr_teacher):+.1f}%]")
        print(f"  {'Ridge':<12} {s.mae_model:>10.5f} {s.sel_err_model:>10.5f} {s.qerr_model:>10.4f}  [{pct(s.qerr_prior, s.qerr_model):+.1f}%]")

    return summaries


# ---------------------------------------------------------------------------
# 输出：CSV + 图
# ---------------------------------------------------------------------------

def save_csv(summaries: List[GroupSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["q_mods,n,mae_prior,mae_teacher,mae_model,"
             "sel_err_prior,sel_err_teacher,sel_err_model,"
             "qerr_prior,qerr_teacher,qerr_model"]
    for s in summaries:
        lines.append(f"{s.q_mods},{s.n},"
                     f"{s.mae_prior:.6f},{s.mae_teacher:.6f},{s.mae_model:.6f},"
                     f"{s.sel_err_prior:.6f},{s.sel_err_teacher:.6f},{s.sel_err_model:.6f},"
                     f"{s.qerr_prior:.6f},{s.qerr_teacher:.6f},{s.qerr_model:.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  CSV → {path}")


def plot_results(summaries: List[GroupSummary], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib 未安装，跳过绘图。pip install matplotlib")
        return

    q_vals = [s.q_mods for s in summaries]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Histogram Correction Ablation: Drift Intensity vs. Estimation Error", fontsize=13)

    configs = [
        ("Q-Error (↓ better)",
         [s.qerr_prior for s in summaries],
         [s.qerr_teacher for s in summaries],
         [s.qerr_model for s in summaries]),
        ("Selectivity MAE (↓ better)",
         [s.sel_err_prior for s in summaries],
         [s.sel_err_teacher for s in summaries],
         [s.sel_err_model for s in summaries]),
        ("Quantile MAE (↓ better)",
         [s.mae_prior for s in summaries],
         [s.mae_teacher for s in summaries],
         [s.mae_model for s in summaries]),
    ]

    for ax, (title, p_vals, t_vals, m_vals) in zip(axes, configs):
        ax.plot(q_vals, p_vals, "o-",  color="#e74c3c", label="Prior (stale)",     linewidth=2)
        ax.plot(q_vals, t_vals, "s--", color="#f39c12", label="Teacher (isotonic)", linewidth=2)
        ax.plot(q_vals, m_vals, "^-",  color="#27ae60", label="Ridge Model",        linewidth=2)
        ax.set_xlabel("Drift Intensity (q_mods)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    print(f"  图表 → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="直方图校正消融实验")
    p.add_argument("--q-values", nargs="+", type=int, default=[1, 3, 5, 10, 20],
                   help="漂移强度列表（默认: 1 3 5 10 20）")
    p.add_argument("--k-train",         type=int, default=128)
    p.add_argument("--k-test",          type=int, default=64)
    p.add_argument("--num-buckets",     type=int, default=10)
    p.add_argument("--max-observations",type=int, default=16)
    p.add_argument("--work-dir",        type=Path, default=Path("work"))
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--no-plot",         action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 58)
    print("  直方图校正消融实验")
    print(f"  漂移梯度 : {args.q_values}")
    print(f"  训练集   : {args.k_train} 条  测试集: {args.k_test} 条/组")
    print(f"  输出目录  : {work_dir.resolve()}")
    print("=" * 58)

    summaries = run_ablation(
        q_values=args.q_values,
        k_train=args.k_train,
        k_test=args.k_test,
        num_buckets=args.num_buckets,
        max_obs=args.max_observations,
        work_dir=work_dir,
        seed=args.seed,
    )

    # 汇总表
    print("\n" + "=" * 72)
    print(f"  {'q':>4}  {'MAE_Prior':>10} {'MAE_T':>8} {'MAE_M':>8}  "
          f"{'QErr_Prior':>11} {'QErr_T':>8} {'QErr_M':>8}")
    print("  " + "-" * 68)
    for s in summaries:
        print(f"  {s.q_mods:>4}  {s.mae_prior:>10.5f} {s.mae_teacher:>8.5f} {s.mae_model:>8.5f}  "
              f"{s.qerr_prior:>11.4f} {s.qerr_teacher:>8.4f} {s.qerr_model:>8.4f}")

    csv_path = work_dir / "results" / "ablation_results.csv"
    save_csv(summaries, csv_path)

    if not args.no_plot:
        plot_results(summaries, work_dir / "results" / "ablation_plot.png")


if __name__ == "__main__":
    main()
