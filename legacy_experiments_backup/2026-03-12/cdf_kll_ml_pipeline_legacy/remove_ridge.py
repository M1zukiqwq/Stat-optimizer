import re

with open('/Users/qichutian/presto/presto-cdf-simulation/cdf_kll_ml_pipeline/ablation_experiment.py', 'r') as f:
    text = f.read()

# Docs
text = re.sub(r"C\) ridge_model  —— 使用训练好的 Ridge 回归模型", r"C) mlp_model    —— 使用训练好的 Attention-pooled MLP 模型", text)

# Imports
text = re.sub(r"from ridge_histogram_model import RidgeMultiOutputRegressor\n", "", text)

# CaseResult
text = re.sub(r"    mae_model: float\n    mae_model_no_ts: float\n", "", text)
text = re.sub(r"    sel_err_model: float\n    sel_err_model_no_ts: float\n", "", text)
text = re.sub(r"    qerr_model: float\n    qerr_model_no_ts: float\n", "", text)

# GroupSummary
text = re.sub(r"    mae_model: float\n    mae_model_no_ts: float\n", "", text)
text = re.sub(r"    sel_err_model: float\n    sel_err_model_no_ts: float\n", "", text)
text = re.sub(r"    qerr_model: float\n    qerr_model_no_ts: float\n", "", text)

# evaluate_case args
text = text.replace(
"""def evaluate_case(
    sample: KllFeedbackSample,
    true_boundaries: List[float],
    model: Optional[RidgeMultiOutputRegressor],
    model_no_ts: Optional[RidgeMultiOutputRegressor],
    mlp_model: Optional[MlpHistogramModel],
    max_observations: int,
    eval_rng: random.Random,
):""",
"""def evaluate_case(
    sample: KllFeedbackSample,
    true_boundaries: List[float],
    mlp_model: Optional[MlpHistogramModel],
    max_observations: int,
    eval_rng: random.Random,
):""")

# evaluate_case return
text = text.replace(
"""    mae_m, sel_m, qerr_m = _run_model(model, use_time_decay=True)
    mae_m_no_ts, sel_m_no_ts, qerr_m_no_ts = _run_model(model_no_ts, use_time_decay=False)
    mae_mlp, sel_mlp, qerr_mlp = _run_model(mlp_model, use_time_decay=False)

    return (mae_p, mae_t, mae_m, mae_m_no_ts, mae_mlp,
            sel_p, sel_t, sel_m, sel_m_no_ts, sel_mlp,
            qerr_p, qerr_t, qerr_m, qerr_m_no_ts, qerr_mlp)""",
"""    mae_mlp, sel_mlp, qerr_mlp = _run_model(mlp_model, use_time_decay=False)

    return (mae_p, mae_t, mae_mlp,
            sel_p, sel_t, sel_mlp,
            qerr_p, qerr_t, qerr_mlp)""")

# train_model function
text = re.sub(r"def train_model\(.*?return RidgeMultiOutputRegressor\.load\(str\(model_path\)\)\n\n\n", "", text, flags=re.DOTALL)

# run_ablation paths
text = text.replace(
"""    train_dirs_label = f"train_q{q_tag}_k{k_train}"
    model_path     = work_dir / "artifacts" / f"ridge_ts_{train_dirs_label}.json"
    model_no_ts_path = work_dir / "artifacts" / f"ridge_nots_{train_dirs_label}.json"
    mlp_model_path = work_dir / "artifacts" / f"mlp_{train_dirs_label}.json\"""",
"""    train_dirs_label = f"train_q{q_tag}_k{k_train}"
    mlp_model_path = work_dir / "artifacts" / f"mlp_{train_dirs_label}.json\"""")

# run_ablation Ridge training block
text = re.sub(r"    # Ridge WITH time-decay.*?# Attention-pooled MLP", "    # Attention-pooled MLP", text, flags=re.DOTALL)

# run_ablation evaluate_case call
text = text.replace(
"""            (mae_p, mae_t, mae_m, mae_m_no_ts, mae_mlp,
             sel_p, sel_t, sel_m, sel_m_no_ts, sel_mlp,
             qerr_p, qerr_t, qerr_m, qerr_m_no_ts, qerr_mlp) = evaluate_case(
                sample, true_boundaries, model, model_no_ts, mlp_model, max_observations, eval_rng
            )""",
"""            (mae_p, mae_t, mae_mlp,
             sel_p, sel_t, sel_mlp,
             qerr_p, qerr_t, qerr_mlp) = evaluate_case(
                sample, true_boundaries, mlp_model, max_observations, eval_rng
            )""")

# run_ablation CaseResult
text = text.replace(
"""                mae_prior=mae_p, mae_teacher=mae_t, mae_model=mae_m,
                mae_model_no_ts=mae_m_no_ts, mae_mlp=mae_mlp,
                sel_err_prior=sel_p, sel_err_teacher=sel_t, sel_err_model=sel_m,
                sel_err_model_no_ts=sel_m_no_ts, sel_err_mlp=sel_mlp,
                qerr_prior=qerr_p, qerr_teacher=qerr_t, qerr_model=qerr_m,
                qerr_model_no_ts=qerr_m_no_ts, qerr_mlp=qerr_mlp,""",
"""                mae_prior=mae_p, mae_teacher=mae_t, mae_mlp=mae_mlp,
                sel_err_prior=sel_p, sel_err_teacher=sel_t, sel_err_mlp=sel_mlp,
                qerr_prior=qerr_p, qerr_teacher=qerr_t, qerr_mlp=qerr_mlp,""")

# run_ablation GroupSummary
text = text.replace(
"""            mae_prior=avg([r.mae_prior for r in results]),
            mae_teacher=avg([r.mae_teacher for r in results]),
            mae_model=avg([r.mae_model for r in results]),
            mae_model_no_ts=avg([r.mae_model_no_ts for r in results]),
            mae_mlp=avg([r.mae_mlp for r in results]),
            sel_err_prior=avg([r.sel_err_prior for r in results]),
            sel_err_teacher=avg([r.sel_err_teacher for r in results]),
            sel_err_model=avg([r.sel_err_model for r in results]),
            sel_err_model_no_ts=avg([r.sel_err_model_no_ts for r in results]),
            sel_err_mlp=avg([r.sel_err_mlp for r in results]),
            qerr_prior=avg([r.qerr_prior for r in results]),
            qerr_teacher=avg([r.qerr_teacher for r in results]),
            qerr_model=avg([r.qerr_model for r in results]),
            qerr_model_no_ts=avg([r.qerr_model_no_ts for r in results]),
            qerr_mlp=avg([r.qerr_mlp for r in results]),""",
"""            mae_prior=avg([r.mae_prior for r in results]),
            mae_teacher=avg([r.mae_teacher for r in results]),
            mae_mlp=avg([r.mae_mlp for r in results]),
            sel_err_prior=avg([r.sel_err_prior for r in results]),
            sel_err_teacher=avg([r.sel_err_teacher for r in results]),
            sel_err_mlp=avg([r.sel_err_mlp for r in results]),
            qerr_prior=avg([r.qerr_prior for r in results]),
            qerr_teacher=avg([r.qerr_teacher for r in results]),
            qerr_mlp=avg([r.qerr_mlp for r in results]),""")

# run_ablation prints
text = text.replace(
"""        print(f"  {'Prior':<18} {summary.mae_prior:<14.5f} {summary.sel_err_prior:<12.5f} {summary.qerr_prior:<10.4f}")
        print(f"  {'Teacher':<18} {summary.mae_teacher:<14.5f} {summary.sel_err_teacher:<12.5f} {summary.qerr_teacher:<10.4f}")
        print(f"  {'Ridge':<18} {summary.mae_model_no_ts:<14.5f} {summary.sel_err_model_no_ts:<12.5f} {summary.qerr_model_no_ts:<10.4f}")
        print(f"  {'MLP (attn)':<18} {summary.mae_mlp:<14.5f} {summary.sel_err_mlp:<12.5f} {summary.qerr_mlp:<10.4f}")

        # 提升率
        def pct_improve(base, new):
            if base < 1e-9: return 0.0
            return (base - new) / base * 100

        print(f"\\n  Q-Error 降低 vs Prior:")
        print(f"    Teacher:       {pct_improve(summary.qerr_prior, summary.qerr_teacher):.1f}%")
        print(f"    Ridge:         {pct_improve(summary.qerr_prior, summary.qerr_model_no_ts):.1f}%")
        print(f"    MLP (attn):    {pct_improve(summary.qerr_prior, summary.qerr_mlp):.1f}%")""",
"""        print(f"  {'Prior':<18} {summary.mae_prior:<14.5f} {summary.sel_err_prior:<12.5f} {summary.qerr_prior:<10.4f}")
        print(f"  {'Teacher':<18} {summary.mae_teacher:<14.5f} {summary.sel_err_teacher:<12.5f} {summary.qerr_teacher:<10.4f}")
        print(f"  {'MLP (attn)':<18} {summary.mae_mlp:<14.5f} {summary.sel_err_mlp:<12.5f} {summary.qerr_mlp:<10.4f}")

        # 提升率
        def pct_improve(base, new):
            if base < 1e-9: return 0.0
            return (base - new) / base * 100

        print(f"\\n  Q-Error 降低 vs Prior:")
        print(f"    Teacher:       {pct_improve(summary.qerr_prior, summary.qerr_teacher):.1f}%")
        print(f"    MLP (attn):    {pct_improve(summary.qerr_prior, summary.qerr_mlp):.1f}%")""")

# save_csv
text = text.replace(
"""        "q_mods,n,mae_prior,mae_teacher,mae_ridge,mae_mlp,"
        "sel_err_prior,sel_err_teacher,sel_err_ridge,sel_err_mlp,"
        "qerr_prior,qerr_teacher,qerr_ridge,qerr_mlp"
    ]
    for s in summaries:
        lines.append(
            f"{s.q_mods},{s.n},"
            f"{s.mae_prior:.6f},{s.mae_teacher:.6f},{s.mae_model_no_ts:.6f},{s.mae_mlp:.6f},"
            f"{s.sel_err_prior:.6f},{s.sel_err_teacher:.6f},{s.sel_err_model_no_ts:.6f},{s.sel_err_mlp:.6f},"
            f"{s.qerr_prior:.6f},{s.qerr_teacher:.6f},{s.qerr_model_no_ts:.6f},{s.qerr_mlp:.6f}"
        )""",
"""        "q_mods,n,mae_prior,mae_teacher,mae_mlp,"
        "sel_err_prior,sel_err_teacher,sel_err_mlp,"
        "qerr_prior,qerr_teacher,qerr_mlp"
    ]
    for s in summaries:
        lines.append(
            f"{s.q_mods},{s.n},"
            f"{s.mae_prior:.6f},{s.mae_teacher:.6f},{s.mae_mlp:.6f},"
            f"{s.sel_err_prior:.6f},{s.sel_err_teacher:.6f},{s.sel_err_mlp:.6f},"
            f"{s.qerr_prior:.6f},{s.qerr_teacher:.6f},{s.qerr_mlp:.6f}"
        )""")

# plot_results
text = text.replace(
"""    fig.suptitle("Histogram Correction: Drift Intensity vs. Error\\n"
                 "(Training: q=[10,20], k=1000 each; 3 methods vs Prior)", fontsize=12)

    colors = {
        "Prior":       "#e74c3c",
        "Teacher":     "#f39c12",
        "Ridge":       "#2980b9",
        "MLP (attn)": "#8e44ad",
    }
    styles = {
        "Prior":       ("o-",  2.0),
        "Teacher":     ("s--", 2.0),
        "Ridge":       ("D:",  1.8),
        "MLP (attn)": ("^-",  2.2),
    }

    metrics = [
        ("Q-Error (↓ better)",
         [s.qerr_prior for s in summaries],
         [s.qerr_teacher for s in summaries],
         [s.qerr_model_no_ts for s in summaries],
         [s.qerr_mlp for s in summaries]),
        ("Selectivity MAE (↓ better)",
         [s.sel_err_prior for s in summaries],
         [s.sel_err_teacher for s in summaries],
         [s.sel_err_model_no_ts for s in summaries],
         [s.sel_err_mlp for s in summaries]),
        ("Quantile MAE (↓ better)",
         [s.mae_prior for s in summaries],
         [s.mae_teacher for s in summaries],
         [s.mae_model_no_ts for s in summaries],
         [s.mae_mlp for s in summaries]),
    ]

    for ax, (title, prior_vals, teacher_vals, ridge_vals, mlp_vals) in zip(axes, metrics):
        for name, vals in [
            ("Prior",       prior_vals),
            ("Teacher",     teacher_vals),
            ("Ridge",       ridge_vals),
            ("MLP (attn)", mlp_vals),
        ]:""",
"""    fig.suptitle("Histogram Correction: Drift Intensity vs. Error\\n"
                 "(Training: q=[10,20], k=1000 each; 2 methods vs Prior)", fontsize=12)

    colors = {
        "Prior":       "#e74c3c",
        "Teacher":     "#f39c12",
        "MLP (attn)": "#8e44ad",
    }
    styles = {
        "Prior":       ("o-",  2.0),
        "Teacher":     ("s--", 2.0),
        "MLP (attn)": ("^-",  2.2),
    }

    metrics = [
        ("Q-Error (↓ better)",
         [s.qerr_prior for s in summaries],
         [s.qerr_teacher for s in summaries],
         [s.qerr_mlp for s in summaries]),
        ("Selectivity MAE (↓ better)",
         [s.sel_err_prior for s in summaries],
         [s.sel_err_teacher for s in summaries],
         [s.sel_err_mlp for s in summaries]),
        ("Quantile MAE (↓ better)",
         [s.mae_prior for s in summaries],
         [s.mae_teacher for s in summaries],
         [s.mae_mlp for s in summaries]),
    ]

    for ax, (title, prior_vals, teacher_vals, mlp_vals) in zip(axes, metrics):
        for name, vals in [
            ("Prior",       prior_vals),
            ("Teacher",     teacher_vals),
            ("MLP (attn)", mlp_vals),
        ]:""")

text = text.replace(
"""    # 打印最终汇总表（Prior / Teacher / Ridge / MLP）
    print("\\n" + "=" * 90)
    print(f"{'q':>4} {'MAE_Prior':>10} {'MAE_Teacher':>12} {'MAE_Ridge':>10} {'MAE_MLP':>9} "
          f"{'QErr_Prior':>11} {'QErr_Teacher':>13} {'QErr_Ridge':>11} {'QErr_MLP':>10}")
    print("-" * 90)
    for s in summaries:
        print(f"{s.q_mods:>4} {s.mae_prior:>10.5f} {s.mae_teacher:>12.5f} "
              f"{s.mae_model_no_ts:>10.5f} {s.mae_mlp:>9.5f} "
              f"{s.qerr_prior:>11.4f} {s.qerr_teacher:>13.4f} "
              f"{s.qerr_model_no_ts:>11.4f} {s.qerr_mlp:>10.4f}")""",
"""    # 打印最终汇总表（Prior / Teacher / MLP）
    print("\\n" + "=" * 70)
    print(f"{'q':>4} {'MAE_Prior':>10} {'MAE_Teacher':>12} {'MAE_MLP':>9} "
          f"{'QErr_Prior':>11} {'QErr_Teacher':>13} {'QErr_MLP':>10}")
    print("-" * 70)
    for s in summaries:
        print(f"{s.q_mods:>4} {s.mae_prior:>10.5f} {s.mae_teacher:>12.5f} "
              f"{s.mae_mlp:>9.5f} "
              f"{s.qerr_prior:>11.4f} {s.qerr_teacher:>13.4f} "
              f"{s.qerr_mlp:>10.4f}")""")

with open('/Users/qichutian/presto/presto-cdf-simulation/cdf_kll_ml_pipeline/ablation_experiment.py', 'w') as f:
    f.write(text)

