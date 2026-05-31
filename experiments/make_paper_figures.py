#!/usr/bin/env python3
"""Generate publication figures from cached experiment summaries.

The script intentionally reads the CSV summaries produced by the experiment
drivers instead of duplicating results.  It is presentation-only: no experiment
is rerun here.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"

# Naming: OASIS = full two-stage system (learned repair + feedback-consistency
# projection); OASIS-noProj = stage-1-only ablation (the raw learned marginal).
COLORS = {
    "Stale": "#666666",
    "Prior": "#666666",
    "STHoles": "#56B4E9",
    "QuickSel-H": "#009E73",
    "ISOMER": "#0072B2",
    "OASIS-noProj": "#E69F00",
    "OASIS": "#D55E00",
    "OASIS-Proj": "#D55E00",
    "Hybrid": "#CC79A7",
    "Fresh": "#000000",
}

MARKERS = {
    "Stale": "o",
    "Prior": "o",
    "STHoles": "s",
    "QuickSel-H": "^",
    "ISOMER": "D",
    "OASIS-noProj": "P",
    "OASIS": "X",
    "OASIS-Proj": "X",
    "Hybrid": "v",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.85,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def method_label(method: str) -> str:
    labels = {
        "stale": "Stale",
        "prior": "Stale",
        "isomer": "ISOMER",
        "oasis": "OASIS-noProj",
        "oasis_projected": "OASIS",
        "hybrid": "Hybrid",
        "fresh": "Fresh",
    }
    return labels.get(method, method)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        va="top",
        ha="left",
    )


def set_qerror_log_axis(ax: plt.Axes, ticks: list[float], ylim: tuple[float, float]) -> None:
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{tick:g}" for tick in ticks])
    ax.minorticks_off()


def plot_single_column_drift() -> None:
    rows = read_csv(
        ROOT
        / "experiments"
        / "results"
        / "synthetic_paper_suite_rerun_20260529"
        / "main"
        / "summary.csv"
    )
    methods = ["Prior", "STHoles", "QuickSel-H", "ISOMER", "OASIS"]
    label_map = {"Prior": "Stale", "OASIS": "OASIS-noProj"}
    q_values = sorted({int(r["q_mods"]) for r in rows if r["method"] in methods})
    by_method = {
        method: {
            int(r["q_mods"]): float(r["qerror_mean"])
            for r in rows
            if r["method"] == method
        }
        for method in methods
    }

    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    for method in methods:
        label = label_map.get(method, method)
        ys = [by_method[method][q] for q in q_values]
        ax.plot(
            q_values,
            ys,
            label=label,
            color=COLORS[label],
            marker=MARKERS[label],
            linewidth=1.8,
            markersize=4.5,
        )

    ax.set_xlabel("Drift intensity q")
    ax.set_ylabel("Selectivity Q-error (lower is better)")
    ax.set_xticks(q_values)
    ax.set_ylim(1.0, 3.55)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    save_figure(fig, "fig_single_column_drift_qerror")


def plot_ood_drift_realism() -> None:
    rows = read_csv(
        ROOT
        / "experiments"
        / "results"
        / "ood_drift_realism_20260529"
        / "summary.csv"
    )
    patterns = [
        ("batch_load", "Batch\nload"),
        ("range_shift", "Range\nshift"),
        ("skew_evol", "Skew\nevolution"),
        ("outlier", "Outlier\nburst"),
        ("multimodal", "Multimodal"),
        ("seasonal", "Seasonal\nmixed"),
    ]
    bar_methods = ["isomer", "oasis", "oasis_projected"]
    by_key = {(r["pattern"], r["method"]): float(r["qerror_gm"]) for r in rows}

    fig, ax = plt.subplots(figsize=(7.1, 3.45))
    x = list(range(len(patterns)))
    width = 0.20
    offsets = [(-1 + i) * width for i in range(len(bar_methods))]
    for method, offset in zip(bar_methods, offsets):
        label = method_label(method)
        ys = [by_key[(pattern, method)] for pattern, _ in patterns]
        ax.bar(
            [i + offset for i in x],
            [value - 1.0 for value in ys],
            bottom=1.0,
            width=width,
            label=label,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.5,
        )

    stale_y = [by_key[(pattern, "stale")] for pattern, _ in patterns]
    ax.plot(
        x,
        stale_y,
        linestyle="none",
        marker="o",
        markersize=4.8,
        markerfacecolor="white",
        markeredgecolor=COLORS["Stale"],
        markeredgewidth=1.2,
        label="Stale",
        zorder=5,
    )
    for xi, yi in zip(x, stale_y):
        ax.text(xi, yi * 1.05, f"{yi:.1f}", ha="center", va="bottom", fontsize=6.8, color=COLORS["Stale"])

    ax.axhline(1.0, color=COLORS["Fresh"], linestyle=(0, (3, 2)), linewidth=1.0)
    ax.text(len(patterns) - 0.1, 1.015, "Fresh", ha="right", va="bottom", fontsize=7)
    ax.set_ylabel("Selectivity Q-error (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in patterns])
    set_qerror_log_axis(ax, ticks=[1.0, 1.2, 1.5, 2.0, 3.0, 4.0], ylim=(0.98, 4.9))
    ax.legend(ncol=4, frameon=False, loc="upper right")
    save_figure(fig, "fig_ood_drift_realism")


def plot_trace_grounded_drift() -> None:
    rows = read_csv(
        ROOT
        / "experiments"
        / "results"
        / "trace_grounded_drift_20260529"
        / "summary.csv"
    )
    traces = [
        ("tpcds_sales_append", "Sales\nappend"),
        ("promotion_price_revision", "Promotion\nprice"),
        ("inventory_restock", "Inventory\nrestock"),
        ("returns_cancellation", "Returns\ncancel"),
        ("customer_segment_churn", "Customer\nchurn"),
        ("seasonal_mixed_maintenance", "Seasonal\nmixed"),
    ]
    bar_methods = ["isomer", "oasis", "oasis_projected", "hybrid"]
    by_key = {(r["trace"], r["method"]): float(r["qerror_gm"]) for r in rows}

    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    x = list(range(len(traces)))
    width = 0.17
    offsets = [(-1.5 + i) * width for i in range(len(bar_methods))]
    for method, offset in zip(bar_methods, offsets):
        label = method_label(method)
        ys = [by_key[(trace, method)] for trace, _ in traces]
        ax.bar(
            [i + offset for i in x],
            [value - 1.0 for value in ys],
            bottom=1.0,
            width=width,
            label=label,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.5,
        )

    stale_y = [by_key[(trace, "stale")] for trace, _ in traces]
    ax.plot(
        x,
        stale_y,
        linestyle="none",
        marker="o",
        markersize=4.8,
        markerfacecolor="white",
        markeredgecolor=COLORS["Stale"],
        markeredgewidth=1.2,
        label="Stale",
        zorder=5,
    )
    for xi, yi in zip(x, stale_y):
        ax.text(xi, yi * 1.035, f"{yi:.1f}", ha="center", va="bottom", fontsize=6.8, color=COLORS["Stale"])

    ax.axhline(1.0, color=COLORS["Fresh"], linestyle=(0, (3, 2)), linewidth=1.0)
    ax.text(len(traces) - 0.1, 1.008, "Fresh", ha="right", va="bottom", fontsize=7)
    ax.set_ylabel("Selectivity Q-error (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in traces])
    set_qerror_log_axis(ax, ticks=[1.0, 1.1, 1.25, 1.5, 2.0], ylim=(0.99, 2.35))
    ax.legend(ncol=5, frameon=False, loc="upper right")
    save_figure(fig, "fig_trace_grounded_drift")


def plot_postgres_plan_family() -> None:
    rows = read_csv(
        ROOT
        / "experiments"
        / "results"
        / "postgres_planner_stats_injection_batch_20260529"
        / "plan_family_breakdown.csv"
    )
    families = [
        ("scan", "Selection"),
        ("join", "Join"),
        ("join_dim_filter", "Join + dim\nfilter"),
    ]
    methods = ["stale", "oasis", "isomer", "oasis_projected"]
    metrics = [
        ("fresh_plan_match_frac", "Fresh-plan match (%)", "A", 0, 100, "cividis"),
        ("plan_recovery_frac", "Recovered stale/fresh\nmismatches (%)", "B", 0, 100, "cividis"),
        ("new_plan_deviation_frac", "New deviations (%)", "C", 0, 20, "cividis_r"),
    ]
    by_key = {(r["family"], r["method"]): r for r in rows}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.05), sharey=True)
    for ax, (metric, title, label, vmin, vmax, cmap) in zip(axes, metrics):
        matrix = [
            [float(by_key[(family, method)][metric]) * 100.0 for family, _ in families]
            for method in methods
        ]
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for row_idx, method in enumerate(methods):
            for col_idx, (family, _) in enumerate(families):
                value = matrix[row_idx][col_idx]
                text_color = "white" if (cmap == "cividis" and value < 60) or (cmap == "cividis_r" and value > 10) else "black"
                ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", fontsize=7.2, color=text_color)
        panel_label(ax, label)
        ax.set_title(title)
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels([family_label for _, family_label in families], rotation=0)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([method_label(method) for method in methods])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=6.5, length=2)
    fig.subplots_adjust(wspace=0.35, top=0.82, bottom=0.18)
    save_figure(fig, "fig_postgres_plan_family")


def plot_feedback_sensitivity() -> None:
    budget_rows = read_csv(
        ROOT
        / "experiments"
        / "results"
        / "feedback_budget_sensitivity_20260529"
        / "summary.csv"
    )
    noise_rows = read_csv(
        ROOT
        / "experiments"
        / "results"
        / "feedback_noise_robustness_20260529"
        / "summary.csv"
    )
    methods = ["stale", "isomer", "oasis", "oasis_projected", "hybrid"]
    display_methods = [method_label(m) for m in methods]
    budget_x = sorted({int(r["feedback_k"]) for r in budget_rows})
    noise_x = sorted({float(r["noise_sigma"]) for r in noise_rows})
    budget = {(int(r["feedback_k"]), r["method"]): r for r in budget_rows}
    noise = {(float(r["noise_sigma"]), r["method"]): r for r in noise_rows}

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    panels = [
        (
            axes[0, 0],
            "A",
            budget_x,
            budget,
            "selectivity_qerr_gm",
            "Feedback window K",
            "Selectivity Q-error",
            (1.0, 3.05),
            False,
        ),
        (
            axes[0, 1],
            "B",
            budget_x,
            budget,
            "join_optimal_match_frac",
            "Feedback window K",
            "Join-optimal choices (%)",
            (78, 101),
            True,
        ),
        (
            axes[1, 0],
            "C",
            noise_x,
            noise,
            "selectivity_qerr_gm",
            "Feedback noise sigma",
            "Selectivity Q-error",
            (1.0, 3.05),
            False,
        ),
        (
            axes[1, 1],
            "D",
            noise_x,
            noise,
            "join_optimal_match_frac",
            "Feedback noise sigma",
            "Join-optimal choices (%)",
            (78, 101),
            True,
        ),
    ]

    for ax, letter, xs, data, metric, xlabel, ylabel, ylim, pct in panels:
        for method, display in zip(methods, display_methods):
            ys = []
            for x_val in xs:
                value = float(data[(x_val, method)][metric])
                ys.append(value * 100.0 if pct else value)
            ax.plot(
                xs,
                ys,
                label=display,
                color=COLORS[display],
                marker=MARKERS[display],
                linewidth=1.5,
                markersize=4.0,
            )
        if pct:
            ax.axhline(100.0, color=COLORS["Fresh"], linestyle=(0, (3, 2)), linewidth=0.9)
        else:
            ax.axhline(1.0, color=COLORS["Fresh"], linestyle=(0, (3, 2)), linewidth=0.9)
        panel_label(ax, letter)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_xticks(xs)
        if xlabel.startswith("Feedback noise"):
            ax.set_xticklabels([f"{int(x * 100)}%" for x in xs])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.subplots_adjust(hspace=0.48, wspace=0.38, top=0.86)
    save_figure(fig, "fig_feedback_sensitivity")


def main() -> None:
    apply_style()
    plot_single_column_drift()
    plot_ood_drift_realism()
    plot_trace_grounded_drift()
    plot_postgres_plan_family()
    plot_feedback_sensitivity()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
