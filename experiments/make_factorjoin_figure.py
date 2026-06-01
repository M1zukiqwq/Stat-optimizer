#!/usr/bin/env python3
"""Publication figure for the FactorJoin + OASIS integration.

Presentation-only: reads ``factorjoin_summary.csv`` produced by
``factorjoin_oasis_experiment.py`` and renders a stale-to-fresh-normalized view
of join-cardinality Q-error per drift intensity. The point of the figure is the
contrast between deployed/calibrated marginals and the fresh-marginal floor. The
no-projection ablation is intentionally left to Table~9 so the figure scale
emphasizes deployable methods. No experiment is rerun here.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
SUMMARY = ROOT / "experiments" / "results" / "factorjoin_oasis_20260531" / "factorjoin_summary.csv"

COLORS = {
    "Stale": "#666666",
    "ISOMER": "#0072B2",
    "OASIS": "#D55E00",
    "Hybrid": "#CC79A7",
    "Fresh": "#000000",
}
MARKERS = {
    "ISOMER": "D",
    "OASIS": "X",
    "Hybrid": "v",
}
# (column suffix in CSV, legend label) in plotting order.
# Naming: OASIS = learned repair + feedback-consistency projection;
# the stage-1-only ablation remains in Table 9.
SERIES = [
    ("isomer", "ISOMER"),
    ("oasis_projected", "OASIS"),
    ("hybrid", "Hybrid"),
]


def apply_style() -> None:
    plt.rcParams.update({
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
        "axes.axisbelow": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main() -> None:
    apply_style()
    with SUMMARY.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["drift_q"] != "all"]
    rows.sort(key=lambda r: int(r["drift_q"]))
    drifts = [r["drift_q"] for r in rows]

    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    x = np.array([int(d) for d in drifts])

    for key, label in SERIES:
        residual_gap = []
        for r in rows:
            stale = float(r["stale_qerr_gm"])
            fresh = float(r["fresh_qerr_gm"])
            value = float(r[f"{key}_qerr_gm"])
            residual_gap.append((value - fresh) / (stale - fresh) * 100.0)
        ax.plot(
            x,
            residual_gap,
            label=label,
            color=COLORS[label],
            marker=MARKERS[label],
            linewidth=1.7,
            markersize=4.8,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}" for d in drifts])
    ax.set_xlabel("Drift intensity q")
    ax.set_ylabel("Residual gap (%)")
    ax.set_ylim(0, 20)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.axhline(0, color=COLORS["Fresh"], linewidth=0.9, zorder=2)
    ax.text(x[-1] + 0.3, 0.5, "Fresh", ha="left", va="bottom",
            fontsize=7, color=COLORS["Fresh"])
    ax.text(0.985, 0.94, "Stale = 100% (not plotted)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.8, color=COLORS["Stale"])
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18),
              frameon=False, columnspacing=1.1, handlelength=1.5)

    fig.tight_layout(pad=0.6)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_factorjoin_oasis.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'fig_factorjoin_oasis.pdf'}")


if __name__ == "__main__":
    main()
