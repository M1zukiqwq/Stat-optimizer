#!/usr/bin/env python3
"""Publication figure for the FactorJoin + OASIS integration.

Presentation-only: reads ``factorjoin_summary.csv`` produced by
``factorjoin_oasis_experiment.py`` and renders grouped bars of join-cardinality
Q-error per drift intensity. The point of the figure is the contrast between
plain OASIS (which can exceed the stale baseline in the bilinear join kernel) and
OASIS-Proj / Hybrid (which recover the gain and track the fresh-marginal floor).
No experiment is rerun here.
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
    "OASIS-noProj": "#E69F00",
    "ISOMER": "#0072B2",
    "OASIS": "#D55E00",
    "Hybrid": "#CC79A7",
    "Fresh": "#000000",
}
# (column suffix in CSV, legend label) in plotting order.
# Naming: OASIS = full two-stage system (learned repair + projection);
# OASIS-noProj = stage-1-only ablation (the raw learned marginal).
SERIES = [
    ("stale", "Stale"),
    ("oasis", "OASIS-noProj"),
    ("isomer", "ISOMER"),
    ("oasis_projected", "OASIS"),
    ("hybrid", "Hybrid"),
    ("fresh", "Fresh"),
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

    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    n_groups = len(rows)
    n_series = len(SERIES)
    group_w = 0.82
    bar_w = group_w / n_series
    x = np.arange(n_groups)

    for si, (key, label) in enumerate(SERIES):
        vals = [float(r[f"{key}_qerr_gm"]) for r in rows]
        offs = (si - (n_series - 1) / 2) * bar_w
        bars = ax.bar(x + offs, vals, bar_w, label=label,
                      color=COLORS[label], edgecolor="white", linewidth=0.4,
                      zorder=3)
        # Flag plain-OASIS bars that exceed the stale baseline (harmful regime).
        if key == "oasis":
            for gi, (b, v) in enumerate(zip(bars, vals)):
                stale_v = float(rows[gi]["stale_qerr_gm"])
                if v > stale_v + 1e-3:
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.006, "↑",
                            ha="center", va="bottom", fontsize=8,
                            color=COLORS["OASIS-noProj"], zorder=4)

    # Per-group stale reference line to make the "above stale = harmful" read.
    for gi, r in enumerate(rows):
        sv = float(r["stale_qerr_gm"])
        ax.plot([gi - group_w / 2, gi + group_w / 2], [sv, sv],
                color=COLORS["Stale"], linestyle=(0, (4, 2)), linewidth=0.8, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"$q={d}$" for d in drifts])
    ax.set_ylabel("Join-cardinality Q-error")
    ax.set_xlabel("Drift intensity")
    ax.set_ylim(1.0, max(float(r["oasis_qerr_gm"]) for r in rows) * 1.07)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.18),
              frameon=False, columnspacing=1.1, handlelength=1.2)
    ax.annotate("OASIS-noProj exceeds stale\n(bilinear join amplifies over-concentration)",
                xy=(0.985, 0.96), xycoords="axes fraction", ha="right", va="top",
                fontsize=6.6, color=COLORS["OASIS-noProj"])

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_factorjoin_oasis.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'fig_factorjoin_oasis.pdf'}")


if __name__ == "__main__":
    main()
