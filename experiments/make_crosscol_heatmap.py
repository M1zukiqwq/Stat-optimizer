#!/usr/bin/env python3
"""Render the cross-column repair table (tab:crosscol_feedback) as a heatmap.

Reads results/crosscol_feedback_v1/summary.csv (produced by
crosscol_feedback_experiment.py) and writes a publication-quality heatmap of the
geometric-mean conjunctive-predicate Q-error for the three methods
(AVI / 2D STHoles / OASIS feedback projection) across the real correlated column
pairs, ordered by |Pearson r|. Lower (paler) is better; every cell is annotated
with its Q-error, and the Pearson r is shown in each row label. The Aggregate row
is separated at the bottom.
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "results", "crosscol_feedback_v1", "summary.csv")
OUTDIR = os.path.join(HERE, os.pardir, "paper", "figures")

METHODS = ["AVI", "STHoles", "OASIS"]
COLS = ["avi_qerr", "sthole_qerr", "feedback_qerr"]


def load():
    pairs, agg = [], None
    with open(SUMMARY) as f:
        for row in csv.DictReader(f):
            rec = {
                "name": row["pair"],
                "r": row["pearson_r"],
                "vals": [float(row[c]) for c in COLS],
            }
            if row["pair"] == "AGGREGATE":
                agg = rec
            else:
                rec["r"] = float(row["pearson_r"])
                pairs.append(rec)
    pairs.sort(key=lambda z: abs(z["r"]))  # ascending |r|, matches the table
    return pairs, agg


def label(name, r=None):
    disp = name.replace("~", "/")
    return f"{disp}\n($r={r:+.2f}$)" if r is not None else disp


def main():
    pairs, agg = load()
    rows = pairs + [agg]
    M = np.array([rec["vals"] for rec in rows])
    ylabels = [label(p["name"], p["r"]) for p in pairs] + ["Aggregate"]

    vmin, vmax = float(M.min()), float(M.max())
    fig, ax = plt.subplots(figsize=(4.3, 5.0))
    im = ax.imshow(M, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    # mark the deployed method
    ax.get_xticklabels()[2].set_fontweight("bold")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.get_yticklabels()[-1].set_fontweight("bold")

    # annotate every cell with its Q-error; white text on dark cells
    norm = (M - vmin) / max(vmax - vmin, 1e-9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                    fontsize=8.5, color="white" if norm[i, j] > 0.6 else "black",
                    fontweight="bold" if j == 2 else "normal")

    # separator above the Aggregate row
    ax.axhline(len(pairs) - 0.5, color="black", lw=1.4)
    # light grid between cells
    ax.set_xticks(np.arange(-0.5, len(METHODS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"geom-mean conjunctive Q-error ($\downarrow$ better)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(OUTDIR, f"fig_crosscol_heatmap.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
