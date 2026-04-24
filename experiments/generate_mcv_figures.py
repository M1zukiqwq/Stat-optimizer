#!/usr/bin/env python3
"""
Generate figures for MCV adapter validation using measured experiment outputs.

Input:
  experiments/results/mcv_validation_results.json

Output:
  paper/figures/mcv_performance_overhead.pdf
  paper/figures/mcv_threshold_sensitivity.pdf
  paper/figures/mcv_validation_summary.pdf
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["axes.labelsize"] = 9
matplotlib.rcParams["axes.titlesize"] = 10
matplotlib.rcParams["xtick.labelsize"] = 8
matplotlib.rcParams["ytick.labelsize"] = 8
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["figure.titlesize"] = 10


BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "results" / "mcv_validation_results.json"
OUTPUT_DIR = BASE_DIR.parent / "paper" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_results() -> Dict[str, object]:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing results file: {RESULTS_PATH}\n"
            "Run experiments/mcv_adapter_validation.py first."
        )
    with RESULTS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _group_performance_configs(results: Dict[str, object]) -> Dict[int, List[Dict[str, float]]]:
    grouped: Dict[int, List[Dict[str, float]]] = {}
    for row in results["performance_overhead"]["configs"]:
        grouped.setdefault(int(row["n_mcv"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["n_buckets"]))
    return grouped


def plot_performance_overhead(results: Dict[str, object]) -> None:
    grouped = _group_performance_configs(results)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    style = {
        10: ("o", "#2E86AB"),
        50: ("s", "#A23B72"),
        100: ("^", "#F18F01"),
    }

    for mcv_count in sorted(grouped.keys()):
        marker, color = style.get(mcv_count, ("o", "#444444"))
        rows = grouped[mcv_count]
        buckets = [int(row["n_buckets"]) for row in rows]
        times = [float(row["total_mean_ms"]) for row in rows]
        ax.plot(
            buckets,
            times,
            marker=marker,
            linewidth=2,
            markersize=5,
            color=color,
            alpha=0.9,
            label=f"MCV={mcv_count}",
        )

    ax.set_xlabel("Residual Bin Count")
    ax.set_ylabel("Round-Trip Time (ms)")
    ax.set_xticks(sorted({int(row["n_buckets"]) for rows in grouped.values() for row in rows}))
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)

    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(31, 1.03, "1 ms budget", fontsize=7, color="red", alpha=0.75)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mcv_performance_overhead.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'mcv_performance_overhead.pdf'}")
    plt.close()


def plot_threshold_sensitivity(results: Dict[str, object]) -> None:
    points = sorted(results["threshold_sensitivity"]["points"], key=lambda row: float(row["threshold"]))
    thresholds = [float(row["threshold"]) for row in points]
    mcv_counts = [int(row["mcv_count"]) for row in points]
    selectivity_mae = [max(float(row["selectivity_mae"]), 1e-8) for row in points]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))

    ax1.plot(thresholds, mcv_counts, marker="o", linewidth=2, markersize=7, color="#2E86AB", alpha=0.9)
    ax1.set_xlabel("MCV Frequency Threshold")
    ax1.set_ylabel("MCV Count")
    ax1.set_xscale("log")
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)
    ax1.annotate(
        "PostgreSQL default\n(~1%)",
        xy=(0.01, mcv_counts[thresholds.index(0.01)]),
        xytext=(0.013, max(mcv_counts) * 0.72),
        arrowprops={"arrowstyle": "->", "color": "red", "alpha": 0.6},
        fontsize=7,
        color="red",
        ha="left",
    )

    ax2.plot(thresholds, selectivity_mae, marker="s", linewidth=2, markersize=7, color="#A23B72", alpha=0.9)
    ax2.set_xlabel("MCV Frequency Threshold")
    ax2.set_ylabel("Selectivity MAE")
    ax2.set_xscale("log")
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)
    ax2.axvline(x=0.01, color="green", linestyle="--", linewidth=1, alpha=0.5)
    ax2.text(0.011, max(selectivity_mae) * 0.82, "PG default", fontsize=7, color="green")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mcv_threshold_sensitivity.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'mcv_threshold_sensitivity.pdf'}")
    plt.close()


def plot_combined_summary(results: Dict[str, object]) -> None:
    per_dist = results["round_trip_accuracy"]["per_distribution"]
    dist_labels = [str(row["distribution"]).capitalize() for row in per_dist]
    translation_error = [max(float(row["residual_quantile_mae_mean"]), 1e-18) for row in per_dist]

    grouped = _group_performance_configs(results)
    perf_rows = []
    for mcv_count in [10, 50, 100]:
        rows = grouped[mcv_count]
        representative = next((row for row in rows if int(row["n_buckets"]) == 20), rows[len(rows) // 2])
        perf_rows.append((mcv_count, float(representative["total_mean_ms"])))

    points = sorted(results["threshold_sensitivity"]["points"], key=lambda row: float(row["threshold"]))
    thresholds_pct = [float(row["threshold"]) * 100.0 for row in points]
    mae_pct = [max(float(row["selectivity_mae"]) * 100.0, 1e-6) for row in points]

    fig = plt.figure(figsize=(7, 2.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.45)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(len(dist_labels)), translation_error, color="#2E86AB", alpha=0.85)
    ax1.set_ylabel("Residual Quantile MAE", fontsize=8)
    ax1.set_yscale("log")
    ax1.set_xticks(range(len(dist_labels)))
    ax1.set_xticklabels(dist_labels, rotation=30, ha="right", fontsize=7)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)
    ax1.set_title("(a) Round-Trip Fidelity", fontsize=9, pad=5)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(len(perf_rows)), [row[1] for row in perf_rows], color="#A23B72", alpha=0.85)
    ax2.set_ylabel("Round-Trip Time (ms)", fontsize=8)
    ax2.set_xlabel("MCV Count", fontsize=8)
    ax2.set_xticks(range(len(perf_rows)))
    ax2.set_xticklabels([str(row[0]) for row in perf_rows], fontsize=7)
    ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax2.text(1.0, 1.03, "1 ms budget", fontsize=6, color="red", alpha=0.75, ha="center")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)
    ax2.set_title("(b) Performance Overhead", fontsize=9, pad=5)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(thresholds_pct, mae_pct, marker="o", linewidth=2, markersize=6, color="#F18F01", alpha=0.9)
    ax3.set_xlabel("MCV Threshold (%)", fontsize=8)
    ax3.set_ylabel("Selectivity MAE (%)", fontsize=8)
    ax3.axvline(x=1.0, color="green", linestyle="--", linewidth=1, alpha=0.5)
    ax3.text(1.08, max(mae_pct) * 0.78, "PG default", fontsize=6, color="green", rotation=90)
    ax3.grid(alpha=0.3, linestyle="--")
    ax3.set_axisbelow(True)
    ax3.set_title("(c) Threshold Sensitivity", fontsize=9, pad=5)

    plt.savefig(OUTPUT_DIR / "mcv_validation_summary.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'mcv_validation_summary.pdf'}")
    plt.close()


if __name__ == "__main__":
    print("=" * 80)
    print("Generating MCV Adapter Validation Figures")
    print("=" * 80)

    loaded = _load_results()
    plot_performance_overhead(loaded)
    plot_threshold_sensitivity(loaded)
    plot_combined_summary(loaded)

    print("=" * 80)
    print("All figures generated successfully.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)
