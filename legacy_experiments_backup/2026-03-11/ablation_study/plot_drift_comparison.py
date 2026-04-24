#!/usr/bin/env python3
"""
Generate a combined 3-panel figure comparing OASIS improvement (%) across
three drift pattern types: Gaussian Compound Drift, SCD Type 2, Fact Table Growth.
"""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Data ────────────────────────────────────────────────────────────────────
# Gaussian compound drift (from ablation_study/work_final/results/final_results.csv)
gaussian_data = {
    "q":         [1,    3,    5,    10,   15,   20,   25,   30],
    "Prior":     [1.169,1.443,1.811,2.826,2.957,3.459,3.342,3.192],
    "STHoles":   [1.093,1.230,1.356,1.705,1.731,1.819,1.722,1.726],
    "ISOMER":    [1.089,1.207,1.306,1.623,1.714,1.749,1.673,1.615],
    "QuickSel-H":[1.186,1.316,1.498,1.939,2.041,2.168,2.084,1.992],
    "OASIS":     [1.151,1.179,1.223,1.249,1.308,1.300,1.312,1.400],
}
gaussian_imp = {
    "STHoles":    [6.5,  14.8, 25.2, 39.7, 41.5, 47.4, 48.5, 45.9],
    "ISOMER":     [6.9,  16.4, 27.9, 42.6, 42.0, 49.4, 49.9, 49.4],
    "QuickSel-H": [-1.4, 8.8,  17.3, 31.4, 31.0, 37.3, 37.6, 37.6],
    "OASIS":      [1.5,  18.3, 32.5, 55.8, 55.8, 62.4, 60.7, 56.1],
}

# SCD2 data (from scd2_ablation_results.csv)
scd2_q = [1, 5, 10, 15, 20, 25, 30]
scd2_data = {
    "Prior":      [1.060, 1.445, 2.090, 2.914, 3.738, 4.579, 4.899],
    "STHoles":    [1.042, 1.250, 1.558, 1.883, 2.185, 2.515, 2.498],
    "ISOMER":     [1.041, 1.246, 1.537, 1.839, 2.183, 2.455, 2.558],
    "QuickSel-H": [1.254, 1.514, 1.957, 2.451, 2.876, 3.302, 3.469],
    "OASIS":      [1.135, 1.182, 1.351, 1.465, 1.471, 1.521, 1.463],
}
def compute_imp(prior_list, method_list):
    return [(p - m) / p * 100 for p, m in zip(prior_list, method_list)]

scd2_imp = {k: compute_imp(scd2_data["Prior"], scd2_data[k])
            for k in ["STHoles", "ISOMER", "QuickSel-H", "OASIS"]}

# Fact Table data
fact_q = [1, 5, 10, 15, 20, 25, 30]
fact_data = {
    "Prior":      [1.049, 1.330, 1.766, 2.171, 2.465, 3.035, 3.383],
    "STHoles":    [1.039, 1.185, 1.406, 1.529, 1.651, 1.871, 1.926],
    "ISOMER":     [1.034, 1.185, 1.377, 1.527, 1.640, 1.851, 1.961],
    "QuickSel-H": [1.245, 1.431, 1.714, 1.954, 2.100, 2.377, 2.531],
    "OASIS":      [1.117, 1.184, 1.268, 1.364, 1.408, 1.478, 1.502],
}
fact_imp = {k: compute_imp(fact_data["Prior"], fact_data[k])
            for k in ["STHoles", "ISOMER", "QuickSel-H", "OASIS"]}

# ── Plot ─────────────────────────────────────────────────────────────────────
methods = ["STHoles", "ISOMER", "QuickSel-H", "OASIS"]
colors = {
    "STHoles":    "#3498db",
    "ISOMER":     "#f39c12",
    "QuickSel-H": "#9b59b6",
    "OASIS":      "#27ae60",
}
markers = {"STHoles": "^", "ISOMER": "s", "QuickSel-H": "D", "OASIS": "v"}
linestyles = {"STHoles": "--", "ISOMER": "-.", "QuickSel-H": ":", "OASIS": "-"}

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'legend.fontsize': 8.5,
    'axes.titlesize': 10.5,
})

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=False)
panels = [
    ("Gaussian Compound Drift\n(Training Distribution)",
     gaussian_data["q"],
     {m: gaussian_data[m] for m in ["Prior", "STHoles", "ISOMER", "QuickSel-H", "OASIS"]}),
    ("SCD Type 2\n(Active/Historical Switch)",
     scd2_q,
     {m: scd2_data[m] for m in ["Prior", "STHoles", "ISOMER", "QuickSel-H", "OASIS"]}),
    ("Fact Table Growth\n(Monotonic Trend Drift)",
     fact_q,
     {m: fact_data[m] for m in ["Prior", "STHoles", "ISOMER", "QuickSel-H", "OASIS"]}),
]

method_style = {
    "Prior":      dict(color="#e74c3c", marker="o",  linestyle="--", linewidth=1.5),
    "STHoles":    dict(color="#3498db", marker="^",  linestyle="--", linewidth=1.5),
    "ISOMER":     dict(color="#f39c12", marker="s",  linestyle="-.", linewidth=1.5),
    "QuickSel-H": dict(color="#9b59b6", marker="D",  linestyle=":",  linewidth=1.5),
    "OASIS":      dict(color="#27ae60", marker="v",  linestyle="-",  linewidth=2.5),
}
method_labels = {
    "Prior": "Stale Prior", "STHoles": "STHoles",
    "ISOMER": "ISOMER", "QuickSel-H": "QuickSel-H", "OASIS": "OASIS (ours)",
}

for ax, (title, q_vals, qerr_dict) in zip(axes, panels):
    for method in ["Prior", "STHoles", "ISOMER", "QuickSel-H", "OASIS"]:
        s = method_style[method]
        ax.plot(q_vals, qerr_dict[method],
                marker=s["marker"], color=s["color"],
                label=method_labels[method],
                linewidth=s["linewidth"],
                linestyle=s["linestyle"],
                markersize=5.5,
                zorder=4 if method == "OASIS" else 2)
    ax.set_title(title, pad=6)
    ax.set_xlabel('Drift Intensity ($q$)')
    ax.set_ylabel('Q-Error ($\\downarrow$ better)')
    ax.set_xticks(q_vals)
    ax.grid(True, alpha=0.25, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Shared legend at bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5,
           frameon=True, bbox_to_anchor=(0.5, -0.06), fontsize=9)

plt.tight_layout(rect=[0, 0.05, 1, 1])
out = "/Users/qichutian/presto/presto-cdf-simulation/paper/figures/drift_pattern_comparison.pdf"
plt.savefig(out, bbox_inches='tight', dpi=150)
print(f"Saved: {out}")
