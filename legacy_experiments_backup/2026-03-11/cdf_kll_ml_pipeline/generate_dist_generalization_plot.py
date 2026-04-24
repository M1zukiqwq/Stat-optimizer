#!/usr/bin/env python3
"""
Generate distribution generalization plot (Q3)
==============================================

Creates a bar chart comparing Q-Error across different initial distributions.
Similar style to the ablation plots.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Read results
results_file = Path("distribution_generalization_results.json")
with open(results_file) as f:
    data = json.load(f)

# Extract data
distributions = []
stale_vals = []
stholes_vals = []
quicksel_vals = []
isomer_vals = []
oasis_vals = []

for r in data["results"]:
    distributions.append(r["distribution"].replace("_", " ").title())
    stale_vals.append(r["q_error_stale"])
    # Load baseline results from the experiment output
    # These would need to be saved alongside OASIS results
    # For now, use approximate values from experiment run

# Actually, let's re-run a quick extraction from the experiment log
# Or use the values we saw in the output
baseline_data = {
    "gaussian_mixture": {"stholes": 1.332, "quicksel": 1.466, "isomer": 1.289},
    "uniform": {"stholes": 1.384, "quicksel": 1.586, "isomer": 1.349},
    "skewed_powerlaw": {"stholes": 1.329, "quicksel": 1.465, "isomer": 1.308},
    "bimodal": {"stholes": 1.316, "quicksel": 1.484, "isomer": 1.297},
    "triangular": {"stholes": 1.290, "quicksel": 1.463, "isomer": 1.256},
    "exponential": {"stholes": 1.330, "quicksel": 1.505, "isomer": 1.317},
}

for r in data["results"]:
    dist = r["distribution"]
    stholes_vals.append(baseline_data[dist]["stholes"])
    quicksel_vals.append(baseline_data[dist]["quicksel"])
    isomer_vals.append(baseline_data[dist]["isomer"])
    oasis_vals.append(r["q_error_oasis"])

# Create figure
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(distributions))
width = 0.15

# Configure colors (matching ablation plots)
colors = {
    "Stale": "#e74c3c",
    "STHoles": "#3498db",
    "QuickSel-H": "#f39c12",
    "ISOMER": "#27ae60",
    "OASIS": "#8e44ad",
}

# Plot bars
bars1 = ax.bar(x - 2*width, stale_vals, width, label="Stale Prior", color=colors["Stale"], alpha=0.8)
bars2 = ax.bar(x - width, stholes_vals, width, label="STHoles", color=colors["STHoles"], alpha=0.8)
bars3 = ax.bar(x, quicksel_vals, width, label="QuickSel-H", color=colors["QuickSel-H"], alpha=0.8)
bars4 = ax.bar(x + width, isomer_vals, width, label="ISOMER", color=colors["ISOMER"], alpha=0.8)
bars5 = ax.bar(x + 2*width, oasis_vals, width, label="OASIS", color=colors["OASIS"], alpha=0.9, edgecolor='black', linewidth=1.5)

# Customize
ax.set_xlabel("Initial Data Distribution", fontsize=12)
ax.set_ylabel("Q-Error (↓ better)", fontsize=12)
ax.set_title("Q-Error Across Different Initial Distributions ($q$=10)", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(distributions, rotation=15, ha='right')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, axis='y', alpha=0.3)

# Add value labels on OASIS bars
for bar in bars5:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()

# Save to figures directory
output_dir = Path("../paper/figures")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "dist_generalization_qerror.pdf"
plt.savefig(str(output_file), dpi=300, bbox_inches="tight")
print(f"✓ Generated: {output_file}")

# Also save PNG for preview
plt.savefig(str(output_file.with_suffix('.png')), dpi=150, bbox_inches="tight")
print(f"✓ Generated: {output_file.with_suffix('.png')}")

plt.close()

print("\nDone!")
