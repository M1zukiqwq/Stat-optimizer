import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['pdf.fonttype'] = 42

# Data aligned with table_distribution.tex
labels = ['Gaussian Mix.', 'Uniform', 'Power-law', 'Bimodal', 'Triangular', 'Exponential']
methods = ['Stale Prior', 'STHoles', 'QuickSel-H', 'ISOMER', 'OASIS']

data = {
    'Stale Prior': [2.497, 2.409, 3.517, 2.488, 2.225, 3.097],
    'STHoles': [1.453, 1.503, 1.679, 1.554, 1.420, 1.592],
    'QuickSel-H': [1.845, 1.977, 2.202, 2.069, 1.833, 2.141],
    'ISOMER': [1.493, 1.534, 1.609, 1.462, 1.505, 1.593],
    'OASIS': [1.262, 1.284, 1.379, 1.276, 1.238, 1.292],
}

colors = {
    'Stale Prior': '#d32f2f',
    'STHoles': '#ff9800',
    'QuickSel-H': '#9e9e9e',
    'ISOMER': '#009688',
    'OASIS': '#1976d2',
}
markers = {
    'Stale Prior': 's',
    'STHoles': '^',
    'QuickSel-H': 'D',
    'ISOMER': 'v',
    'OASIS': 'o',
}
linestyles = {
    'Stale Prior': '-',
    'STHoles': '--',
    'QuickSel-H': ':',
    'ISOMER': '-.',
    'OASIS': '-',
}

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8, 4.6))

for method in methods:
    ax.plot(
        x,
        data[method],
        label=method,
        color=colors[method],
        marker=markers[method],
        linestyle=linestyles[method],
        linewidth=2.0 if method == 'OASIS' else 1.7,
        markersize=6.5 if method == 'OASIS' else 5.8,
        markeredgecolor='black' if method == 'OASIS' else colors[method],
        markeredgewidth=0.8 if method == 'OASIS' else 0.0,
        alpha=0.95,
        zorder=3 if method == 'OASIS' else 2,
    )

ax.set_ylabel('Q-Error ($\\downarrow$)', fontsize=12)
ax.set_xlabel('Initial Distribution', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(1.1, 3.7)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.grid(axis='x', visible=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper left', ncol=2, fontsize=9.5, frameon=True, edgecolor='black', fancybox=False)

fig.tight_layout()
plt.savefig('paper/figures/dist_generalization_qerror.pdf', format='pdf', bbox_inches='tight')
plt.savefig('paper/figures/dist_generalization_qerror.png', format='png', dpi=300, bbox_inches='tight')
