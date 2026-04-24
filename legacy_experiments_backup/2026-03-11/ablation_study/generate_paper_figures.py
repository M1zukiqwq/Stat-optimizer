#!/usr/bin/env python3
"""生成论文用图表（仅 Prior vs OASIS）"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

results_file = Path("/Users/qichutian/presto/presto-cdf-simulation/ablation_study/work_simplified/results/simplified_results.json")
with open(results_file) as f:
    data = json.load(f)

q_values = sorted(set(d['q_mods'] for d in data))

prior_qerr = [next(d['qerror_mean'] for d in data if d['q_mods'] == q and d['model_name'] == 'Prior') for q in q_values]
oasis_qerr = [next(d['qerror_mean'] for d in data if d['q_mods'] == q and d['model_name'] == 'OASIS') for q in q_values]

prior_sel = [next(d['sel_error_mean'] for d in data if d['q_mods'] == q and d['model_name'] == 'Prior') for q in q_values]
oasis_sel = [next(d['sel_error_mean'] for d in data if d['q_mods'] == q and d['model_name'] == 'OASIS') for q in q_values]

prior_mae = [next(d['quantile_mae_mean'] for d in data if d['q_mods'] == q and d['model_name'] == 'Prior') for q in q_values]
oasis_mae = [next(d['quantile_mae_mean'] for d in data if d['q_mods'] == q and d['model_name'] == 'OASIS') for q in q_values]

output_dir = Path("/Users/qichutian/presto/presto-cdf-simulation/paper/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# 通用样式
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# 图1: Q-Error
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.plot(q_values, prior_qerr, 'o-', color='#e74c3c', label='Stale Prior', linewidth=2, markersize=6)
ax.plot(q_values, oasis_qerr, 's-', color='#2ecc71', label='OASIS', linewidth=2, markersize=6)
ax.set_xlabel('Drift Intensity ($q$)')
ax.set_ylabel('Q-Error ($\\downarrow$ better)')
ax.set_xticks(q_values)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / 'ablation_qerror.pdf', bbox_inches='tight')
print(f"Saved: {output_dir / 'ablation_qerror.pdf'}")
plt.close()

# 图2: Selectivity Error
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.plot(q_values, prior_sel, 'o-', color='#e74c3c', label='Stale Prior', linewidth=2, markersize=6)
ax.plot(q_values, oasis_sel, 's-', color='#2ecc71', label='OASIS', linewidth=2, markersize=6)
ax.set_xlabel('Drift Intensity ($q$)')
ax.set_ylabel('Selectivity MAE ($\\downarrow$ better)')
ax.set_xticks(q_values)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / 'ablation_selerror.pdf', bbox_inches='tight')
print(f"Saved: {output_dir / 'ablation_selerror.pdf'}")
plt.close()

# 图3: Quantile MAE
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.plot(q_values, prior_mae, 'o-', color='#e74c3c', label='Stale Prior', linewidth=2, markersize=6)
ax.plot(q_values, oasis_mae, 's-', color='#2ecc71', label='OASIS', linewidth=2, markersize=6)
ax.set_xlabel('Drift Intensity ($q$)')
ax.set_ylabel('Quantile MAE ($\\downarrow$ better)')
ax.set_xticks(q_values)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / 'ablation_mae.pdf', bbox_inches='tight')
print(f"Saved: {output_dir / 'ablation_mae.pdf'}")
plt.close()

print("\n✓ 所有图表已生成！")
