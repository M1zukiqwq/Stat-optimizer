#!/usr/bin/env python3
"""Emit the real-data single-column LaTeX table from the stage1swap v1 result CSVs.

Message: on real data (Power/Forest/Census), feedback projection repairs stale single-column
statistics, and a learned prior gives no further gain over the max-entropy projection
(ISOMER) -- i.e. single-column repair is "easy" and ML is dispensable there. Numbers are
read straight from results/real_<ds>_<tag>_v1/estimator_swap_overall.csv (no recompute).
"""
import csv
import os

DS = [("power", "Power"), ("forest", "Forest"), ("census", "Census")]
MODE = [("dense", "16"), ("sparse", "2")]
ROOT = "results"


def load(p):
    if not os.path.exists(p):
        return None
    return {r["source"]: r for r in csv.DictReader(open(p))}


def main():
    lines = [
        r"\begin{table}[t]", r"  \centering\small",
        r"  \caption{Single-column repair on held-out \emph{real} datasets (Power, Forest,",
        r"  Census), reusing the prior trained only on a disjoint pool of real columns. Stale is",
        r"  the uncorrected histogram; ISOMER is the maximum-entropy feedback projection; STHoles",
        r"  and QuickSel-H are self-tuning histograms; the learned prior is reported routed. All",
        r"  feedback methods cluster tightly and the learned prior gives \emph{no} gain over the",
        r"  max-entropy projection: on real single-column data, feedback projection is already",
        r"  near-optimal. Geometric-mean future-predicate Q-error ($\downarrow$).}",
        r"  \label{tab:real_singlecol}", r"  \setlength{\tabcolsep}{6pt}",
        r"  \begin{tabular}{l c r r r r r}", r"    \toprule",
        r"    Dataset & $K$ & Stale & ISOMER & STHoles & QuickSel-H & Learned \\",
        r"    \midrule",
    ]
    for ds, disp in DS:
        for tag, kl in MODE:
            R = load(os.path.join(ROOT, f"real_{ds}_{tag}_v1", "estimator_swap_overall.csv"))
            if not R:
                lines.append(f"    {disp} & {kl} & \\multicolumn{{5}}{{c}}{{(missing)}} \\\\")
                continue
            stale = float(R["stale"]["none_qerr"])
            iso = float(R["stale"]["hard_qerr"])
            sth = float(R["stholes"]["hard_qerr"])
            qs = float(R["quicksel_h"]["hard_qerr"])
            learned = float(R["oasis_mlp"]["router_qerr"])
            name = disp if tag == "dense" else ""
            lines.append(f"    {name} & {kl} & {stale:.3f} & {iso:.3f} & {sth:.3f} & {qs:.3f} & {learned:.3f} \\\\")
        lines.append(r"    \addlinespace[2pt]")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    out = os.path.join(ROOT, "real_singlecol_v1")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "table_real_singlecol.tex"), "w") as h:
        h.write("\n".join(lines) + "\n")
    print(f"wrote {out}/table_real_singlecol.tex")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
