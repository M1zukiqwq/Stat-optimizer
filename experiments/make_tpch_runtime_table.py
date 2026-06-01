#!/usr/bin/env python3
"""Presentation-only: build the TPC-H multi-seed runtime/accuracy LaTeX table.

Aggregates the three refresh-stream seeds
(experiments/results/postgres_runtime_tpch_seed{1,2,3}_20260601/) into mean +/- std
per method on the "all" subset, and emits table_tpch_runtime.tex. No experiment is
run; numbers are read from the per-seed summary CSVs.

Key reported quantities (all on the 6 curated TPC-H date-sensitive queries):
  * scan Q-error on the drifting date column (accuracy),
  * warm-cache execution time relative to FRESH statistics (the robust invariant:
    calibrated statistics reproduce fresh-statistics runtime),
  * warm-cache execution time relative to STALE (reported for completeness; note
    that stale's wall-clock can be coincidentally low because a mis-estimate may
    pick a cheaper-to-run plan -- a cost-model artifact, not an OASIS effect).
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
SEED_DIRS = [_REPO / "experiments" / "results" / f"postgres_runtime_tpch_seed{s}_20260601" for s in (1, 2, 3)]
OUT = _REPO / "experiments" / "results" / "postgres_runtime_tpch_seed1_20260601" / "table_tpch_runtime.tex"

LABEL = {
    "stale": "Stale",
    "isomer": "ISOMER",
    "oasis": "OASIS-noProj",
    "oasis_projected": "OASIS",
    "calibrated_hybrid": "Router",
    "fresh": "Fresh",
}
ORDER = ["stale", "isomer", "oasis", "oasis_projected", "calibrated_hybrid", "fresh"]


def load_all_subset(d: Path):
    rows = {}
    for r in csv.DictReader(open(d / "tpch_runtime_summary.csv")):
        if r["subset"] == "all":
            rows[r["method"]] = r
    return rows


def ms(vals):
    m = statistics.mean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s


def main() -> None:
    seeds = [load_all_subset(d) for d in SEED_DIRS]

    agg = {}
    for method in ORDER:
        qerr = [float(s[method]["scan_qerr_gm"]) for s in seeds]
        vfresh = [float(s[method]["geomean_ratio_vs_fresh"]) for s in seeds]
        vstale = [float(s[method]["geomean_ratio_vs_stale"]) for s in seeds]
        agg[method] = {"qerr": ms(qerr), "vfresh": ms(vfresh), "vstale": ms(vstale)}

    def cell(pair, nd=2):
        m, s = pair
        return f"{m:.{nd}f}\\,$\\pm$\\,{s:.{nd}f}"

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{TPC-H SF10 DML-drift study, aggregated over three "
                 r"independent refresh-stream seeds (mean\,$\pm$\,std; six curated "
                 r"date-sensitive queries each; planner-only statistics injection, "
                 r"PostgreSQL~14). The pretrained OASIS prior is reused with no TPC-H "
                 r"retraining. \emph{Scan Q-err} is the row-estimation error on the "
                 r"drifting date column. \emph{Time/Fresh} and \emph{Time/Stale} are "
                 r"geometric-mean ratios of warm-cache \texttt{EXPLAIN ANALYZE} "
                 r"execution time to the fresh- and stale-statistics runs. Calibrated "
                 r"statistics (OASIS, Router) match fresh statistics on both accuracy "
                 r"and runtime (Time/Fresh $\approx 1$). The stale wall-clock baseline "
                 r"is not a reliable target: a stale mis-estimate can pick a "
                 r"coincidentally cheaper-to-run plan (Time/Stale $>1$ here), whereas "
                 r"in other configurations it picks a much slower one; we therefore "
                 r"make no runtime-superiority claim and report that OASIS reproduces "
                 r"fresh-statistics behavior.}")
    lines.append(r"\label{tab:tpch_runtime}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Scan Q-err & Time/Fresh & Time/Stale \\")
    lines.append(r"\midrule")
    for m in ORDER:
        a = agg[m]
        lines.append(f"{LABEL[m]} & {cell(a['qerr'])} & {cell(a['vfresh'])} & {cell(a['vstale'])} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    # also echo the aggregate for the writeup
    for m in ORDER:
        a = agg[m]
        print(f"{m:18} qerr={a['qerr'][0]:.2f}±{a['qerr'][1]:.2f}  "
              f"vFresh={a['vfresh'][0]:.3f}±{a['vfresh'][1]:.3f}  "
              f"vStale={a['vstale'][0]:.3f}±{a['vstale'][1]:.3f}")


if __name__ == "__main__":
    main()
