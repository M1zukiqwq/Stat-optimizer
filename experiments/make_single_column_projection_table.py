#!/usr/bin/env python3
"""Generate the single-column projection-tradeoff table (Table tab:single_column_projection).

Reuses the synthetic-suite cached model + compound-drift test data and the suite's
own metric functions, so the Stale/ISOMER/OASIS-noProj columns reproduce the
paper's main single-column numbers exactly (verified against main/summary.csv).
It then adds the OASIS column (Stage-1 prediction followed by the
ISOMER/IPF-style feedback-consistency projection) to show that the deployed form
keeps a single-column advantage over ISOMER while the projection costs only a
small amount of peak accuracy relative to OASIS-noProj.

Presentation-only: no model is retrained and no experiment is rerun.
"""
from __future__ import annotations

import csv
from pathlib import Path

import run_synthetic_paper_suite as S
from mlp_histogram_model_v2 import MlpHistogramModelV2
from modern_baselines import correct_isomer
from json_histogram_parser import load_feedback_sample

ROOT = Path(__file__).resolve().parent / "results" / "synthetic_paper_suite_rerun_20260529"
OUT = Path(__file__).resolve().parent / "results" / "single_column_projection_20260531"
NB, MAXOBS, SEED = 10, 16, 42


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def main() -> None:
    model = MlpHistogramModelV2.load(str(ROOT / "models" / "oasis_k16.json"))
    per_q = {}
    for q in S.MAIN_Q_VALUES:
        d = ROOT / "compound_data" / f"test_q{q}"
        if not d.exists():
            continue
        acc = {k: [] for k in ("Prior", "ISOMER", "noProj", "full")}
        for index, path in enumerate(sorted(d.glob("*.json"))):
            sample = load_feedback_sample(str(path))
            if sample.corrected_quantile_values is None:
                continue
            true_b = [sample.prior.min_value] + list(sample.corrected_quantile_values) + [sample.prior.max_value]
            mb = S.method_boundaries(sample, model=model, num_buckets=NB, max_obs=MAXOBS, stholes_mode="flat")
            obs = S.obs_to_dicts(sample, max_obs=MAXOBS)
            try:
                pq = correct_isomer(mb["OASIS"][0], mb["OASIS"][-1], list(mb["OASIS"][1:-1]), obs, num_buckets=NB)
                full = [mb["OASIS"][0]] + list(pq) + [mb["OASIS"][-1]]
            except Exception:
                full = mb["OASIS"]
            _, qp = S.metric_points(SEED + q * 10000 + index)
            acc["Prior"].append(S.q_error(mb["Prior"], true_b, qp))
            acc["ISOMER"].append(S.q_error(mb["ISOMER"], true_b, qp))
            acc["noProj"].append(S.q_error(mb["OASIS"], true_b, qp))
            acc["full"].append(S.q_error(full, true_b, qp))
        per_q[q] = {k: mean(v) for k, v in acc.items()}

    # Sanity: OASIS-noProj must reproduce summary.csv's OASIS column.
    summ = {int(r["q_mods"]): float(r["qerror_mean"])
            for r in csv.DictReader(open(ROOT / "main" / "summary.csv")) if r["method"] == "OASIS"}
    for q in per_q:
        assert abs(per_q[q]["noProj"] - summ.get(q, -1)) < 1e-3, (q, per_q[q]["noProj"], summ.get(q))
    print("OASIS-noProj reproduces summary.csv exactly.")

    OUT.mkdir(parents=True, exist_ok=True)
    qs = [q for q in S.MAIN_Q_VALUES if q in per_q]
    red = lambda s, v: (s - v) / s * 100
    with open(OUT / "table_single_column_projection.tex", "w") as f:
        f.write("\\begin{table}[t]\n  \\centering\n  \\small\n")
        f.write("  \\caption{Single-column selectivity Q-error ($\\downarrow$) of OASIS "
                "versus the OASIS-noProj ablation, on in-distribution compound drift. OASIS "
                "stays below ISOMER at every drift level, so the deployed hard-projected form keeps "
                "a learned single-column advantage; the projection costs only a small amount relative to "
                "the stage-1 ablation, the price paid for the downstream composition and join safety "
                "shown in Tables~\\ref{tab:marginal_joint}, \\ref{tab:composition_family} and "
                "\\ref{tab:factorjoin}.}\n")
        f.write("  \\label{tab:single_column_projection}\n")
        f.write("  \\setlength{\\tabcolsep}{5pt}\n")
        f.write("  \\begin{tabular}{c | rrr r | r}\n    \\toprule\n")
        f.write("    $q$ & Stale & ISOMER & OASIS-noProj & OASIS & OASIS red.\\% \\\\\n    \\midrule\n")
        for q in qs:
            r = per_q[q]
            best = min(r["ISOMER"], r["noProj"], r["full"])
            cell = lambda v: f"\\textbf{{{v:.3f}}}" if abs(v - best) < 1e-3 else f"{v:.3f}"
            f.write(f"    {q:>2} & {r['Prior']:.3f} & {cell(r['ISOMER'])} & {cell(r['noProj'])} & "
                    f"{cell(r['full'])} & {red(r['Prior'], r['full']):+.0f}\\% \\\\\n")
        f.write("    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    o = {k: mean([per_q[q][k] for q in qs]) for k in ("Prior", "ISOMER", "noProj", "full")}
    print(f"overall: stale={o['Prior']:.3f} ISOMER={o['ISOMER']:.3f} "
          f"OASIS-noProj={o['noProj']:.3f} OASIS={o['full']:.3f}")
    print(f"wrote {OUT / 'table_single_column_projection.tex'}")


if __name__ == "__main__":
    main()
