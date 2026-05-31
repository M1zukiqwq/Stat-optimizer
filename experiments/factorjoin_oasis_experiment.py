#!/usr/bin/env python3
"""
FactorJoin + OASIS Integration Experiment.

Goal
----
Embed OASIS into the FactorJoin (Wu et al., SIGMOD 2023) join-cardinality
estimation kernel and show that correcting the single-column statistics it
consumes propagates into better *join* estimates -- i.e. OASIS is a drop-in
upgrade for a real learned-CE join estimator, not only for single-table
composition.

What is reimplemented
---------------------
FactorJoin factorizes a join query into per-table single-column distributions
and a binned join-key factor: for an equi-join A.k = B.k it bins the join-key
domain into M bins and estimates

    |A JOIN B| = sum_bin  cntA[bin] * cntB[bin] / dv[bin]

where cntT[bin] is the number of rows of table T whose join key falls in the
bin and dv[bin] is the number of distinct key values the bin covers (uniform
spread within a bin). This is FactorJoin's bin-based join kernel. We reimplement
exactly this kernel; we do not pull in their full multi-join factor graph.

The OASIS integration point
---------------------------
cntT[bin] is read from a single-column histogram of table T's join-key
distribution: cntT[bin] = N_T * (F_hist(bin_hi) - F_hist(bin_lo)). When the key
distribution drifts, a stale histogram yields wrong per-bin frequencies and a
wrong join estimate. OASIS repairs that single-column histogram from query
feedback before FactorJoin's kernel consumes it.

Honesty boundary
----------------
The two tables' join keys are generated independently; the only quantity that
varies across {stale, isomer, oasis, oasis_projected, hybrid, fresh} is the
quality of each table's single-column join-key histogram. The binning, the
distinct-value-per-bin assumption, and the row counts N_T are identical for
every method. No query runtime is measured; the metric is join-cardinality
Q-error against the exact join size on the drifted data.

Reuses the data generator and OASIS/ISOMER correction from
``copula_oasis_experiment`` so the marginal-repair path is byte-for-byte
identical to the single-column and composition experiments.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_PIPELINE_DIR = _REPO_DIR / "cdf_kll_ml_pipeline"
for _p in (_PIPELINE_DIR, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from modern_baselines import correct_isomer
from mlp_histogram_model_v2 import MlpHistogramModelV2

from copula_oasis_experiment import (
    GaussianCopula,
    generate_correlated_columns,
    get_histogram_boundaries,
    correct_marginal_with_oasis,
    choose_hybrid_marginal,
    qerr,
    geomean,
    pct_improvement,
)
from histogram_math import evaluate_piecewise_cdf


METHOD_ORDER = ["stale", "isomer", "oasis", "oasis_projected", "hybrid", "fresh"]


# ─── FactorJoin bin-based join kernel ────────────────────────────────────────

def discretize(values: Sequence[float], domain: int) -> np.ndarray:
    """Map normalized [0,1] key values to integer keys in {0, ..., domain-1}."""
    arr = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return np.clip(np.round(arr * (domain - 1)).astype(int), 0, domain - 1)


def exact_join_size(keys_a: np.ndarray, keys_b: np.ndarray, domain: int) -> float:
    """Ground-truth equi-join size |A JOIN B| on A.k = B.k by value counts."""
    cnt_a = np.bincount(keys_a, minlength=domain).astype(float)
    cnt_b = np.bincount(keys_b, minlength=domain).astype(float)
    return float(cnt_a @ cnt_b)


def factorjoin_estimate(
    copula: GaussianCopula,
    bounds_a: Sequence[float],
    bounds_b: Sequence[float],
    n_a: int,
    n_b: int,
    edges: np.ndarray,
    domain: int,
) -> float:
    """FactorJoin bin-based join estimate from two single-column key histograms.

    cntT[bin] = N_T * (F_T(bin_hi) - F_T(bin_lo)); dv = domain / num_bins;
    |A JOIN B| ~= sum_bin cntA[bin] * cntB[bin] / dv.
    """
    num_bins = len(edges) - 1
    dv = max(domain / num_bins, 1.0)
    cdf_a = np.array([copula.marginal_cdf(bounds_a, e) for e in edges])
    cdf_b = np.array([copula.marginal_cdf(bounds_b, e) for e in edges])
    mass_a = np.clip(np.diff(cdf_a), 0.0, None)
    mass_b = np.clip(np.diff(cdf_b), 0.0, None)
    cnt_a = n_a * mass_a
    cnt_b = n_b * mass_b
    est = float(np.sum(cnt_a * cnt_b) / dv)
    return max(est, 1e-6)


# ─── Per-table marginal repair (identical recipe to composition exp) ─────────

def build_feedback(
    drifted: Sequence[float],
    stale_bounds: Sequence[float],
    num_buckets: int,
    n_obs: int,
    seed: int,
) -> List[dict]:
    obs_list = []
    sorted_d = sorted(drifted)
    n = len(sorted_d)
    rng = random.Random(seed)
    cdf_p = [i / num_buckets for i in range(num_buckets + 1)]
    for _ in range(n_obs):
        v = sorted_d[rng.randint(0, n - 1)]
        ptype = rng.choice(["<", "<=", ">=", ">"])
        actual_cdf = sum(1 for x in drifted if x <= v) / max(n, 1)
        est_cdf = evaluate_piecewise_cdf(stale_bounds, cdf_p, v)
        if ptype in {"<", "<="}:
            act, est = actual_cdf, est_cdf
        else:
            act, est = 1.0 - actual_cdf, 1.0 - est_cdf
        obs_list.append({"predicate_type": ptype, "value": v,
                         "estimated_sel": est, "actual_sel": act})
    return obs_list


def repair_all_methods(
    copula: GaussianCopula,
    stale_bounds: Sequence[float],
    drifted: Sequence[float],
    observations: List[dict],
    model: MlpHistogramModelV2,
    num_buckets: int,
    max_obs: int,
    fresh_bounds: Sequence[float],
    hybrid_min_improvement: float,
) -> Dict[str, List[float]]:
    oasis = correct_marginal_with_oasis(stale_bounds, observations, model, num_buckets, max_obs)
    try:
        iq = correct_isomer(stale_bounds[0], stale_bounds[-1], list(stale_bounds[1:-1]),
                            observations, num_buckets=num_buckets)
        isomer = [stale_bounds[0]] + list(iq) + [stale_bounds[-1]]
    except Exception:
        isomer = list(stale_bounds)
    try:
        pq = correct_isomer(oasis[0], oasis[-1], list(oasis[1:-1]),
                            observations, num_buckets=num_buckets)
        projected = [oasis[0]] + list(pq) + [oasis[-1]]
    except Exception:
        projected = list(oasis)
    hybrid, _, _ = choose_hybrid_marginal(
        copula=copula, stale_bounds=stale_bounds, isomer_bounds=isomer,
        oasis_bounds=oasis, oasis_projected_bounds=projected,
        observations=observations, min_improvement=hybrid_min_improvement)
    return {"stale": list(stale_bounds), "isomer": isomer, "oasis": oasis,
            "oasis_projected": projected, "hybrid": hybrid, "fresh": list(fresh_bounds)}


# ─── Main experiment ─────────────────────────────────────────────────────────

def run_experiment(args):
    copula = GaussianCopula(num_buckets=args.num_buckets)
    model = MlpHistogramModelV2.load(str(args.model_path))
    edges = np.linspace(0.0, 1.0, args.join_bins + 1)
    results: List[dict] = []

    for q in args.drift_levels:
        for trial in range(args.n_trials):
            seed = args.seed + trial * 1000 + q
            seed_a, seed_b = seed * 2 + 1, seed * 2 + 7
            print(f"q={q} trial={trial}...", end=" ", flush=True)

            iniA, dftA = generate_correlated_columns(args.n_rows, 1, 0.0, seed_a, q)
            iniB, dftB = generate_correlated_columns(args.n_rows, 1, 0.0, seed_b, q)
            keysA_init, keysA_drift = iniA[0], dftA[0]
            keysB_init, keysB_drift = iniB[0], dftB[0]

            staleA = get_histogram_boundaries(keysA_init, args.num_buckets)
            staleB = get_histogram_boundaries(keysB_init, args.num_buckets)
            freshA = get_histogram_boundaries(keysA_drift, args.num_buckets)
            freshB = get_histogram_boundaries(keysB_drift, args.num_buckets)

            obsA = build_feedback(keysA_drift, staleA, args.num_buckets, args.n_observations, seed_a + 3)
            obsB = build_feedback(keysB_drift, staleB, args.num_buckets, args.n_observations, seed_b + 3)

            methodsA = repair_all_methods(copula, staleA, keysA_drift, obsA, model,
                                          args.num_buckets, args.max_observations,
                                          freshA, args.hybrid_min_improvement)
            methodsB = repair_all_methods(copula, staleB, keysB_drift, obsB, model,
                                          args.num_buckets, args.max_observations,
                                          freshB, args.hybrid_min_improvement)

            dkeysA = discretize(keysA_drift, args.domain)
            dkeysB = discretize(keysB_drift, args.domain)
            true_join = exact_join_size(dkeysA, dkeysB, args.domain)
            if true_join < 1.0:
                print("skip(empty join)")
                continue

            n_a = len(keysA_drift)
            n_b = len(keysB_drift)
            for method in METHOD_ORDER:
                est = factorjoin_estimate(copula, methodsA[method], methodsB[method],
                                          n_a, n_b, edges, args.domain)
                results.append({
                    "drift_q": q, "trial": trial, "method": method,
                    "true_join": true_join, "est_join": est,
                    "join_qerr": qerr(est, true_join),
                })
            print("done")

    _summarize_and_save(results, args)
    return results


def _summarize_and_save(results: List[dict], args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Overall (across all drift) + per-drift breakdown.
    overall = {m: geomean([r["join_qerr"] for r in results if r["method"] == m]) for m in METHOD_ORDER}
    by_q = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_q[r["drift_q"]][r["method"]].append(r["join_qerr"])

    print("\n" + "=" * 104)
    print(f"{'Drift q':>8} | {'Stale':>8} | {'ISOMER':>8} | {'OASIS':>8} | "
          f"{'OASIS-Proj':>10} | {'Hybrid':>8} | {'Fresh':>8} | {'OASIS%':>7} | {'Proj%':>7} | {'Hyb%':>7}")
    print("=" * 104)
    summary_rows = []
    for q in sorted(by_q.keys()):
        gm = {m: geomean(by_q[q][m]) for m in METHOD_ORDER}
        row = {"drift_q": q, **{f"{m}_qerr_gm": gm[m] for m in METHOD_ORDER},
               "oasis_improvement_pct": pct_improvement(gm["stale"], gm["oasis"]),
               "isomer_improvement_pct": pct_improvement(gm["stale"], gm["isomer"]),
               "oasis_projected_improvement_pct": pct_improvement(gm["stale"], gm["oasis_projected"]),
               "hybrid_improvement_pct": pct_improvement(gm["stale"], gm["hybrid"])}
        summary_rows.append(row)
        print(f"{q:>8} | {gm['stale']:8.3f} | {gm['isomer']:8.3f} | {gm['oasis']:8.3f} | "
              f"{gm['oasis_projected']:10.3f} | {gm['hybrid']:8.3f} | {gm['fresh']:8.3f} | "
              f"{row['oasis_improvement_pct']:+6.1f}% | {row['oasis_projected_improvement_pct']:+6.1f}% | "
              f"{row['hybrid_improvement_pct']:+6.1f}%")
    print("-" * 104)
    print(f"{'ALL':>8} | {overall['stale']:8.3f} | {overall['isomer']:8.3f} | {overall['oasis']:8.3f} | "
          f"{overall['oasis_projected']:10.3f} | {overall['hybrid']:8.3f} | {overall['fresh']:8.3f} | "
          f"{pct_improvement(overall['stale'], overall['oasis']):+6.1f}% | "
          f"{pct_improvement(overall['stale'], overall['oasis_projected']):+6.1f}% | "
          f"{pct_improvement(overall['stale'], overall['hybrid']):+6.1f}%")

    summary_rows.append({"drift_q": "all", **{f"{m}_qerr_gm": overall[m] for m in METHOD_ORDER},
                         "oasis_improvement_pct": pct_improvement(overall["stale"], overall["oasis"]),
                         "isomer_improvement_pct": pct_improvement(overall["stale"], overall["isomer"]),
                         "oasis_projected_improvement_pct": pct_improvement(overall["stale"], overall["oasis_projected"]),
                         "hybrid_improvement_pct": pct_improvement(overall["stale"], overall["hybrid"])})

    with open(output_dir / "factorjoin_results.json", "w") as f:
        json.dump(results, f)
    with open(output_dir / "factorjoin_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    with open(output_dir / "factorjoin_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    _generate_latex_table(summary_rows, output_dir)
    print(f"\nSaved to {output_dir}")


def _generate_latex_table(summary_rows, output_dir):
    path = output_dir / "table_factorjoin.tex"
    with open(path, "w") as f:
        f.write("\\begin{table}[t]\n  \\centering\n  \\small\n")
        f.write("  \\caption{OASIS embedded in the FactorJoin bin-based join kernel. "
                "Join-cardinality Q-error ($\\downarrow$) for a two-table equi-join whose join-key "
                "distributions drift; only the single-column key histogram consumed by FactorJoin "
                "varies across methods. The full two-stage OASIS and Hybrid recover most of the "
                "stale-to-fresh join error, whereas OASIS-noProj (the learned stage without "
                "projection) is actively harmful---it exceeds the stale baseline in this bilinear "
                "kernel.}\n")
        f.write("  \\label{tab:factorjoin}\n")
        f.write("  \\setlength{\\tabcolsep}{4pt}\n")
        f.write("  \\begin{tabular}{l | rrrrrr | rr}\n    \\toprule\n")
        f.write("    $q$ & Stale & OASIS-noProj & ISOMER & OASIS & Hybrid & Fresh & OASIS +\\% & Hybrid +\\% \\\\\n")
        f.write("    \\midrule\n")
        for r in summary_rows:
            q = r["drift_q"]
            label = "\\textbf{All}" if q == "all" else f"{q}"
            best = min(r["oasis_qerr_gm"], r["isomer_qerr_gm"],
                       r["oasis_projected_qerr_gm"], r["hybrid_qerr_gm"])
            def cell(v):
                return f"\\textbf{{{v:.3f}}}" if abs(v - best) < 1e-3 else f"{v:.3f}"
            if q == "all":
                f.write("    \\midrule\n")
            f.write(f"    {label} & {r['stale_qerr_gm']:.3f} & {cell(r['oasis_qerr_gm'])} & "
                    f"{cell(r['isomer_qerr_gm'])} & {cell(r['oasis_projected_qerr_gm'])} & "
                    f"{cell(r['hybrid_qerr_gm'])} & {r['fresh_qerr_gm']:.3f} & "
                    f"{r['oasis_projected_improvement_pct']:+.1f}\\% & "
                    f"{r['hybrid_improvement_pct']:+.1f}\\% \\\\\n")
        f.write("    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    print(f"LaTeX table saved to {path}")


def main():
    p = argparse.ArgumentParser(description="FactorJoin + OASIS integration experiment")
    p.add_argument("--model-path", type=Path,
                   default=_REPO_DIR / "experiments" / "results" / "copula_model" / "oasis_k16.json")
    p.add_argument("--output-dir", type=Path,
                   default=_REPO_DIR / "experiments" / "results" / "factorjoin_oasis")
    p.add_argument("--num-buckets", type=int, default=10)
    p.add_argument("--max-observations", type=int, default=16)
    p.add_argument("--n-rows", type=int, default=5000)
    p.add_argument("--n-observations", type=int, default=16)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--drift-levels", type=int, nargs="+", default=[5, 10, 20, 30])
    p.add_argument("--domain", type=int, default=1000, help="Distinct join-key values.")
    p.add_argument("--join-bins", type=int, default=50, help="FactorJoin key bins (M).")
    p.add_argument("--hybrid-min-improvement", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
