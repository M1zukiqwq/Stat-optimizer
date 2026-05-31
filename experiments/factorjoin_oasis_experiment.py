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
varies across {stale, isomer, oasis, oasis_projected, oasis_soft_projection,
hybrid, aggressive_hybrid, fresh} is the
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
from typing import Dict, List, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_PIPELINE_DIR = _REPO_DIR / "cdf_kll_ml_pipeline"
for _p in (_PIPELINE_DIR, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from modern_baselines import correct_isomer, correct_soft_isomer
from mlp_histogram_model_v2 import MlpHistogramModelV2

from copula_oasis_experiment import (
    GaussianCopula,
    generate_correlated_columns,
    get_histogram_boundaries,
    correct_marginal_with_oasis,
    choose_hybrid_marginal,
    feedback_residual_score,
    qerr,
    geomean,
    pct_improvement,
)
from histogram_math import evaluate_piecewise_cdf


METHOD_ORDER = [
    "stale",
    "isomer",
    "oasis",
    "oasis_projected",
    "oasis_soft_projection",
    "hybrid",
    "aggressive_hybrid",
    "fresh",
]


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


def _project(prior_bounds: Sequence[float], observations: Sequence[dict], num_buckets: int,
             max_iter: int, tol: float) -> List[float]:
    try:
        q = correct_isomer(
            prior_bounds[0], prior_bounds[-1], list(prior_bounds[1:-1]),
            list(observations), num_buckets=num_buckets, max_iter=max_iter, tol=tol,
        )
        return [prior_bounds[0]] + list(q) + [prior_bounds[-1]]
    except Exception:
        return list(prior_bounds)


def _soft_project(
    prior_bounds: Sequence[float],
    observations: Sequence[dict],
    num_buckets: int,
    strength: float,
    recency_decay: float,
    target_blend: float,
    observation_window: int,
    max_iter: int,
    lr: float,
    tol: float,
    active_set: bool,
    conflict_aware: bool = False,
    conflict_ref_window: int = 8,
    conflict_tau: float = 0.05,
    conflict_floor: float = 0.0,
) -> List[float]:
    try:
        soft_observations = (
            list(observations[-observation_window:])
            if 0 < observation_window < len(observations)
            else list(observations)
        )
        q = correct_soft_isomer(
            prior_bounds[0], prior_bounds[-1], list(prior_bounds[1:-1]),
            soft_observations, num_buckets=num_buckets,
            constraint_strength=strength, recency_decay=recency_decay,
            target_blend=target_blend, max_iter=max_iter,
            learning_rate=lr, tol=tol, active_set=active_set,
            conflict_aware=conflict_aware, conflict_ref_window=conflict_ref_window,
            conflict_tau=conflict_tau, conflict_floor=conflict_floor,
        )
        return [prior_bounds[0]] + list(q) + [prior_bounds[-1]]
    except Exception:
        return list(prior_bounds)


def _damp_observations(
    copula: GaussianCopula,
    observations: Sequence[dict],
    anchor_bounds: Sequence[float],
    alpha: float,
) -> List[dict]:
    damped: List[dict] = []
    for obs in observations:
        anchored = max(copula.marginal_cdf(anchor_bounds, float(obs["value"])), 1e-9)
        if obs["predicate_type"] in {">", ">="}:
            anchored = max(1.0 - copula.marginal_cdf(anchor_bounds, float(obs["value"])), 1e-9)
        target = alpha * float(obs["actual_sel"]) + (1.0 - alpha) * anchored
        item = dict(obs)
        item["actual_sel"] = max(1e-6, min(1.0 - 1e-6, target))
        damped.append(item)
    return damped


def choose_aggressive_marginal(
    copula: GaussianCopula,
    stale: Sequence[float],
    isomer: Sequence[float],
    oasis: Sequence[float],
    projected: Sequence[float],
    hybrid: Sequence[float],
    observations: Sequence[dict],
    num_buckets: int,
    damping_grid: Sequence[float],
    recent_windows: Sequence[int],
    projection_iters: int,
    projection_tol: float,
) -> Tuple[List[float], str, Dict[str, float]]:
    candidates: Dict[str, List[float]] = {
        "stale": list(stale),
        "isomer": list(isomer),
        "oasis": list(oasis),
        "oasis_projected": list(projected),
        "hybrid": list(hybrid),
    }
    for alpha in damping_grid:
        damped_obs = _damp_observations(copula, observations, oasis, alpha)
        candidates[f"damped_a{int(round(alpha * 100)):02d}"] = _project(
            oasis, damped_obs, num_buckets, projection_iters, projection_tol,
        )
    for window in recent_windows:
        if 0 < window < len(observations):
            recent = list(observations[-window:])
            candidates[f"oasis_recent_k{window}"] = _project(
                oasis, recent, num_buckets, projection_iters, projection_tol,
            )
            candidates[f"isomer_recent_k{window}"] = _project(
                stale, recent, num_buckets, projection_iters, projection_tol,
            )

    scores = {
        name: feedback_residual_score(copula, bounds, observations)
        for name, bounds in candidates.items()
    }
    choice = min(scores, key=lambda name: scores[name])
    return candidates[choice], choice, scores


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
    soft_projection_strength: float,
    soft_projection_recency_decay: float,
    soft_projection_target_blend: float,
    soft_projection_window: int,
    soft_projection_iters: int,
    soft_projection_lr: float,
    soft_projection_tol: float,
    soft_projection_active_set: bool,
    damping_grid: Sequence[float],
    recent_windows: Sequence[int],
    projection_iters: int,
    projection_tol: float,
    soft_projection_conflict_aware: bool = False,
    soft_projection_conflict_ref_window: int = 8,
    soft_projection_conflict_tau: float = 0.05,
    soft_projection_conflict_floor: float = 0.0,
) -> Tuple[Dict[str, List[float]], Dict[str, str]]:
    oasis = correct_marginal_with_oasis(stale_bounds, observations, model, num_buckets, max_obs)
    isomer = _project(stale_bounds, observations, num_buckets, projection_iters, projection_tol)
    projected = _project(oasis, observations, num_buckets, projection_iters, projection_tol)
    soft_projected = _soft_project(
        oasis, observations, num_buckets,
        soft_projection_strength, soft_projection_recency_decay,
        soft_projection_target_blend, soft_projection_window, soft_projection_iters,
        soft_projection_lr, soft_projection_tol, soft_projection_active_set,
        conflict_aware=soft_projection_conflict_aware,
        conflict_ref_window=soft_projection_conflict_ref_window,
        conflict_tau=soft_projection_conflict_tau,
        conflict_floor=soft_projection_conflict_floor,
    )
    hybrid, _, _ = choose_hybrid_marginal(
        copula=copula, stale_bounds=stale_bounds, isomer_bounds=isomer,
        oasis_bounds=oasis, oasis_projected_bounds=projected,
        observations=observations, min_improvement=hybrid_min_improvement)
    aggressive, aggressive_choice, _ = choose_aggressive_marginal(
        copula=copula, stale=stale_bounds, isomer=isomer, oasis=oasis,
        projected=projected, hybrid=hybrid, observations=observations,
        num_buckets=num_buckets, damping_grid=damping_grid,
        recent_windows=recent_windows, projection_iters=projection_iters,
        projection_tol=projection_tol,
    )
    methods = {"stale": list(stale_bounds), "isomer": isomer, "oasis": oasis,
               "oasis_projected": projected, "oasis_soft_projection": soft_projected,
               "hybrid": hybrid,
               "aggressive_hybrid": aggressive, "fresh": list(fresh_bounds)}
    return methods, {"aggressive_choice": aggressive_choice}


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

            methodsA, metaA = repair_all_methods(
                copula, staleA, keysA_drift, obsA, model,
                args.num_buckets, args.max_observations,
                freshA, args.hybrid_min_improvement,
                args.soft_projection_strength,
                args.soft_projection_recency_decay,
                args.soft_projection_target_blend,
                args.soft_projection_window,
                args.soft_projection_iters,
                args.soft_projection_lr,
                args.soft_projection_tol,
                args.soft_projection_active_set,
                args.aggressive_damping_grid,
                args.aggressive_recent_windows,
                args.projection_iters, args.projection_tol,
                soft_projection_conflict_aware=args.soft_projection_conflict_aware,
                soft_projection_conflict_ref_window=args.soft_projection_conflict_ref_window,
                soft_projection_conflict_tau=args.soft_projection_conflict_tau,
                soft_projection_conflict_floor=args.soft_projection_conflict_floor,
            )
            methodsB, metaB = repair_all_methods(
                copula, staleB, keysB_drift, obsB, model,
                args.num_buckets, args.max_observations,
                freshB, args.hybrid_min_improvement,
                args.soft_projection_strength,
                args.soft_projection_recency_decay,
                args.soft_projection_target_blend,
                args.soft_projection_window,
                args.soft_projection_iters,
                args.soft_projection_lr,
                args.soft_projection_tol,
                args.soft_projection_active_set,
                args.aggressive_damping_grid,
                args.aggressive_recent_windows,
                args.projection_iters, args.projection_tol,
                soft_projection_conflict_aware=args.soft_projection_conflict_aware,
                soft_projection_conflict_ref_window=args.soft_projection_conflict_ref_window,
                soft_projection_conflict_tau=args.soft_projection_conflict_tau,
                soft_projection_conflict_floor=args.soft_projection_conflict_floor,
            )

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
                residual = 0.5 * (
                    feedback_residual_score(copula, methodsA[method], obsA)
                    + feedback_residual_score(copula, methodsB[method], obsB)
                )
                results.append({
                    "drift_q": q, "trial": trial, "method": method,
                    "true_join": true_join, "est_join": est,
                    "join_qerr": qerr(est, true_join),
                    "feedback_residual": residual,
                    "aggressive_choice_a": metaA["aggressive_choice"],
                    "aggressive_choice_b": metaB["aggressive_choice"],
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

    print("\n" + "=" * 120)
    print(f"{'Drift q':>8} | {'Stale':>8} | {'ISOMER':>8} | {'OASIS':>8} | "
          f"{'OASIS-Proj':>10} | {'Soft':>8} | {'Hybrid':>8} | {'Aggressive':>10} | {'Fresh':>8} | "
          f"{'Proj%':>7} | {'Soft%':>7} | {'Aggr%':>7}")
    print("=" * 120)
    summary_rows = []
    for q in sorted(by_q.keys()):
        gm = {m: geomean(by_q[q][m]) for m in METHOD_ORDER}
        stale_by_trial = {
            r["trial"]: r["join_qerr"]
            for r in results
            if r["drift_q"] == q and r["method"] == "stale"
        }
        row = {"drift_q": q, **{f"{m}_qerr_gm": gm[m] for m in METHOD_ORDER},
               "oasis_improvement_pct": pct_improvement(gm["stale"], gm["oasis"]),
               "isomer_improvement_pct": pct_improvement(gm["stale"], gm["isomer"]),
               "oasis_projected_improvement_pct": pct_improvement(gm["stale"], gm["oasis_projected"]),
               "oasis_soft_projection_improvement_pct": pct_improvement(gm["stale"], gm["oasis_soft_projection"]),
               "hybrid_improvement_pct": pct_improvement(gm["stale"], gm["hybrid"]),
               "aggressive_hybrid_improvement_pct": pct_improvement(gm["stale"], gm["aggressive_hybrid"])}
        for method in METHOD_ORDER:
            method_rows = [r for r in results if r["drift_q"] == q and r["method"] == method]
            row[f"{method}_feedback_residual_mean"] = (
                sum(r["feedback_residual"] for r in method_rows) / max(len(method_rows), 1)
            )
            row[f"{method}_worse_than_stale_frac"] = (
                sum(r["join_qerr"] > stale_by_trial.get(r["trial"], float("inf")) for r in method_rows)
                / max(len(method_rows), 1)
            )
        summary_rows.append(row)
        print(f"{q:>8} | {gm['stale']:8.3f} | {gm['isomer']:8.3f} | {gm['oasis']:8.3f} | "
              f"{gm['oasis_projected']:10.3f} | {gm['oasis_soft_projection']:8.3f} | {gm['hybrid']:8.3f} | "
              f"{gm['aggressive_hybrid']:10.3f} | {gm['fresh']:8.3f} | "
              f"{row['oasis_projected_improvement_pct']:+6.1f}% | "
              f"{row['oasis_soft_projection_improvement_pct']:+6.1f}% | "
              f"{row['aggressive_hybrid_improvement_pct']:+6.1f}%")
    print("-" * 120)
    print(f"{'ALL':>8} | {overall['stale']:8.3f} | {overall['isomer']:8.3f} | {overall['oasis']:8.3f} | "
          f"{overall['oasis_projected']:10.3f} | {overall['oasis_soft_projection']:8.3f} | {overall['hybrid']:8.3f} | "
          f"{overall['aggressive_hybrid']:10.3f} | {overall['fresh']:8.3f} | "
          f"{pct_improvement(overall['stale'], overall['oasis_projected']):+6.1f}% | "
          f"{pct_improvement(overall['stale'], overall['oasis_soft_projection']):+6.1f}% | "
          f"{pct_improvement(overall['stale'], overall['aggressive_hybrid']):+6.1f}%")

    summary_rows.append({"drift_q": "all", **{f"{m}_qerr_gm": overall[m] for m in METHOD_ORDER},
                         "oasis_improvement_pct": pct_improvement(overall["stale"], overall["oasis"]),
                         "isomer_improvement_pct": pct_improvement(overall["stale"], overall["isomer"]),
                         "oasis_projected_improvement_pct": pct_improvement(overall["stale"], overall["oasis_projected"]),
                         "oasis_soft_projection_improvement_pct": pct_improvement(overall["stale"], overall["oasis_soft_projection"]),
                         "hybrid_improvement_pct": pct_improvement(overall["stale"], overall["hybrid"]),
                         "aggressive_hybrid_improvement_pct": pct_improvement(overall["stale"], overall["aggressive_hybrid"])})
    for method in METHOD_ORDER:
        method_rows = [r for r in results if r["method"] == method]
        stale_by_key = {
            (r["drift_q"], r["trial"]): r["join_qerr"]
            for r in results
            if r["method"] == "stale"
        }
        summary_rows[-1][f"{method}_feedback_residual_mean"] = (
            sum(r["feedback_residual"] for r in method_rows) / max(len(method_rows), 1)
        )
        summary_rows[-1][f"{method}_worse_than_stale_frac"] = (
            sum(r["join_qerr"] > stale_by_key.get((r["drift_q"], r["trial"]), float("inf")) for r in method_rows)
            / max(len(method_rows), 1)
        )

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
        f.write("  \\adjustbox{max width=\\columnwidth}{%\n")
        f.write("  \\begin{tabular}{l | rrrrrrrr | rrr}\n    \\toprule\n")
        f.write("    $q$ & Stale & OASIS-noProj & ISOMER & OASIS & Soft & Hybrid & Aggr. & Fresh & OASIS +\\% & Soft +\\% & Aggr. +\\% \\\\\n")
        f.write("    \\midrule\n")
        for r in summary_rows:
            q = r["drift_q"]
            label = "\\textbf{All}" if q == "all" else f"{q}"
            best = min(r["oasis_qerr_gm"], r["isomer_qerr_gm"],
                       r["oasis_projected_qerr_gm"], r["hybrid_qerr_gm"],
                       r["oasis_soft_projection_qerr_gm"],
                       r["aggressive_hybrid_qerr_gm"])
            def cell(v):
                return f"\\textbf{{{v:.3f}}}" if abs(v - best) < 1e-3 else f"{v:.3f}"
            if q == "all":
                f.write("    \\midrule\n")
            f.write(f"    {label} & {r['stale_qerr_gm']:.3f} & {cell(r['oasis_qerr_gm'])} & "
                    f"{cell(r['isomer_qerr_gm'])} & {cell(r['oasis_projected_qerr_gm'])} & "
                    f"{cell(r['oasis_soft_projection_qerr_gm'])} & "
                    f"{cell(r['hybrid_qerr_gm'])} & {cell(r['aggressive_hybrid_qerr_gm'])} & "
                    f"{r['fresh_qerr_gm']:.3f} & "
                    f"{r['oasis_projected_improvement_pct']:+.1f}\\% & "
                    f"{r['oasis_soft_projection_improvement_pct']:+.1f}\\% & "
                    f"{r['aggressive_hybrid_improvement_pct']:+.1f}\\% \\\\\n")
        f.write("    \\bottomrule\n  \\end{tabular}\n  }\n\\end{table}\n")
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
    p.add_argument("--soft-projection-strength", type=float, default=30.0)
    p.add_argument("--soft-projection-recency-decay", type=float, default=0.80)
    p.add_argument("--soft-projection-target-blend", type=float, default=1.0)
    p.add_argument("--soft-projection-window", type=int, default=0,
                   help="Use only the most recent N observations for soft projection; 0 uses the full window.")
    p.add_argument("--soft-projection-iters", type=int, default=500)
    p.add_argument("--soft-projection-lr", type=float, default=0.05)
    p.add_argument("--soft-projection-tol", type=float, default=1e-9)
    p.add_argument("--soft-projection-active-set", action="store_true",
                   help="Apply soft projection only to the latest hard-feasible feedback suffix.")
    p.add_argument("--soft-projection-conflict-aware", action="store_true",
                   help="Down-weight feedback constraints contradicted by the most recent observations.")
    p.add_argument("--soft-projection-conflict-ref-window", type=int, default=8,
                   help="Number of most-recent observations treated as the trusted conflict reference.")
    p.add_argument("--soft-projection-conflict-tau", type=float, default=0.05,
                   help="Conflict bandwidth; smaller tau suppresses contradicted observations more aggressively.")
    p.add_argument("--soft-projection-conflict-floor", type=float, default=0.0,
                   help="Minimum residual weight for a contradicted observation (0 fully removes it).")
    p.add_argument("--aggressive-damping-grid", type=float, nargs="+",
                   default=[0.35, 0.50, 0.65, 0.80, 0.95])
    p.add_argument("--aggressive-recent-windows", type=int, nargs="+",
                   default=[4, 8, 12])
    p.add_argument("--projection-iters", type=int, default=200)
    p.add_argument("--projection-tol", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
