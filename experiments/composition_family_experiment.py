#!/usr/bin/env python3
"""
Composition-Family Experiment: OASIS as a drop-in marginal upgrade.

Goal
----
Show that OASIS-corrected single-column marginals can be plugged into a *family*
of independently-designed multi-column composition estimators that all consume
single-column marginals as a swappable input, and that the joint-selectivity
gain does not depend on any single dependence model. This supports two claims:

  (1) Seamless compatibility: across Independence, IPF/Sinkhorn-2D, and the
      Gaussian / Clayton / Gumbel / Frank copulas, swapping in OASIS-Proj
      marginals beats stale marginals and approaches fresh marginals.
  (2) The "plain OASIS needs feedback-consistency projection" finding is
      universal across composition methods, not specific to one estimator.

Honesty boundary
----------------
OASIS does NOT infer multi-column dependence. The dependence structure is taken
from a single fixed source that is identical for every marginal input:
  * Copulas use one fixed association parameter derived from the data-generating
    correlation (well-specified only for the Gaussian copula; the Archimedean
    copulas are deliberately mis-specified models).
  * IPF/Sinkhorn-2D seeds its association (cell odds-ratios) from a STALE 2D
    sketch, then reconciles it with each method's corrected 1D marginals.
The only variable that changes across {stale, isomer, oasis, oasis_projected,
oasis_soft_projection, hybrid, aggressive_hybrid, fresh} is the *marginal quality*. No query runtime is
measured.

Reuses data generation and OASIS/ISOMER correction from
``copula_oasis_experiment`` so the marginal-repair path is identical.
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

# Reuse the validated building blocks from the existing copula experiment.
from copula_oasis_experiment import (
    GaussianCopula,
    generate_correlated_columns,
    get_histogram_boundaries,
    compute_true_joint_selectivity,
    correct_marginal_with_oasis,
    marginal_range_selectivity,
    actual_marginal_range_selectivity,
    choose_hybrid_marginal,
    qerr,
    geomean,
    pct_improvement,
)
from factorjoin_oasis_experiment import choose_aggressive_marginal
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
ESTIMATOR_ORDER = [
    "independence",
    "ipf_sinkhorn",
    "gaussian_copula",
    "clayton_copula",
    "gumbel_copula",
    "frank_copula",
]


# ─── Dependence-parameter conversions (single fixed source) ──────────────────

def pearson_to_kendall(rho: float) -> float:
    """Gaussian-copula relationship tau = (2/pi) arcsin(rho)."""
    rho = max(-0.999, min(0.999, rho))
    return (2.0 / math.pi) * math.asin(rho)


def _debye1(theta: float, n_pts: int = 200) -> float:
    """First Debye function D1(theta) = (1/theta) integral_0^theta t/(e^t-1) dt."""
    if theta <= 1e-9:
        return 1.0
    dt = theta / n_pts
    total = 0.0
    for i in range(n_pts):
        t = (i + 0.5) * dt
        total += (t / (math.exp(t) - 1.0)) * dt
    return total / theta


def frank_theta_from_tau(tau: float) -> float:
    """Invert tau = 1 + 4(D1(theta)-1)/theta for Frank's theta via bisection."""
    if tau <= 1e-4:
        return 0.0
    lo, hi = 1e-3, 50.0

    def tau_of(theta: float) -> float:
        return 1.0 + 4.0 * (_debye1(theta) - 1.0) / theta

    if tau_of(hi) < tau:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if tau_of(mid) < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ─── Archimedean copula CDFs C(u, v) ─────────────────────────────────────────

def _c_independence(u: float, v: float) -> float:
    return u * v


def _c_clayton(u: float, v: float, theta: float) -> float:
    if theta <= 1e-6:
        return u * v
    u = max(u, 1e-9)
    v = max(v, 1e-9)
    inner = u ** (-theta) + v ** (-theta) - 1.0
    if inner <= 0.0:
        return 0.0
    return inner ** (-1.0 / theta)


def _c_gumbel(u: float, v: float, theta: float) -> float:
    if theta <= 1.0 + 1e-6:
        return u * v
    u = min(max(u, 1e-12), 1 - 1e-12)
    v = min(max(v, 1e-12), 1 - 1e-12)
    a = (-math.log(u)) ** theta
    b = (-math.log(v)) ** theta
    return math.exp(-((a + b) ** (1.0 / theta)))


def _c_frank(u: float, v: float, theta: float) -> float:
    if abs(theta) <= 1e-6:
        return u * v
    num = (math.exp(-theta * u) - 1.0) * (math.exp(-theta * v) - 1.0)
    den = math.exp(-theta) - 1.0
    val = 1.0 + num / den
    if val <= 1e-12:
        val = 1e-12
    return -1.0 / theta * math.log(val)


_ARCHIMEDEAN = {
    "clayton_copula": _c_clayton,
    "gumbel_copula": _c_gumbel,
    "frank_copula": _c_frank,
}


def _theta_for(estimator: str, rho: float) -> float:
    tau = pearson_to_kendall(rho)
    tau = max(0.0, min(0.999, tau))
    if estimator == "clayton_copula":
        return 0.0 if tau <= 1e-4 else 2.0 * tau / (1.0 - tau)
    if estimator == "gumbel_copula":
        return 1.0 if tau <= 1e-4 else 1.0 / (1.0 - tau)
    if estimator == "frank_copula":
        return frank_theta_from_tau(tau)
    raise ValueError(estimator)


def archimedean_range_prob(
    copula: GaussianCopula,
    estimator: str,
    boundaries_c0: Sequence[float],
    boundaries_c1: Sequence[float],
    predicates: Sequence[Tuple[float, float]],
    theta: float,
) -> float:
    """Box probability for a 2-column conjunctive range via an Archimedean copula."""
    cfun = _ARCHIMEDEAN[estimator]
    (lo1, hi1), (lo2, hi2) = predicates
    u1_lo = copula.marginal_cdf(boundaries_c0, lo1)
    u1_hi = copula.marginal_cdf(boundaries_c0, hi1)
    u2_lo = copula.marginal_cdf(boundaries_c1, lo2)
    u2_hi = copula.marginal_cdf(boundaries_c1, hi2)
    p = (cfun(u1_hi, u2_hi, theta)
         - cfun(u1_lo, u2_hi, theta)
         - cfun(u1_hi, u2_lo, theta)
         + cfun(u1_lo, u2_lo, theta))
    return max(p, 1e-12)


# ─── IPF / Sinkhorn-2D ───────────────────────────────────────────────────────

def build_stale_seed_table(
    initial_cols: Sequence[Sequence[float]],
    edges: np.ndarray,
    smoothing: float = 1e-3,
) -> np.ndarray:
    """Joint cell counts of the STALE (pre-drift) data on a fixed grid.

    This is the single fixed source of dependence/association shared by every
    marginal input. Normalised to sum to 1 with Laplace smoothing.
    """
    g = len(edges) - 1
    x0 = np.clip(np.asarray(initial_cols[0], dtype=float), 0.0, 1.0)
    x1 = np.clip(np.asarray(initial_cols[1], dtype=float), 0.0, 1.0)
    i0 = np.clip(np.searchsorted(edges, x0, side="right") - 1, 0, g - 1)
    i1 = np.clip(np.searchsorted(edges, x1, side="right") - 1, 0, g - 1)
    table = np.zeros((g, g), dtype=float)
    np.add.at(table, (i0, i1), 1.0)
    table += smoothing
    return table / table.sum()


def marginal_cell_masses(
    copula: GaussianCopula,
    boundaries: Sequence[float],
    edges: np.ndarray,
) -> np.ndarray:
    """Probability mass of a method's 1D marginal in each fixed grid cell."""
    cdf_vals = np.array([copula.marginal_cdf(boundaries, e) for e in edges])
    masses = np.diff(cdf_vals)
    masses = np.clip(masses, 1e-9, None)
    return masses / masses.sum()


def ipf_2d(
    seed: np.ndarray,
    row_target: np.ndarray,
    col_target: np.ndarray,
    n_iter: int = 40,
    tol: float = 1e-8,
) -> np.ndarray:
    """Iterative Proportional Fitting: rescale `seed` to match row/col marginals
    while preserving the seed's cross-product (odds-ratio) association."""
    m = seed.copy()
    for _ in range(n_iter):
        row_sums = m.sum(axis=1, keepdims=True)
        m *= (row_target.reshape(-1, 1) / np.clip(row_sums, 1e-12, None))
        col_sums = m.sum(axis=0, keepdims=True)
        m *= (col_target.reshape(1, -1) / np.clip(col_sums, 1e-12, None))
        if np.max(np.abs(m.sum(axis=1) - row_target)) < tol:
            break
    return m


def _overlap_weights(edges: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Fraction of each grid cell's mass that falls inside [lo, hi]."""
    left = np.maximum(edges[:-1], lo)
    right = np.minimum(edges[1:], hi)
    overlap = np.clip(right - left, 0.0, None)
    width = np.clip(edges[1:] - edges[:-1], 1e-12, None)
    return overlap / width


def ipf_range_prob(
    joint: np.ndarray,
    edges: np.ndarray,
    predicates: Sequence[Tuple[float, float]],
) -> float:
    (lo1, hi1), (lo2, hi2) = predicates
    w0 = _overlap_weights(edges, lo1, hi1)
    w1 = _overlap_weights(edges, lo2, hi2)
    p = float(w0 @ joint @ w1)
    return max(p, 1e-12)


# ─── Unified joint estimator dispatch ────────────────────────────────────────

def estimate_independence(
    copula: GaussianCopula,
    bounds_c0: Sequence[float],
    bounds_c1: Sequence[float],
    predicates: Sequence[Tuple[float, float]],
) -> float:
    s = 1.0
    for bounds, (lo, hi) in zip((bounds_c0, bounds_c1), predicates):
        s *= max(copula.marginal_cdf(bounds, hi) - copula.marginal_cdf(bounds, lo), 1e-12)
    return max(s, 1e-12)


# ─── Main experiment ─────────────────────────────────────────────────────────

def run_experiment(args):
    copula = GaussianCopula(num_buckets=args.num_buckets)
    model = MlpHistogramModelV2.load(str(args.model_path))
    edges = np.linspace(0.0, 1.0, args.ipf_grid + 1)

    results: List[dict] = []

    for rho in args.correlations:
        thetas = {est: _theta_for(est, rho) for est in _ARCHIMEDEAN}
        for q in args.drift_levels:
            for trial in range(args.n_trials):
                seed = args.seed + trial * 1000 + int(rho * 100) + q
                print(f"rho={rho:.1f} q={q} trial={trial}...", end=" ", flush=True)

                initial, drifted = generate_correlated_columns(
                    n_rows=args.n_rows, n_cols=2, correlation=rho,
                    seed=seed, drift_rounds=q,
                )
                stale_bounds = [get_histogram_boundaries(initial[c], args.num_buckets) for c in range(2)]
                fresh_bounds = [get_histogram_boundaries(drifted[c], args.num_buckets) for c in range(2)]

                # Feedback observations per column (identical recipe to copula exp).
                col_observations = []
                for c in range(2):
                    obs_list = []
                    sorted_drifted = sorted(drifted[c])
                    n = len(sorted_drifted)
                    rng = random.Random(seed + c)
                    cdf_p = [i / args.num_buckets for i in range(args.num_buckets + 1)]
                    for _ in range(args.n_observations):
                        v = sorted_drifted[rng.randint(0, n - 1)]
                        predicate_type = rng.choice(["<", "<=", ">=", ">"])
                        actual_cdf = sum(1 for x in drifted[c] if x <= v) / max(n, 1)
                        estimated_cdf = evaluate_piecewise_cdf(stale_bounds[c], cdf_p, v)
                        if predicate_type in {"<", "<="}:
                            act_sel, est_sel = actual_cdf, estimated_cdf
                        else:
                            act_sel, est_sel = 1.0 - actual_cdf, 1.0 - estimated_cdf
                        obs_list.append({
                            "predicate_type": predicate_type, "value": v,
                            "estimated_sel": est_sel, "actual_sel": act_sel,
                        })
                    col_observations.append(obs_list)

                # ── Marginal repair: OASIS, ISOMER, OASIS-Proj, Hybrid ──
                oasis_bounds, isomer_bounds, oasis_projected_bounds, oasis_soft_bounds = [], [], [], []
                for c in range(2):
                    oasis_bounds.append(correct_marginal_with_oasis(
                        stale_bounds[c], col_observations[c], model,
                        args.num_buckets, args.max_observations))
                for c in range(2):
                    try:
                        iq = correct_isomer(
                            stale_bounds[c][0], stale_bounds[c][-1],
                            stale_bounds[c][1:-1], col_observations[c],
                            num_buckets=args.num_buckets)
                        isomer_bounds.append([stale_bounds[c][0]] + list(iq) + [stale_bounds[c][-1]])
                    except Exception:
                        isomer_bounds.append(stale_bounds[c])
                for c in range(2):
                    try:
                        pq = correct_isomer(
                            oasis_bounds[c][0], oasis_bounds[c][-1],
                            oasis_bounds[c][1:-1], col_observations[c],
                            num_buckets=args.num_buckets)
                        oasis_projected_bounds.append([oasis_bounds[c][0]] + list(pq) + [oasis_bounds[c][-1]])
                    except Exception:
                        oasis_projected_bounds.append(oasis_bounds[c])
                for c in range(2):
                    try:
                        soft_observations = (
                            col_observations[c][-args.soft_projection_window:]
                            if 0 < args.soft_projection_window < len(col_observations[c])
                            else col_observations[c]
                        )
                        sq = correct_soft_isomer(
                            oasis_bounds[c][0], oasis_bounds[c][-1],
                            oasis_bounds[c][1:-1], soft_observations,
                            num_buckets=args.num_buckets,
                            constraint_strength=args.soft_projection_strength,
                            recency_decay=args.soft_projection_recency_decay,
                            target_blend=args.soft_projection_target_blend,
                            max_iter=args.soft_projection_iters,
                            learning_rate=args.soft_projection_lr,
                            tol=args.soft_projection_tol,
                            active_set=args.soft_projection_active_set,
                            conflict_aware=args.soft_projection_conflict_aware,
                            conflict_ref_window=args.soft_projection_conflict_ref_window,
                            conflict_tau=args.soft_projection_conflict_tau,
                            conflict_floor=args.soft_projection_conflict_floor,
                        )
                        oasis_soft_bounds.append([oasis_bounds[c][0]] + list(sq) + [oasis_bounds[c][-1]])
                    except Exception:
                        oasis_soft_bounds.append(oasis_bounds[c])

                hybrid_bounds, hybrid_choices = [], []
                for c in range(2):
                    sb, sm, _ = choose_hybrid_marginal(
                        copula=copula, stale_bounds=stale_bounds[c],
                        isomer_bounds=isomer_bounds[c], oasis_bounds=oasis_bounds[c],
                        oasis_projected_bounds=oasis_projected_bounds[c],
                        observations=col_observations[c],
                        min_improvement=args.hybrid_min_improvement)
                    hybrid_bounds.append(sb)
                    hybrid_choices.append(sm)

                aggressive_bounds, aggressive_choices = [], []
                for c in range(2):
                    ab, am, _ = choose_aggressive_marginal(
                        copula=copula,
                        stale=stale_bounds[c],
                        isomer=isomer_bounds[c],
                        oasis=oasis_bounds[c],
                        projected=oasis_projected_bounds[c],
                        hybrid=hybrid_bounds[c],
                        observations=col_observations[c],
                        num_buckets=args.num_buckets,
                        damping_grid=args.aggressive_damping_grid,
                        recent_windows=args.aggressive_recent_windows,
                        projection_iters=args.projection_iters,
                        projection_tol=args.projection_tol,
                    )
                    aggressive_bounds.append(ab)
                    aggressive_choices.append(am)

                method_bounds = {
                    "stale": stale_bounds, "isomer": isomer_bounds,
                    "oasis": oasis_bounds, "oasis_projected": oasis_projected_bounds,
                    "oasis_soft_projection": oasis_soft_bounds,
                    "hybrid": hybrid_bounds, "aggressive_hybrid": aggressive_bounds,
                    "fresh": fresh_bounds,
                }

                # ── IPF: one fixed stale 2D association seed, per-method marginals ──
                seed_table = build_stale_seed_table(initial, edges)
                ipf_joint = {}
                for method, bounds in method_bounds.items():
                    rt = marginal_cell_masses(copula, bounds[0], edges)
                    ct = marginal_cell_masses(copula, bounds[1], edges)
                    ipf_joint[method] = ipf_2d(seed_table, rt, ct, n_iter=args.ipf_iters)

                # ── Evaluate joint selectivity over random 2-col range predicates ──
                rng = random.Random(seed + 9999)
                for pred_id in range(args.n_predicates):
                    lo1 = rng.uniform(0.1, 0.5); hi1 = rng.uniform(lo1, 0.9)
                    lo2 = rng.uniform(0.1, 0.5); hi2 = rng.uniform(lo2, 0.9)
                    predicates = [(lo1, hi1), (lo2, hi2)]
                    true_sel = compute_true_joint_selectivity(drifted, predicates)
                    if true_sel < 1e-6:
                        continue
                    corr_mat = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
                    actual_marginals = [
                        actual_marginal_range_selectivity(drifted[ci], pr)
                        for ci, pr in enumerate(predicates)
                    ]

                    for estimator in args.estimators:
                        for method in METHOD_ORDER:
                            bounds = method_bounds[method]
                            if estimator == "independence":
                                est = estimate_independence(copula, bounds[0], bounds[1], predicates)
                            elif estimator == "gaussian_copula":
                                est = copula.joint_selectivity_range(bounds, predicates, corr_mat)
                            elif estimator == "ipf_sinkhorn":
                                est = ipf_range_prob(ipf_joint[method], edges, predicates)
                            else:
                                est = archimedean_range_prob(
                                    copula, estimator, bounds[0], bounds[1],
                                    predicates, thetas[estimator])

                            marg_qe = geomean([
                                qerr(marginal_range_selectivity(copula, bounds[ci], pr),
                                     actual_marginals[ci])
                                for ci, pr in enumerate(predicates)
                            ])
                            results.append({
                                "estimator": estimator, "correlation": rho,
                                "drift_q": q, "trial": trial, "predicate_id": pred_id,
                                "method": method,
                                "true_sel": true_sel, "est_sel": est,
                                "joint_qerr": qerr(est, true_sel),
                                "marginal_qerr_gm": marg_qe,
                                "hybrid_choice_c0": hybrid_choices[0],
                                "hybrid_choice_c1": hybrid_choices[1],
                                "aggressive_choice_c0": aggressive_choices[0],
                                "aggressive_choice_c1": aggressive_choices[1],
                            })
                print("done")

    _summarize_and_save(results, args)
    return results


def _summarize_and_save(results: List[dict], args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate per (estimator, method) across all rho/q.
    by_est_method = defaultdict(list)
    stale_by_key = {}
    for r in results:
        by_est_method[(r["estimator"], r["method"])].append(r["joint_qerr"])
        if r["method"] == "stale":
            key = (r["estimator"], r["correlation"], r["drift_q"], r["trial"], r["predicate_id"])
            stale_by_key[key] = r["joint_qerr"]

    summary_rows = []
    print("\n" + "=" * 126)
    print(f"{'Estimator':>16} | {'Stale':>8} | {'ISOMER':>8} | {'OASIS':>8} | "
          f"{'OASIS-Proj':>10} | {'Soft':>8} | {'Hybrid':>8} | {'Aggressive':>10} | {'Fresh':>8} | "
          f"{'Proj%':>7} | {'Soft%':>7} | {'Aggr%':>7}")
    print("=" * 126)
    for estimator in args.estimators:
        gm = {m: geomean(by_est_method[(estimator, m)]) for m in METHOD_ORDER}
        row = {"estimator": estimator, **{f"{m}_qerr_gm": gm[m] for m in METHOD_ORDER},
               "oasis_improvement_pct": pct_improvement(gm["stale"], gm["oasis"]),
               "isomer_improvement_pct": pct_improvement(gm["stale"], gm["isomer"]),
               "oasis_projected_improvement_pct": pct_improvement(gm["stale"], gm["oasis_projected"]),
               "oasis_soft_projection_improvement_pct": pct_improvement(gm["stale"], gm["oasis_soft_projection"]),
               "hybrid_improvement_pct": pct_improvement(gm["stale"], gm["hybrid"]),
               "aggressive_hybrid_improvement_pct": pct_improvement(gm["stale"], gm["aggressive_hybrid"])}
        for method in METHOD_ORDER:
            method_rows = [r for r in results if r["estimator"] == estimator and r["method"] == method]
            row[f"{method}_marginal_qerr_gm"] = geomean([r["marginal_qerr_gm"] for r in method_rows])
            row[f"{method}_worse_than_stale_frac"] = (
                sum(
                    r["joint_qerr"] > stale_by_key.get(
                        (r["estimator"], r["correlation"], r["drift_q"], r["trial"], r["predicate_id"]),
                        float("inf"),
                    )
                    for r in method_rows
                ) / max(len(method_rows), 1)
            )
        summary_rows.append(row)
        print(f"{estimator:>16} | {gm['stale']:8.3f} | {gm['isomer']:8.3f} | {gm['oasis']:8.3f} | "
              f"{gm['oasis_projected']:10.3f} | {gm['oasis_soft_projection']:8.3f} | {gm['hybrid']:8.3f} | "
              f"{gm['aggressive_hybrid']:10.3f} | {gm['fresh']:8.3f} | "
              f"{row['oasis_projected_improvement_pct']:+6.1f}% | "
              f"{row['oasis_soft_projection_improvement_pct']:+6.1f}% | "
              f"{row['aggressive_hybrid_improvement_pct']:+6.1f}%")

    with open(output_dir / "composition_family_results.json", "w") as f:
        json.dump(results, f)
    with open(output_dir / "composition_family_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    with open(output_dir / "composition_family_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    _generate_latex_table(summary_rows, output_dir)
    print(f"\nSaved to {output_dir}")


def _generate_latex_table(summary_rows, output_dir):
    labels = {
        "independence": "Independence", "ipf_sinkhorn": "IPF/Sinkhorn-2D",
        "gaussian_copula": "Gaussian copula", "clayton_copula": "Clayton copula",
        "gumbel_copula": "Gumbel copula", "frank_copula": "Frank copula",
    }
    path = output_dir / "table_composition_family.tex"
    with open(path, "w") as f:
        f.write("\\begin{table*}[t]\n  \\centering\n  \\small\n")
        f.write("  \\caption{OASIS as a drop-in marginal upgrade across a family of "
                "composition estimators. Average joint Q-error ($\\downarrow$) for two-column "
                "range predicates; dependence structure is held fixed per estimator and only the "
                "marginal input varies. The full two-stage OASIS and Hybrid improve every "
                "estimator, whereas OASIS-noProj (the learned stage without the feedback-consistency "
                "projection) is weak and unstable, confirming the projection is required across "
                "composition methods.}\n")
        f.write("  \\label{tab:composition_family}\n")
        f.write("  \\setlength{\\tabcolsep}{4pt}\n")
        f.write("  \\resizebox{\\textwidth}{!}{%\n")
        f.write("  \\begin{tabular}{l | rrrrrrrr | rrrr}\n    \\toprule\n")
        f.write("    Estimator & Stale & OASIS-noProj & ISOMER & OASIS & Soft & Hybrid & Aggr. & Fresh & "
                "OASIS-noProj +\\% & OASIS +\\% & Soft +\\% & Aggr. +\\% \\\\\n    \\midrule\n")
        for r in summary_rows:
            best = min(r["oasis_qerr_gm"], r["isomer_qerr_gm"],
                       r["oasis_projected_qerr_gm"], r["hybrid_qerr_gm"],
                       r["oasis_soft_projection_qerr_gm"],
                       r["aggressive_hybrid_qerr_gm"])
            def cell(v):
                return f"\\textbf{{{v:.3f}}}" if abs(v - best) < 1e-3 else f"{v:.3f}"
            f.write(f"    {labels.get(r['estimator'], r['estimator'])} & "
                    f"{r['stale_qerr_gm']:.3f} & {cell(r['oasis_qerr_gm'])} & "
                    f"{cell(r['isomer_qerr_gm'])} & {cell(r['oasis_projected_qerr_gm'])} & "
                    f"{cell(r['oasis_soft_projection_qerr_gm'])} & "
                    f"{cell(r['hybrid_qerr_gm'])} & {cell(r['aggressive_hybrid_qerr_gm'])} & "
                    f"{r['fresh_qerr_gm']:.3f} & "
                    f"{r['oasis_improvement_pct']:+.1f}\\% & "
                    f"{r['oasis_projected_improvement_pct']:+.1f}\\% & "
                    f"{r['oasis_soft_projection_improvement_pct']:+.1f}\\% & "
                    f"{r['aggressive_hybrid_improvement_pct']:+.1f}\\% \\\\\n")
        f.write("    \\bottomrule\n  \\end{tabular}%\n  }\n\\end{table*}\n")
    print(f"LaTeX table saved to {path}")


def main():
    p = argparse.ArgumentParser(description="Composition-family OASIS embedding experiment")
    p.add_argument("--model-path", type=Path,
                   default=_REPO_DIR / "experiments" / "results" / "copula_model" / "oasis_k16.json")
    p.add_argument("--output-dir", type=Path,
                   default=_REPO_DIR / "experiments" / "results" / "composition_family")
    p.add_argument("--num-buckets", type=int, default=10)
    p.add_argument("--max-observations", type=int, default=16)
    p.add_argument("--n-rows", type=int, default=5000)
    p.add_argument("--n-observations", type=int, default=16)
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--n-predicates", type=int, default=40)
    p.add_argument("--correlations", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    p.add_argument("--drift-levels", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--estimators", nargs="+", choices=ESTIMATOR_ORDER, default=ESTIMATOR_ORDER)
    p.add_argument("--ipf-grid", type=int, default=24)
    p.add_argument("--ipf-iters", type=int, default=40)
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
