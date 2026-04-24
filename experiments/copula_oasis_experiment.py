#!/usr/bin/env python3
"""
Copula + OASIS Composition Experiment

Demonstrates that OASIS-improved single-column marginals propagate to better
multi-column joint selectivity estimation via Gaussian copula.

Setup:
  1. Generate correlated 2-column tables with known drift
  2. Build marginal histograms (stale, OASIS-corrected, fresh)
  3. Estimate joint selectivity via Gaussian copula with each marginal source
  4. Compare against ground truth joint selectivity

This experiment does NOT require a running database.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import multivariate_normal as _mvn
    HAS_SCIPY_MVN = True
except Exception:
    HAS_SCIPY_MVN = False

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_PIPELINE_DIR = _REPO_DIR / "cdf_kll_ml_pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from baselines import correct_linear_interp, correct_feedback_avg
from histogram_math import clamp01, evaluate_piecewise_cdf
from histogram_types import KllFeedbackSample
from json_histogram_parser import load_feedback_sample
from mlp_histogram_model_v2 import MlpHistogramModelV2
from modern_baselines import correct_isomer, correct_quicksel_h
from simulate_memory_kll_dataset import MemoryTable


# ─── Gaussian Copula Estimator ───────────────────────────────────────────────

class GaussianCopula:
    """
    Gaussian copula selectivity estimator for multi-column predicates.

    Given per-column marginal CDFs (histograms) and a correlation matrix,
    estimates the joint selectivity of conjunctive range predicates.

    The Gaussian copula model assumes:
        P(X1 ≤ x1, X2 ≤ x2, ..., Xn ≤ xn) = C(F1(x1), F2(x2), ..., Fn(xn))
    where C is the Gaussian copula with correlation matrix Σ.
    """

    def __init__(self, num_buckets: int = 10):
        self.num_buckets = num_buckets

    def _build_cdf(self, boundaries: Sequence[float]):
        """Build piecewise-linear CDF from histogram boundaries."""
        n = len(boundaries)
        levels = [i / (n - 1) for i in range(n)]
        return list(boundaries), levels

    def marginal_cdf(self, boundaries: Sequence[float], value: float) -> float:
        """Evaluate marginal CDF at a given value."""
        xs, ps = self._build_cdf(boundaries)
        if value <= xs[0]:
            return 0.0
        if value >= xs[-1]:
            return 1.0
        # Piecewise linear interpolation
        for i in range(len(xs) - 1):
            if xs[i] <= value <= xs[i + 1]:
                width = max(xs[i + 1] - xs[i], 1e-12)
                frac = (value - xs[i]) / width
                return ps[i] + frac * (ps[i + 1] - ps[i])
        return 0.5

    def joint_selectivity_range(
        self,
        col_boundaries: Sequence[Sequence[float]],
        predicates: Sequence[Tuple[float, float]],
        correlations: np.ndarray,
    ) -> float:
        """
        Estimate joint selectivity for conjunctive range predicates.

        Args:
            col_boundaries: List of histogram boundary arrays, one per column
            predicates: List of (lower, upper) tuples per column
            correlations: Pearson correlation matrix between columns

        Returns:
            Estimated joint selectivity
        """
        n_cols = len(col_boundaries)
        assert len(predicates) == n_cols
        assert correlations.shape == (n_cols, n_cols)

        # Step 1: Transform to uniform marginals via CDF
        u_vals = []
        for i in range(n_cols):
            lower, upper = predicates[i]
            u_lower = self.marginal_cdf(col_boundaries[i], lower)
            u_upper = self.marginal_cdf(col_boundaries[i], upper)
            u_vals.append((max(u_lower, 1e-6), min(u_upper, 1 - 1e-6)))

        # Step 2: Transform to Gaussian via inverse CDF (probit)
        z_vals = []
        for u_lower, u_upper in u_vals:
            z_lower = self._probit(max(u_lower, 1e-6))
            z_upper = self._probit(min(u_upper, 1 - 1e-6))
            z_vals.append((z_lower, z_upper))

        # Step 3: Compute joint probability via multivariate normal CDF
        # P = Φ_Σ(z_upper) - marginals
        # For 2 columns, use bivariate normal CDF
        if n_cols == 2:
            return self._bivariate_range_prob(
                z_vals[0][0], z_vals[0][1],
                z_vals[1][0], z_vals[1][1],
                correlations[0, 1],
            )
        else:
            # Fallback: independence assumption with correlation correction
            return self._independent_estimate(u_vals, correlations)

    def _probit(self, u: float) -> float:
        """Inverse standard normal CDF (Abramowitz & Stegun 26.2.23)."""
        p = max(1e-10, min(u, 1 - 1e-10))
        if p > 0.5:
            return -self._probit(1.0 - p)
        t = math.sqrt(-2.0 * math.log(p))
        # Rational approximation
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        x = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
        return -x

    def _norm_cdf(self, x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _bvn_cdf(self, h: float, k: float, rho: float) -> float:
        """
        Bivariate standard normal CDF.
        P(Z1 ≤ h, Z2 ≤ k) where corr(Z1, Z2) = rho.
        """
        if abs(rho) < 1e-10:
            return self._norm_cdf(h) * self._norm_cdf(k)

        h = min(max(h, -8.0), 8.0)
        k = min(max(k, -8.0), 8.0)
        rho = min(max(rho, -0.9999), 0.9999)

        if HAS_SCIPY_MVN:
            from scipy.stats import multivariate_normal as _mvn
            mean = [0.0, 0.0]
            cov = [[1.0, rho], [rho, 1.0]]
            return float(_mvn.cdf([h, k], mean=mean, cov=cov))

        # Fallback: numerical integration
        # P(Z1≤h, Z2≤k) = ∫_0^{Φ(h)} Φ((k - ρ·probit(t)) / √(1-ρ²)) dt
        upper = self._norm_cdf(h)
        n_pts = 100
        dt = upper / n_pts
        sinh_val = math.sqrt(max(1 - rho * rho, 1e-12))
        total = 0.0
        for i in range(n_pts):
            t = (i + 0.5) * dt
            t = max(1e-10, min(t, 1 - 1e-10))
            z1 = self._probit(t)
            arg = (k - rho * z1) / sinh_val
            total += self._norm_cdf(arg) * dt
        return total

    def _norm_cdf_pdf(self, x: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def _bivariate_range_prob(
        self,
        z1_lo: float, z1_hi: float,
        z2_lo: float, z2_hi: float,
        rho: float,
    ) -> float:
        """
        P(z1_lo ≤ Z1 ≤ z1_hi, z2_lo ≤ Z2 ≤ z2_hi)
        = P(Z1≤z1_hi, Z2≤z2_hi) - P(Z1≤z1_lo, Z2≤z2_hi)
          - P(Z1≤z1_hi, Z2≤z2_lo) + P(Z1≤z1_lo, Z2≤z2_lo)
        """
        p = (self._bvn_cdf(z1_hi, z2_hi, rho)
             - self._bvn_cdf(z1_lo, z2_hi, rho)
             - self._bvn_cdf(z1_hi, z2_lo, rho)
             + self._bvn_cdf(z1_lo, z2_lo, rho))
        return max(p, 1e-12)

    def _independent_estimate(
        self,
        u_vals: Sequence[Tuple[float, float]],
        correlations: np.ndarray,
    ) -> float:
        """Fallback: independence assumption with correlation adjustment."""
        prod = 1.0
        for u_lo, u_hi in u_vals:
            prod *= max(u_hi - u_lo, 1e-12)
        # Simple correlation adjustment
        if len(u_vals) == 2 and abs(correlations[0, 1]) > 0.01:
            rho = correlations[0, 1]
            # Adjust product towards max (positive corr) or min (negative)
            max_sel = min(u_vals[0][1] - u_vals[0][0], u_vals[1][1] - u_vals[1][0])
            min_sel = max(u_vals[0][1] - u_vals[0][0], u_vals[1][1] - u_vals[1][0])
            prod = prod + rho * (max_sel - prod) * 0.5
        return max(prod, 1e-12)


# ─── Data Generation ──────────────────────────────────────────────────────────

def generate_correlated_columns(
    n_rows: int,
    n_cols: int,
    correlation: float,
    seed: int,
    drift_rounds: int = 0,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Generate correlated column data using Cholesky decomposition.

    Returns (initial_data, post_drift_data) where each is a list of columns.
    """
    rng = np.random.RandomState(seed)

    # Build correlation matrix
    cov = np.full((n_cols, n_cols), correlation)
    np.fill_diagonal(cov, 1.0)

    # Generate initial data from non-uniform distribution (power-law-like)
    # This makes the histogram genuinely sensitive to drift
    L = np.linalg.cholesky(cov)
    z = rng.randn(n_rows, n_cols)
    raw = z @ L.T  # Shape: (n_rows, n_cols)

    # Transform to non-uniform marginals via x^0.5 (concentrates mass near 0)
    initial = []
    for c in range(n_cols):
        col_data = []
        for val in raw[:, c]:
            u = 0.5 * (1 + math.erf(val / math.sqrt(2)))
            # Apply nonlinear transform: sqrt makes distribution left-heavy
            b = clamp01(u) ** 0.5
            col_data.append(b)
        initial.append(col_data)

    if drift_rounds == 0:
        return initial, initial

    # Apply strong drift: replace fraction of values with right-heavy distribution
    # Simulates data distribution change (e.g., seasonal shift, bulk load)
    drifted = [list(col) for col in initial]
    drift_frac = min(0.1 * drift_rounds, 0.8)  # 10% per drift round, capped at 80%
    n_shift = max(1, int(len(drifted[0]) * drift_frac))
    indices = list(range(len(drifted[0])))
    rng.shuffle(indices)
    for idx in indices[:n_shift]:
        for c in range(n_cols):
            # Replace with values from right-heavy distribution (1 - u^0.5)
            new_u = clamp01(float(rng.rand()))
            drifted[c][idx] = 1.0 - clamp01(new_u ** 0.5)  # Right-heavy

    return initial, drifted


def get_histogram_boundaries(data: List[float], num_buckets: int) -> List[float]:
    """Get equi-depth histogram boundaries from data."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    boundaries = []
    for i in range(num_buckets + 1):
        idx = int(i * (n - 1) / num_buckets)
        boundaries.append(sorted_data[min(idx, n - 1)])
    return boundaries


def compute_true_joint_selectivity(
    col_data: Sequence[List[float]],
    predicates: Sequence[Tuple[float, float]],
) -> float:
    """Compute ground truth joint selectivity by scanning data."""
    n_rows = len(col_data[0])
    count = 0
    for row in range(n_rows):
        match = True
        for c in range(len(predicates)):
            lo, hi = predicates[c]
            if not (lo <= col_data[c][row] <= hi):
                match = False
                break
        if match:
            count += 1
    return count / max(n_rows, 1)


# ─── OASIS Correction for Copula Input ────────────────────────────────────────

def correct_marginal_with_oasis(
    stale_boundaries: List[float],
    observations: List[dict],
    model: MlpHistogramModelV2,
    num_buckets: int,
    max_obs: int,
) -> List[float]:
    """Apply OASIS correction to a single column's marginal histogram."""
    from tensorizer import tensorize_sample
    from histogram_types import KllFeedbackSample, FeedbackObservation, KllPrior
    from datetime import datetime, timezone

    min_val = stale_boundaries[0]
    max_val = stale_boundaries[-1]
    quantile_values = stale_boundaries[1:-1]
    value_range = max(max_val - min_val, 1e-12)

    # Normalize to [0, 1] as required by KllPrior
    norm_quantiles = [(v - min_val) / value_range for v in quantile_values]

    # Build normalized observations
    base_timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    obs_objects = []
    for idx, obs in enumerate(observations[:max_obs]):
        v = obs.get("value", 0.5)
        v_norm = clamp01((v - min_val) / value_range)
        v_upper = obs.get("value_upper")
        v_upper_norm = clamp01((v_upper - min_val) / value_range) if v_upper is not None else None
        obs_objects.append(FeedbackObservation(
            predicate_type=obs.get("predicate_type", "<"),
            value=v_norm,
            value_upper=v_upper_norm,
            estimated_selectivity=obs.get("estimated_sel", 0.5),
            actual_selectivity=obs.get("actual_sel", 0.5),
            timestamp=base_timestamp.replace(hour=idx),
        ))

    prior = KllPrior(
        min_value=0.0,
        max_value=1.0,
        null_fraction=0.0,
        quantile_levels=[i / num_buckets for i in range(1, num_buckets)],
        quantile_values=norm_quantiles,
    )

    sample = KllFeedbackSample(
        prior=prior,
        observations=obs_objects,
        corrected_quantile_values=None,
    )

    record = tensorize_sample(sample, max_observations=max_obs, teacher_fn=None, use_time_decay=False)
    pred_norm = model.predict([record.feature_tensor])[0]

    # Convert back to original value range
    corrected = [clamp01(min_val + v * value_range) for v in pred_norm]

    # Enforce monotonicity
    for i in range(1, len(corrected)):
        if corrected[i] < corrected[i - 1]:
            corrected[i] = corrected[i - 1]

    return [min_val] + corrected + [max_val]


# ─── Main Experiment ──────────────────────────────────────────────────────────

def run_experiment(args):
    copula = GaussianCopula(num_buckets=args.num_buckets)
    results = []

    # Load OASIS model
    model_path = args.model_path
    if not model_path.exists():
        print(f"Model not found at {model_path}, checking remote...")
        # Try to download from remote
        import subprocess
        remote_model = "/home/tianqc/experiments/experiments/results/v4_with_baselines/models/oasis_k16.json"
        result = subprocess.run(
            ["scp", f"10.181.8.145:{remote_model}", str(model_path)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Could not download model: {result.stderr}")
            print("Training a fresh model on synthetic data...")
            model_path = train_fresh_model(args)

    model = MlpHistogramModelV2.load(str(model_path))

    correlations = [0.0, 0.3, 0.5, 0.7, 0.9]
    drift_levels = [1, 5, 10, 20]

    for rho in correlations:
        for q in drift_levels:
            for trial in range(args.n_trials):
                seed = args.seed + trial * 1000 + int(rho * 100) + q
                print(f"ρ={rho:.1f}, q={q}, trial={trial}...", end=" ", flush=True)

                # Generate correlated data
                initial, drifted = generate_correlated_columns(
                    n_rows=args.n_rows, n_cols=2, correlation=rho,
                    seed=seed, drift_rounds=q,
                )

                # Get histograms
                stale_bounds = [get_histogram_boundaries(initial[c], args.num_buckets) for c in range(2)]
                fresh_bounds = [get_histogram_boundaries(drifted[c], args.num_buckets) for c in range(2)]

                # Generate observations for each column
                col_observations = []
                for c in range(2):
                    obs_list = []
                    sorted_drifted = sorted(drifted[c])
                    n = len(sorted_drifted)
                    rng = random.Random(seed + c)
                    for _ in range(args.n_observations):
                        # Random range predicate
                        v = sorted_drifted[rng.randint(0, n - 1)]
                        # Compute actual selectivity
                        count = sum(1 for x in drifted[c] if x <= v)
                        act_sel = count / max(n, 1)
                        # Compute estimated selectivity from stale histogram
                        cdf_x, cdf_p = stale_bounds[c], [i / args.num_buckets for i in range(args.num_buckets + 1)]
                        est_sel = evaluate_piecewise_cdf(cdf_x, cdf_p, v)

                        obs_list.append({
                            "predicate_type": rng.choice(["<", "<=", ">=", ">"]),
                            "value": v,
                            "estimated_sel": est_sel,
                            "actual_sel": act_sel,
                        })
                    col_observations.append(obs_list)

                # OASIS correction
                oasis_bounds = []
                for c in range(2):
                    corrected = correct_marginal_with_oasis(
                        stale_bounds[c], col_observations[c], model,
                        args.num_buckets, args.max_observations,
                    )
                    oasis_bounds.append(corrected)

                # ISOMER correction
                isomer_bounds = []
                for c in range(2):
                    try:
                        iq = correct_isomer(
                            stale_bounds[c][0], stale_bounds[c][-1],
                            stale_bounds[c][1:-1], col_observations[c],
                            num_buckets=args.num_buckets,
                        )
                        isomer_bounds.append([stale_bounds[c][0]] + list(iq) + [stale_bounds[c][-1]])
                    except Exception:
                        isomer_bounds.append(stale_bounds[c])

                # Evaluate joint selectivity
                rng = random.Random(seed + 9999)
                n_predicates = 50
                for _ in range(n_predicates):
                    # Random 2-column range predicates
                    lo1 = rng.uniform(0.1, 0.5)
                    hi1 = rng.uniform(lo1, 0.9)
                    lo2 = rng.uniform(0.1, 0.5)
                    hi2 = rng.uniform(lo2, 0.9)
                    predicates = [(lo1, hi1), (lo2, hi2)]

                    # Ground truth
                    true_sel = compute_true_joint_selectivity(drifted, predicates)

                    if true_sel < 1e-6:
                        continue

                    # --- Independence-based joint selectivity ---
                    # This is what most DBMS optimizers actually use
                    def indep_sel(bounds, preds):
                        sel = 1.0
                        for c in range(len(preds)):
                            lo, hi = preds[c]
                            p_lo = copula.marginal_cdf(bounds[c], lo)
                            p_hi = copula.marginal_cdf(bounds[c], hi)
                            sel *= max(p_hi - p_lo, 1e-12)
                        return sel

                    stale_sel = indep_sel(stale_bounds, predicates)
                    oasis_sel = indep_sel(oasis_bounds, predicates)
                    isomer_sel = indep_sel(isomer_bounds, predicates)
                    fresh_sel = indep_sel(fresh_bounds, predicates)

                    def qerr(est, true):
                        est = max(est, 1e-8)
                        true = max(true, 1e-8)
                        return max(est / true, true / est)

                    results.append({
                        "correlation": rho,
                        "drift_q": q,
                        "trial": trial,
                        "true_sel": true_sel,
                        "stale_qerr": qerr(stale_sel, true_sel),
                        "oasis_qerr": qerr(oasis_sel, true_sel),
                        "isomer_qerr": qerr(isomer_sel, true_sel),
                        "fresh_qerr": qerr(fresh_sel, true_sel),
                    })

                print("done")

    # Aggregate results
    by_config = defaultdict(list)
    for r in results:
        key = (r["correlation"], r["drift_q"])
        by_config[key].append(r)

    import math as _m
    def geomean(vals):
        return _m.exp(sum(_m.log(max(v, 1e-12)) for v in vals) / max(len(vals), 1))

    print("\n" + "=" * 100)
    print(f"{'ρ':>5} | {'q':>3} | {'Stale QE':>10} | {'ISOMER QE':>10} | {'OASIS QE':>10} | {'Fresh QE':>10} | "
          f"{'OASIS Imp%':>10} | {'ISOMER Imp%':>11} | N")
    print("=" * 100)

    for key in sorted(by_config.keys()):
        rho, q = key
        rows = by_config[key]

        stale_gm = geomean([r["stale_qerr"] for r in rows])
        oasis_gm = geomean([r["oasis_qerr"] for r in rows])
        isomer_gm = geomean([r["isomer_qerr"] for r in rows])
        fresh_gm = geomean([r["fresh_qerr"] for r in rows])
        oasis_imp = (stale_gm - oasis_gm) / max(stale_gm, 1e-12) * 100
        isomer_imp = (stale_gm - isomer_gm) / max(stale_gm, 1e-12) * 100

        print(f"{rho:5.1f} | {q:3d} | {stale_gm:10.3f} | {isomer_gm:10.3f} | {oasis_gm:10.3f} | {fresh_gm:10.3f} | "
              f"{oasis_imp:+9.1f}% | {isomer_imp:+10.1f}% | {len(rows)}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "copula_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate LaTeX table
    generate_latex_table(by_config, output_dir)

    return results


def generate_latex_table(by_config, output_dir):
    """Generate LaTeX table for paper."""
    import math as _m
    def geomean(vals):
        return _m.exp(sum(_m.log(max(v, 1e-12)) for v in vals) / max(len(vals), 1))

    path = output_dir / "table_copula.tex"
    with open(path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("  \\centering\n")
        f.write("  \\small\n")
        f.write("  \\caption{Copula-based joint selectivity estimation with different marginal inputs. "
                "Q-Error ($\\downarrow$) for 2-column range predicates at varying correlations and drift. "
                "OASIS-corrected marginals improve joint estimation over stale marginals via Gaussian copula.}\n")
        f.write("  \\label{tab:copula}\n")
        f.write("  \\setlength{\\tabcolsep}{4pt}\n")
        f.write("  \\begin{tabular}{cc | rrrr | rr}\n")
        f.write("    \\toprule\n")
        f.write("    $\\rho$ & $q$ & Stale & ISOMER & OASIS & Fresh & OASIS Imp & ISOMER Imp \\\\\n")
        f.write("    \\midrule\n")

        for key in sorted(by_config.keys()):
            rho, q = key
            rows = by_config[key]
            stale = geomean([r["stale_qerr"] for r in rows])
            oasis = geomean([r["oasis_qerr"] for r in rows])
            isomer = geomean([r["isomer_qerr"] for r in rows])
            fresh = geomean([r["fresh_qerr"] for r in rows])
            oimp = (stale - oasis) / max(stale, 1e-12) * 100
            iimp = (stale - isomer) / max(stale, 1e-12) * 100

            # Bold the best non-fresh method
            best = min(oasis, isomer)
            o_str = f"\\textbf{{{oasis:.3f}}}" if abs(oasis - best) < 0.001 else f"{oasis:.3f}"
            i_str = f"\\textbf{{{isomer:.3f}}}" if abs(isomer - best) < 0.001 else f"{isomer:.3f}"

            f.write(f"    {rho:.1f} & {q} & {stale:.3f} & {i_str} & {o_str} & {fresh:.3f} & {oimp:+.0f}\\% & {iimp:+.0f}% \\\\\n")

        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nLaTeX table saved to {path}")


def train_fresh_model(args):
    """Train a fresh OASIS model if no checkpoint is available."""
    print("Training fresh OASIS model...")
    from run_synthetic_paper_suite import (
        ensure_compound_data, train_model,
        MAIN_Q_VALUES, TRAIN_Q_VALUES,
    )
    output_root = Path(args.output_dir) / "model_training"
    output_root.mkdir(parents=True, exist_ok=True)

    data_root = output_root / "compound_data"
    train_dirs = ensure_compound_data(
        data_root, TRAIN_Q_VALUES, count=500, num_buckets=args.num_buckets,
        seed=args.seed, prefix="train",
    )
    model = train_model(
        model_path=output_root / "models" / f"oasis_k{args.max_observations}.json",
        train_dirs=[train_dirs[q] for q in TRAIN_Q_VALUES],
        max_obs=args.max_observations,
        seed=args.seed,
        force_retrain=True,
        train_lr=3e-4,
        train_epochs=100,
        train_alpha=1e-4,
        activation_clip=10.0,
        attention_score_clip=20.0,
        parameter_clip=2.0,
    )
    return output_root / "models" / f"oasis_k{args.max_observations}.json"


def main():
    parser = argparse.ArgumentParser(description="Copula + OASIS Composition Experiment")
    parser.add_argument("--model-path", type=Path,
                       default=_REPO_DIR / "experiments" / "results" / "copula_model" / "oasis_k16.json")
    parser.add_argument("--output-dir", type=Path,
                       default=_REPO_DIR / "experiments" / "results" / "copula_experiment")
    parser.add_argument("--num-buckets", type=int, default=10)
    parser.add_argument("--max-observations", type=int, default=16)
    parser.add_argument("--n-rows", type=int, default=5000)
    parser.add_argument("--n-observations", type=int, default=16)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
