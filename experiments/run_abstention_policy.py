#!/usr/bin/env python3
"""
Abstention Policy: Coverage-Risk Curve for OASIS Deployment
============================================================

Evaluates a simple delta-threshold + K_min trigger policy:
  - Apply OASIS correction only when:
    1. The predicted correction delta exceeds delta_min (norm threshold), AND
    2. At least K_min valid observations are available
  - Otherwise, fall back to the stale prior

Generates a coverage-risk curve:
  - X-axis: coverage rate (% of samples where OASIS is applied)
  - Y-axis: worst-case Q-Error (protection against degradation)
  - Family of curves parameterized by (delta_min, K_min)

This is the minimum evidence needed to move the abstention policy
from "design prose" to "evidence."
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_PIPELINE_DIR = _REPO_DIR / "cdf_kll_ml_pipeline"

if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

import numpy as np

from histogram_math import clamp01
from histogram_types import KllFeedbackSample
from json_histogram_parser import load_feedback_sample
from mlp_histogram_model_v2 import MlpHistogramModelV2
from tensorizer import tensorize_sample


# ═══════════════════════════════════════════════════════════════════════════
# Abstention Policy
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TriggerConfig:
    """Configuration for the abstention trigger."""
    delta_min: float  # Minimum predicted correction delta (L2 norm)
    k_min: int         # Minimum number of valid observations


@dataclass
class TriggerResult:
    """Result of applying the trigger policy to one sample."""
    sample_id: str
    delta_norm: float
    k_valid: int
    triggered: bool
    used_oasis: bool
    prior_qerror: float
    oasis_qerror: float
    applied_qerror: float  # OASIS if triggered, prior otherwise
    q: int


def predict_oasis(
    sample: KllFeedbackSample,
    model: MlpHistogramModelV2,
    max_obs: int,
) -> Tuple[List[float], float]:
    """OASIS prediction + delta norm."""
    record = tensorize_sample(sample, max_observations=max_obs, teacher_fn=None, use_time_decay=False)
    feature = record.feature_tensor

    prior_quantiles = np.array(sample.prior.quantile_values, dtype=np.float64)
    pred_norm = model.predict([feature])[0]
    pred_norm = np.array(pred_norm, dtype=np.float64)

    # Delta norm: L2 distance between predicted and prior quantiles (normalized)
    delta = np.sqrt(np.mean((pred_norm - prior_quantiles) ** 2))

    value_range = max(sample.prior.value_range, 1e-12)
    quantiles = [clamp01(sample.prior.min_value + v * value_range) for v in pred_norm]
    for idx in range(1, len(quantiles)):
        if quantiles[idx] < quantiles[idx - 1]:
            quantiles[idx] = quantiles[idx - 1]

    return [sample.prior.min_value] + quantiles + [sample.prior.max_value], float(delta)


def apply_trigger(
    sample: KllFeedbackSample,
    model: MlpHistogramModelV2,
    max_obs: int,
    config: TriggerConfig,
    prior_boundaries: List[float],
    true_boundaries: List[float],
    eval_points: List[float],
    q_mods: int,
) -> TriggerResult:
    """Apply abstention trigger policy to a sample."""
    # Count valid observations
    k_valid = len(sample.observations)

    # Get OASIS prediction
    oasis_bounds, delta_norm = predict_oasis(sample, model, max_obs)

    # Trigger decision
    triggered = (delta_norm >= config.delta_min) and (k_valid >= config.k_min)

    from histogram_math import evaluate_piecewise_cdf
    eps = 1e-6

    def compute_qe(bounds, truth):
        errors = []
        for point in eval_points:
            est = max(evaluate_piecewise_cdf(bounds,
                     [i/(len(bounds)-1) for i in range(len(bounds))], point), eps)
            act = max(evaluate_piecewise_cdf(truth,
                     [i/(len(truth)-1) for i in range(len(truth))], point), eps)
            errors.append(max(est / act, act / est))
        return sum(errors) / len(errors)

    prior_qe = compute_qe(prior_boundaries, true_boundaries)
    oasis_qe = compute_qe(oasis_bounds, true_boundaries)
    applied_qe = oasis_qe if triggered else prior_qe

    return TriggerResult(
        sample_id=getattr(sample, 'source_path', 'unknown'),
        delta_norm=delta_norm,
        k_valid=k_valid,
        triggered=triggered,
        used_oasis=triggered,
        prior_qerror=prior_qe,
        oasis_qerror=oasis_qe,
        applied_qerror=applied_qe,
        q=q_mods,
    )


def scan_thresholds(
    samples: List[KllFeedbackSample],
    model: MlpHistogramModelV2,
    max_obs: int,
    q_mods: int,
    delta_range: List[float],
    k_range: List[int],
    eval_points: List[float],
) -> List[dict]:
    """Scan over (delta_min, K_min) to build coverage-risk curve."""
    results = []

    for k_min in k_range:
        for delta_min in delta_range:
            config = TriggerConfig(delta_min=delta_min, k_min=k_min)

            triggered_samples = []
            all_results = []
            for sample in samples:
                true_bounds = ([sample.prior.min_value]
                               + list(sample.corrected_quantile_values or sample.prior.quantile_values)
                               + [sample.prior.max_value])
                prior_bounds = ([sample.prior.min_value]
                                + list(sample.prior.quantile_values)
                                + [sample.prior.max_value])

                tr = apply_trigger(
                    sample, model, max_obs, config,
                    prior_bounds, true_bounds, eval_points, q_mods,
                )
                all_results.append(tr)
                if tr.triggered:
                    triggered_samples.append(tr)

            coverage = len(triggered_samples) / max(len(samples), 1)

            # Worst-case: max Q-Error among triggered samples
            worst_case = max((tr.applied_qerror for tr in all_results), default=1.0)

            # Mean Q-Error with trigger policy
            mean_qe = sum(tr.applied_qerror for tr in all_results) / max(len(all_results), 1)

            # How many would OASIS have degraded if always applied?
            n_would_degrade = sum(
                1 for tr in all_results if tr.oasis_qerror > tr.prior_qerror
            )
            # How many degradations does trigger PREVENT?
            n_prevented = sum(
                1 for tr in all_results
                if tr.oasis_qerror > tr.prior_qerror and not tr.triggered
            )

            results.append({
                "delta_min": delta_min,
                "k_min": k_min,
                "coverage": coverage,
                "worst_case_qerror": worst_case,
                "mean_qerror": mean_qe,
                "n_triggered": len(triggered_samples),
                "n_total": len(samples),
                "n_would_degrade": n_would_degrade,
                "n_prevented": n_prevented,
                "q": q_mods,
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Abstention Policy: Coverage-Risk Curve"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="results/abstention_policy")
    parser.add_argument("--q-values", type=int, nargs="+",
                        default=[1, 3, 5, 10, 15, 20])
    parser.add_argument("--max-observations", type=int, default=16)
    parser.add_argument("--num-buckets", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading OASIS model...")
    model = MlpHistogramModelV2.load(str(Path(args.model_path)))

    data_dir = Path(args.data_dir)
    rng = random.Random(args.seed)

    # Threshold ranges to scan
    delta_range = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
    k_range = [0, 1, 2, 4, 8, 12, 16]

    all_results = []

    for q in args.q_values:
        q_dir = data_dir / f"test_q{q}"
        if not q_dir.exists():
            print(f"  Skipping q={q}: no data at {q_dir}")
            continue

        print(f"\n=== q={q} ===")
        samples = []
        for path in sorted(q_dir.glob("*.json")):
            sample = load_feedback_sample(str(path))
            if sample.corrected_quantile_values is not None:
                samples.append(sample)

        if len(samples) > 64:
            samples = rng.sample(samples, 64)
        print(f"  {len(samples)} samples")

        eval_points = [rng.uniform(0.05, 0.95) for _ in range(50)]

        q_results = scan_thresholds(
            samples, model, args.max_observations, q,
            delta_range, k_range, eval_points,
        )
        all_results.extend(q_results)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "coverage_risk_scan.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Find Pareto-optimal configurations for each q
    print("\n=== Pareto-Optimal Configurations ===")
    for q in args.q_values:
        q_results = [r for r in all_results if r["q"] == q]
        if not q_results:
            continue

        # Filter: coverage > 50%, worst_case < prior QE
        good = [r for r in q_results if r["coverage"] > 0.5 and r["n_prevented"] > 0]
        good.sort(key=lambda r: (-r["coverage"], r["worst_case_qerror"]))

        print(f"\n  q={q}:")
        # Top 3 configurations
        for r in good[:3]:
            print(
                f"    delta_min={r['delta_min']:.3f}, K_min={r['k_min']}: "
                f"coverage={r['coverage']:.1%}, worst_QE={r['worst_case_qerror']:.3f}, "
                f"mean_QE={r['mean_qerror']:.3f}, prevented={r['n_prevented']}/{r['n_would_degrade']}"
            )

    # Write LaTeX table
    table_path = output_dir / "table_abstention.tex"
    with open(table_path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("  \\centering\n")
        f.write("  \\caption{Abstention policy coverage-risk analysis. ")
        f.write("For each drift intensity $q$, we show configurations ")
        f.write("that achieve high coverage while preventing OASIS-induced degradations.}\n")
        f.write("  \\label{tab:abstention}\n")
        f.write("  \\small\n")
        f.write("  \\begin{tabular}{c c c c c c c}\n")
        f.write("    \\toprule\n")
        f.write("    $q$ & $\\delta_{\\min}$ & $K_{\\min}$ & Coverage & Mean QE & Worst QE & Degrad. Prevented \\\\\n")
        f.write("    \\midrule\n")

        for q in args.q_values:
            q_results = [r for r in all_results if r["q"] == q]
            if not q_results:
                continue
            good = [r for r in q_results if r["coverage"] > 0.5 and r["n_prevented"] > 0]
            if not good:
                # Show best available
                good = sorted(q_results, key=lambda r: (-r["coverage"], r["worst_case_qerror"]))
            good.sort(key=lambda r: (-r["coverage"], r["worst_case_qerror"]))
            best = good[0]
            f.write(
                f"    {q:2d} & {best['delta_min']:.3f} & {best['k_min']:2d} & "
                f"{best['coverage']:.1%} & {best['mean_qerror']:.3f} & "
                f"{best['worst_case_qerror']:.3f} & "
                f"{best['n_prevented']}/{best['n_would_degrade']} \\\\\n"
            )

        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nLaTeX table: {table_path}")
    print(f"Full results: {output_dir / 'coverage_risk_scan.json'}")


if __name__ == "__main__":
    main()
