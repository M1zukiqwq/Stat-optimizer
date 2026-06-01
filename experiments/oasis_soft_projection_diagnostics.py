#!/usr/bin/env python3
"""Case-level diagnostics for OASIS soft Stage-2 projection.

This additive script reuses the cached synthetic paper-suite data and the
existing OASIS checkpoint (no retraining, no rerun of full paper experiments).
It compares hard ISOMER/IPF projection against several soft-projection variants
and emits per-case features that explain *when* soft projection helps or hurts:

  - hard active suffix length (ISOMER's implicit stale-constraint filter);
  - per-observation conflict against the most recent trusted feedback;
  - count of consistent old observations (kept by Soft but
    discarded by a fixed recent window);
  - signed feedback residual split by observation age;
  - KL movement from the OASIS-noProj prior for each variant;
  - boundary L1 distance of each soft variant to hard projection;
  - future selectivity Q-error (geomean over generated probe predicates) per
    variant, plus deltas versus hard projection.

The goal is to test the hypothesis that a fixed recent window (the current
robust soft config) is too blunt: it throws away old-but-consistent feedback
that still improves future accuracy, while conflict-aware weighting can keep it.
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
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from json_histogram_parser import load_feedback_sample
from mlp_histogram_model_v2 import MlpHistogramModelV2
import modern_baselines as mb

import oasis_accuracy_smoke as smoke


def variant_boundaries(
    oasis: Sequence[float],
    observations: Sequence[dict],
    args: argparse.Namespace,
) -> Dict[str, List[float]]:
    """Build hard and soft-projection variants from the same OASIS prior."""
    full_obs = list(observations)
    recent_obs = (
        full_obs[-args.recent_window:]
        if 0 < args.recent_window < len(full_obs)
        else full_obs
    )
    variants: Dict[str, List[float]] = {}
    variants["hard"] = smoke.project_boundaries(
        oasis, full_obs, args.num_buckets, args.projection_iters, args.projection_tol
    )
    variants["soft_full"] = smoke.soft_project_boundaries(
        oasis, full_obs, args.num_buckets, args.soft_strength, 1.0, 1.0,
        args.soft_iters, args.soft_lr, args.soft_tol, False,
    )
    variants["soft_recent"] = smoke.soft_project_boundaries(
        oasis, recent_obs, args.num_buckets, args.soft_strength, 1.0, 1.0,
        args.soft_iters, args.soft_lr, args.soft_tol, False,
    )
    variants["soft_conflict"] = smoke.soft_project_boundaries(
        oasis, full_obs, args.num_buckets, args.soft_strength, 1.0, 1.0,
        args.soft_iters, args.soft_lr, args.soft_tol, False,
        conflict_aware=True,
        conflict_ref_window=args.recent_window,
        conflict_tau=args.conflict_tau,
        conflict_floor=args.conflict_floor,
    )
    return variants


def conflict_features(
    oasis: Sequence[float],
    observations: Sequence[dict],
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Deployment-visible conflict/consistency features for one case."""
    prior_min = float(oasis[0])
    prior_max = float(oasis[-1])
    prior_quantiles = list(oasis[1:-1])
    parsed = []
    for obs in observations:
        interval = mb._isomer_interval_from_observation(prior_min, prior_max, obs)
        if interval is not None:
            parsed.append(interval)

    feats = {
        "n_obs": float(len(observations)),
        "n_constraints": float(len(parsed)),
        "hard_active_suffix_len": 0.0,
        "n_old": 0.0,
        "n_old_consistent": 0.0,
        "n_old_conflicted": 0.0,
        "max_conflict": 0.0,
        "mean_conflict_old": 0.0,
        "signed_resid_old": 0.0,
        "signed_resid_recent": 0.0,
    }
    if not parsed:
        return feats

    suffix = mb._isomer_latest_feasible_suffix(
        prior_min, prior_max, prior_quantiles, parsed,
        max_iter=args.active_iters, tol=args.active_tol,
    )
    feats["hard_active_suffix_len"] = float(len(suffix))

    cell_boundaries, prior_probs, masks, targets = mb._isomer_build_partition(
        prior_min, prior_max, prior_quantiles, parsed
    )
    if not masks:
        return feats
    prior = np.maximum(prior_probs.astype(float), 1e-12)
    prior /= max(float(prior.sum()), 1e-12)
    mask_matrix = np.vstack([m.astype(float) for m in masks])
    targets = np.asarray(targets, dtype=float)

    ref_count = min(max(1, args.recent_window), len(masks))
    ref_probs, _, _ = mb._isomer_fit_active_set(
        prior, list(masks[-ref_count:]), targets[-ref_count:],
        max_iter=args.active_iters, tol=args.active_tol,
    )
    ref_estimates = mask_matrix @ ref_probs
    conflict = np.abs(ref_estimates - targets)

    prior_estimates = mask_matrix @ prior
    signed = prior_estimates - targets  # positive = prior over-estimates target

    n = len(masks)
    old_idx = list(range(0, max(0, n - ref_count)))
    recent_idx = list(range(max(0, n - ref_count), n))
    feats["n_old"] = float(len(old_idx))
    feats["max_conflict"] = float(np.max(conflict)) if n else 0.0
    if old_idx:
        old_conf = conflict[old_idx]
        feats["mean_conflict_old"] = float(np.mean(old_conf))
        feats["n_old_consistent"] = float(np.sum(old_conf <= args.conflict_thr))
        feats["n_old_conflicted"] = float(np.sum(old_conf > args.conflict_thr))
        feats["signed_resid_old"] = float(np.mean(signed[old_idx]))
    if recent_idx:
        feats["signed_resid_recent"] = float(np.mean(signed[recent_idx]))
    return feats


def kl_to_prior(boundaries: Sequence[float], oasis: Sequence[float], grid: int = 200) -> float:
    """KL(p_variant || p_prior) approximated on a uniform value grid."""
    lo = float(oasis[0])
    hi = float(oasis[-1])
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, grid + 1)
    levels_v = smoke.cdf_levels(boundaries)
    levels_p = smoke.cdf_levels(oasis)
    cdf_v = np.array([smoke.evaluate_piecewise_cdf(list(boundaries), levels_v, float(e)) for e in edges])
    cdf_p = np.array([smoke.evaluate_piecewise_cdf(list(oasis), levels_p, float(e)) for e in edges])
    p = np.maximum(np.diff(cdf_v), 1e-9)
    q = np.maximum(np.diff(cdf_p), 1e-9)
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def boundary_l1(a: Sequence[float], b: Sequence[float]) -> float:
    inner_a = list(a)[1:-1]
    inner_b = list(b)[1:-1]
    span = max(float(a[-1]) - float(a[0]), 1e-12)
    return sum(abs(x - y) for x, y in zip(inner_a, inner_b)) / max(len(inner_a), 1) / span


def future_qerror(boundaries: Sequence[float], predicates: Sequence[dict], fresh: Sequence[float]) -> float:
    errs = []
    for pred in predicates:
        truth = smoke.estimate_selectivity(fresh, pred)
        est = smoke.estimate_selectivity(boundaries, pred)
        errs.append(smoke.qerr(est, truth))
    return smoke.geomean(errs)


def run(args: argparse.Namespace) -> None:
    model = MlpHistogramModelV2.load(str(args.model_path))
    paths = smoke.sample_paths(args.data_root, args.q_values, args.max_cases_per_q)
    if not paths:
        raise FileNotFoundError(f"No test samples found under {args.data_root}")

    variant_names = ["hard", "soft_full", "soft_recent", "soft_conflict"]
    rows: List[dict] = []

    for sample_index, (q_mods, path) in enumerate(paths):
        sample = load_feedback_sample(str(path))
        if sample.corrected_quantile_values is None:
            continue
        observations = smoke.observations_to_dicts(sample, args.max_observations)
        oasis = smoke.oasis_boundaries(sample, model, args.max_observations)
        fresh = smoke.boundaries_from_quantiles(
            sample.corrected_quantile_values or sample.prior.quantile_values
        )

        variants = variant_boundaries(oasis, observations, args)
        feats = conflict_features(oasis, observations, args)

        rng = random.Random(args.seed + q_mods * 100_000 + sample_index)
        predicates = smoke.generate_predicates(fresh, rng, args.predicates_per_case, args.min_true_selectivity)

        row = {"q_mods": q_mods, "case_id": path.stem}
        row.update(feats)
        qerrs = {}
        for name in variant_names:
            b = variants[name]
            qerrs[name] = future_qerror(b, predicates, fresh)
            row[f"qerr_{name}"] = qerrs[name]
            row[f"kl_{name}"] = kl_to_prior(b, oasis)
            row[f"l1_to_hard_{name}"] = boundary_l1(b, variants["hard"])
            mean_resid, _ = smoke.feedback_residuals(b, observations)
            row[f"feedresid_{name}"] = mean_resid
        # Deltas vs hard (positive = variant better than hard).
        for name in ["soft_full", "soft_recent", "soft_conflict"]:
            row[f"dqerr_{name}_vs_hard"] = qerrs["hard"] - qerrs[name]
        row["dqerr_conflict_vs_recent"] = qerrs["soft_recent"] - qerrs["soft_conflict"]
        rows.append(row)

    write_outputs(args.output_dir, rows, variant_names, args)


def write_outputs(output_dir: Path, rows: List[dict], variant_names: List[str], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / "case_diagnostics.csv").open("w", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def gm(key: str) -> float:
        return smoke.geomean([r[key] for r in rows]) if rows else float("nan")

    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / max(len(rows), 1)

    summary = {
        "n_cases": len(rows),
        "future_qerror_gm": {name: gm(f"qerr_{name}") for name in variant_names},
        "kl_to_prior_mean": {name: mean(f"kl_{name}") for name in variant_names},
        "l1_to_hard_mean": {name: mean(f"l1_to_hard_{name}") for name in variant_names},
        "feedresid_mean": {name: mean(f"feedresid_{name}") for name in variant_names},
        "hard_active_suffix_len_mean": mean("hard_active_suffix_len"),
        "n_obs_mean": mean("n_obs"),
        "n_old_mean": mean("n_old"),
        "n_old_consistent_mean": mean("n_old_consistent"),
        "n_old_conflicted_mean": mean("n_old_conflicted"),
        "mean_conflict_old_mean": mean("mean_conflict_old"),
    }
    # How often does conflict-aware beat the fixed recent window, and is it
    # explained by the presence of consistent old observations?
    if rows:
        conflict_wins = [r for r in rows if r["dqerr_conflict_vs_recent"] > 1e-6]
        summary["conflict_beats_recent_frac"] = len(conflict_wins) / len(rows)
        summary["conflict_vs_recent_mean_dqerr"] = mean("dqerr_conflict_vs_recent")
        with_consistent_old = [r for r in rows if r["n_old_consistent"] >= 1.0]
        summary["cases_with_consistent_old_frac"] = len(with_consistent_old) / len(rows)
        if with_consistent_old:
            summary["conflict_vs_recent_dqerr_when_consistent_old"] = (
                sum(r["dqerr_conflict_vs_recent"] for r in with_consistent_old) / len(with_consistent_old)
            )
        without_consistent_old = [r for r in rows if r["n_old_consistent"] < 1.0]
        if without_consistent_old:
            summary["conflict_vs_recent_dqerr_when_no_consistent_old"] = (
                sum(r["dqerr_conflict_vs_recent"] for r in without_consistent_old) / len(without_consistent_old)
            )

    summary["config"] = {
        "recent_window": args.recent_window,
        "conflict_tau": args.conflict_tau,
        "conflict_floor": args.conflict_floor,
        "conflict_thr": args.conflict_thr,
        "soft_strength": args.soft_strength,
        "soft_iters": args.soft_iters,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = [
        "OASIS soft projection diagnostics",
        "=" * 44,
        f"Cases: {summary['n_cases']}",
        "",
        "Future selectivity Q-error (geomean), lower is better:",
    ]
    for name in variant_names:
        lines.append(f"  {name:<14s} {summary['future_qerror_gm'][name]:.4f}")
    lines += [
        "",
        f"Mean obs/case: {summary['n_obs_mean']:.2f}; hard active suffix len: {summary['hard_active_suffix_len_mean']:.2f}",
        f"Mean old obs: {summary['n_old_mean']:.2f}; old-consistent: {summary['n_old_consistent_mean']:.2f}; old-conflicted: {summary['n_old_conflicted_mean']:.2f}",
    ]
    if rows:
        lines += [
            f"conflict-aware beats fixed recent window on future Q-error in {summary['conflict_beats_recent_frac']*100:.1f}% of cases (mean delta {summary['conflict_vs_recent_mean_dqerr']:+.4f})",
            f"cases with >=1 consistent old obs: {summary['cases_with_consistent_old_frac']*100:.1f}%",
        ]
        if "conflict_vs_recent_dqerr_when_consistent_old" in summary:
            lines.append(
                f"  mean conflict-vs-recent delta | consistent-old present: {summary['conflict_vs_recent_dqerr_when_consistent_old']:+.4f}"
            )
        if "conflict_vs_recent_dqerr_when_no_consistent_old" in summary:
            lines.append(
                f"  mean conflict-vs-recent delta | no consistent-old:     {summary['conflict_vs_recent_dqerr_when_no_consistent_old']:+.4f}"
            )
    text = "\n".join(lines)
    (output_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


def parse_args() -> argparse.Namespace:
    root = _REPO_DIR / "experiments" / "results" / "synthetic_paper_suite_rerun_20260529"
    parser = argparse.ArgumentParser(description="Case-level diagnostics for OASIS soft projection")
    parser.add_argument("--data-root", type=Path, default=root / "compound_data")
    parser.add_argument("--model-path", type=Path, default=root / "models" / "oasis_k16.json")
    parser.add_argument("--output-dir", type=Path,
                        default=_REPO_DIR / "experiments" / "results" / "oasis_soft_diagnostics_20260531")
    parser.add_argument("--q-values", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--max-cases-per-q", type=int, default=48)
    parser.add_argument("--predicates-per-case", type=int, default=32)
    parser.add_argument("--num-buckets", type=int, default=10)
    parser.add_argument("--max-observations", type=int, default=16)
    parser.add_argument("--projection-iters", type=int, default=200)
    parser.add_argument("--projection-tol", type=float, default=1e-4)
    parser.add_argument("--soft-strength", type=float, default=30.0)
    parser.add_argument("--soft-iters", type=int, default=500)
    parser.add_argument("--soft-lr", type=float, default=0.05)
    parser.add_argument("--soft-tol", type=float, default=1e-9)
    parser.add_argument("--recent-window", type=int, default=8)
    parser.add_argument("--conflict-tau", type=float, default=0.05)
    parser.add_argument("--conflict-floor", type=float, default=0.0)
    parser.add_argument("--conflict-thr", type=float, default=0.05,
                        help="Residual above which an old observation counts as conflicted.")
    parser.add_argument("--active-iters", type=int, default=50)
    parser.add_argument("--active-tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--min-true-selectivity", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
