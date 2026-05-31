#!/usr/bin/env python3
"""Tiny OASIS accuracy/robustness smoke test.

This additive smoke reuses cached synthetic paper-suite data and an existing
OASIS checkpoint. It does not retrain models or rerun full paper experiments.

Candidates tested here: soft-constrained OASIS projection and aggressive
residual-routed calibration. The soft projection directly replaces hard Stage-2
feedback matching with a KL-to-prior plus weighted feedback-residual objective.
The aggressive calibration keeps a wider candidate pool selected only by
feedback residuals.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_PIPELINE_DIR = _REPO_DIR / "cdf_kll_ml_pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from histogram_math import clamp01, evaluate_piecewise_cdf, inverse_piecewise_cdf, project_monotonic
from json_histogram_parser import load_feedback_sample
from mlp_histogram_model_v2 import MlpHistogramModelV2
from modern_baselines import correct_isomer, correct_soft_isomer, correct_band_isomer
from tensorizer import tensorize_sample


METHOD_ORDER = [
    "stale",
    "isomer",
    "oasis_no_proj",
    "oasis_full",
    "oasis_soft_projection",
    "oasis_band_projection",
    "oasis_damped_guarded",
    "oasis_aggressive_hybrid",
    "oasis_recency_hybrid",
    "residual_hybrid",
    "calibrated_hybrid",
    "fresh",
]


def boundaries_from_quantiles(quantiles: Sequence[float]) -> List[float]:
    values = [clamp01(float(value)) for value in quantiles]
    return [0.0] + project_monotonic(values) + [1.0]


def cdf_levels(boundaries: Sequence[float]) -> List[float]:
    buckets = max(len(boundaries) - 1, 1)
    return [index / buckets for index in range(len(boundaries))]


def estimate_selectivity(boundaries: Sequence[float], predicate: Dict[str, object]) -> float:
    levels = cdf_levels(boundaries)
    pred_type = str(predicate["predicate_type"])
    value = float(predicate["value"])

    if pred_type in {"<", "<="}:
        return max(evaluate_piecewise_cdf(boundaries, levels, value), 1e-9)
    if pred_type in {">", ">="}:
        return max(1.0 - evaluate_piecewise_cdf(boundaries, levels, value), 1e-9)
    if pred_type == "BETWEEN":
        upper = float(predicate.get("value_upper", value))
        lo, hi = sorted((value, upper))
        return max(
            evaluate_piecewise_cdf(boundaries, levels, hi)
            - evaluate_piecewise_cdf(boundaries, levels, lo),
            1e-9,
        )

    width = 0.01
    return max(
        evaluate_piecewise_cdf(boundaries, levels, min(1.0, value + width))
        - evaluate_piecewise_cdf(boundaries, levels, max(0.0, value - width)),
        1e-9,
    )


def qerr(estimate: float, truth: float) -> float:
    estimate = max(float(estimate), 1e-9)
    truth = max(float(truth), 1e-9)
    return max(estimate / truth, truth / estimate)


def geomean(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    return math.exp(sum(math.log(max(float(value), 1e-12)) for value in values) / len(values))


def pct_improvement(base: float, value: float) -> float:
    return (base - value) / max(base, 1e-12) * 100.0


def pct_delta(base: float, value: float) -> float:
    return (value - base) / max(base, 1e-12) * 100.0


def observations_to_dicts(sample, max_observations: int) -> List[dict]:
    result: List[dict] = []
    for obs in sample.observations[-max_observations:]:
        item = {
            "predicate_type": obs.predicate_type,
            "value": obs.value,
            "estimated_sel": obs.estimated_selectivity,
            "actual_sel": obs.actual_selectivity,
        }
        if obs.value_upper is not None:
            item["value_upper"] = obs.value_upper
        result.append(item)
    return result


def feedback_residuals(boundaries: Sequence[float], observations: Sequence[dict]) -> Tuple[float, float]:
    if not observations:
        return float("inf"), float("inf")
    errors = feedback_errors(boundaries, observations)
    return sum(errors) / len(errors), max(errors)


def feedback_errors(boundaries: Sequence[float], observations: Sequence[dict]) -> List[float]:
    return [
        abs(estimate_selectivity(boundaries, obs) - float(obs["actual_sel"]))
        for obs in observations
    ]


def weighted_feedback_residuals(
    boundaries: Sequence[float],
    observations: Sequence[dict],
    recency_decay: float,
) -> Tuple[float, float]:
    if not observations:
        return float("inf"), float("inf")
    errors = feedback_errors(boundaries, observations)
    decay = max(min(float(recency_decay), 1.0), 1e-6)
    # observations are oldest -> newest; newest gets weight 1.0.
    weights = [decay ** (len(errors) - index - 1) for index in range(len(errors))]
    weight_sum = max(sum(weights), 1e-12)
    mean = sum(weight * err for weight, err in zip(weights, errors)) / weight_sum
    return mean, max(errors)


def quantile_mae(boundaries: Sequence[float], fresh_boundaries: Sequence[float]) -> float:
    pred = list(boundaries)[1:-1]
    truth = list(fresh_boundaries)[1:-1]
    return sum(abs(a - b) for a, b in zip(pred, truth)) / max(len(truth), 1)


def oasis_boundaries(sample, model: MlpHistogramModelV2, max_observations: int) -> List[float]:
    record = tensorize_sample(sample, max_observations=max_observations, teacher_fn=None, use_time_decay=False)
    prediction = model.predict([record.feature_tensor])[0]
    return boundaries_from_quantiles(prediction)


def project_boundaries(
    prior_boundaries: Sequence[float],
    observations: Sequence[dict],
    num_buckets: int,
    max_iter: int,
    tol: float,
) -> List[float]:
    try:
        corrected = correct_isomer(
            float(prior_boundaries[0]),
            float(prior_boundaries[-1]),
            list(prior_boundaries[1:-1]),
            list(observations),
            num_buckets=num_buckets,
            max_iter=max_iter,
            tol=tol,
        )
        return boundaries_from_quantiles(corrected)
    except Exception:
        return list(prior_boundaries)


def soft_project_boundaries(
    prior_boundaries: Sequence[float],
    observations: Sequence[dict],
    num_buckets: int,
    constraint_strength: float,
    recency_decay: float,
    target_blend: float,
    max_iter: int,
    learning_rate: float,
    tol: float,
    active_set: bool,
    conflict_aware: bool = False,
    conflict_ref_window: int = 8,
    conflict_tau: float = 0.05,
    conflict_floor: float = 0.0,
) -> List[float]:
    try:
        corrected = correct_soft_isomer(
            float(prior_boundaries[0]),
            float(prior_boundaries[-1]),
            list(prior_boundaries[1:-1]),
            list(observations),
            num_buckets=num_buckets,
            constraint_strength=constraint_strength,
            recency_decay=recency_decay,
            target_blend=target_blend,
            max_iter=max_iter,
            learning_rate=learning_rate,
            tol=tol,
            active_set=active_set,
            conflict_aware=conflict_aware,
            conflict_ref_window=conflict_ref_window,
            conflict_tau=conflict_tau,
            conflict_floor=conflict_floor,
        )
        return boundaries_from_quantiles(corrected)
    except Exception:
        return list(prior_boundaries)


def band_project_boundaries(
    prior_boundaries: Sequence[float],
    observations: Sequence[dict],
    num_buckets: int,
    band_kappa: float,
    band_floor: float,
    max_iter: int,
    tol: float,
    conflict_aware: bool = False,
    conflict_ref_window: int = 8,
    conflict_tau: float = 0.05,
    conflict_drop: float = 0.10,
) -> List[float]:
    try:
        corrected = correct_band_isomer(
            float(prior_boundaries[0]),
            float(prior_boundaries[-1]),
            list(prior_boundaries[1:-1]),
            list(observations),
            num_buckets=num_buckets,
            band_kappa=band_kappa,
            band_floor=band_floor,
            max_iter=max_iter,
            tol=tol,
            conflict_aware=conflict_aware,
            conflict_ref_window=conflict_ref_window,
            conflict_tau=conflict_tau,
            conflict_drop=conflict_drop,
        )
        return boundaries_from_quantiles(corrected)
    except Exception:
        return list(prior_boundaries)


def damp_observations(
    observations: Sequence[dict],
    anchor_boundaries: Sequence[float],
    alpha: float,
) -> List[dict]:
    damped: List[dict] = []
    for obs in observations:
        anchored = estimate_selectivity(anchor_boundaries, obs)
        target = alpha * float(obs["actual_sel"]) + (1.0 - alpha) * anchored
        item = dict(obs)
        item["actual_sel"] = max(1e-6, min(1.0 - 1e-6, target))
        damped.append(item)
    return damped


def choose_residual_hybrid(
    method_boundaries: Dict[str, List[float]],
    observations: Sequence[dict],
    candidates: Optional[Sequence[str]] = None,
) -> Tuple[str, List[float]]:
    if candidates is None:
        candidates = ["stale", "isomer", "oasis_no_proj", "oasis_full"]
    candidates = [m for m in candidates if m in method_boundaries]
    scores = {method: feedback_residuals(method_boundaries[method], observations)[0] for method in candidates}
    choice = min(candidates, key=lambda method: scores[method])
    return choice, method_boundaries[choice]


def candidate_label(prefix: str, value) -> str:
    if isinstance(value, float):
        return f"{prefix}{int(round(value * 100)):02d}"
    return f"{prefix}{value}"


def build_aggressive_candidates(
    observations: Sequence[dict],
    stale: Sequence[float],
    isomer: Sequence[float],
    oasis: Sequence[float],
    oasis_full: Sequence[float],
    guarded: Sequence[float],
    args: argparse.Namespace,
) -> Dict[str, List[float]]:
    candidates = {
        "stale": list(stale),
        "isomer": list(isomer),
        "oasis_no_proj": list(oasis),
        "oasis_full": list(oasis_full),
        "oasis_damped_guarded": list(guarded),
    }

    for alpha in args.damping_grid:
        damped_observations = damp_observations(observations, oasis, alpha=alpha)
        label = candidate_label("damped_a", alpha)
        candidates[label] = project_boundaries(
            oasis,
            damped_observations,
            args.num_buckets,
            args.projection_iters,
            args.projection_tol,
        )

    for window in args.recent_projection_windows:
        if 0 < window < len(observations):
            recent = list(observations[-window:])
            candidates[candidate_label("oasis_recent_k", window)] = project_boundaries(
                oasis,
                recent,
                args.num_buckets,
                args.projection_iters,
                args.projection_tol,
            )
            candidates[candidate_label("isomer_recent_k", window)] = project_boundaries(
                stale,
                recent,
                args.num_buckets,
                args.projection_iters,
                args.projection_tol,
            )
    return candidates


def choose_candidate_by_residual(
    candidates: Dict[str, List[float]],
    observations: Sequence[dict],
    recency_decay: Optional[float] = None,
) -> Tuple[str, List[float]]:
    if recency_decay is None:
        scores = {
            name: feedback_residuals(bounds, observations)
            for name, bounds in candidates.items()
        }
    else:
        scores = {
            name: weighted_feedback_residuals(bounds, observations, recency_decay)
            for name, bounds in candidates.items()
        }
    choice = min(scores, key=lambda name: (scores[name][0], scores[name][1]))
    return choice, candidates[choice]


def build_method_boundaries(
    sample,
    model: MlpHistogramModelV2,
    args: argparse.Namespace,
) -> Tuple[Dict[str, List[float]], Dict[str, object]]:
    observations = observations_to_dicts(sample, args.max_observations)
    stale = boundaries_from_quantiles(sample.prior.quantile_values)
    fresh = boundaries_from_quantiles(sample.corrected_quantile_values or sample.prior.quantile_values)
    oasis = oasis_boundaries(sample, model, args.max_observations)
    isomer = project_boundaries(stale, observations, args.num_buckets, args.projection_iters, args.projection_tol)
    oasis_full = project_boundaries(oasis, observations, args.num_buckets, args.projection_iters, args.projection_tol)
    oasis_soft = soft_project_boundaries(
        oasis,
        observations[-args.soft_projection_window:]
        if 0 < args.soft_projection_window < len(observations)
        else observations,
        args.num_buckets,
        args.soft_projection_strength,
        args.soft_projection_recency_decay,
        args.soft_projection_target_blend,
        args.soft_projection_iters,
        args.soft_projection_lr,
        args.soft_projection_tol,
        args.soft_projection_active_set,
        args.soft_projection_conflict_aware,
        args.soft_projection_conflict_ref_window,
        args.soft_projection_conflict_tau,
        args.soft_projection_conflict_floor,
    )

    oasis_band = band_project_boundaries(
        oasis,
        observations,
        args.num_buckets,
        args.band_kappa,
        args.band_floor,
        args.projection_iters,
        args.projection_tol,
        args.band_conflict_aware,
        args.band_conflict_ref_window,
        args.band_conflict_tau,
        args.band_conflict_drop,
    )

    damped_observations = damp_observations(observations, oasis, alpha=args.damping_alpha)
    damped = project_boundaries(oasis, damped_observations, args.num_buckets, args.projection_iters, args.projection_tol)

    full_mean, full_max = feedback_residuals(oasis_full, observations)
    damped_mean, damped_max = feedback_residuals(damped, observations)
    accept_damped = (
        damped_mean <= full_mean * (1.0 + args.residual_guard_frac)
        and damped_max <= full_max * (1.0 + args.max_residual_guard_frac)
    )
    guarded = damped if accept_damped else oasis_full

    method_boundaries = {
        "stale": stale,
        "isomer": isomer,
        "oasis_no_proj": oasis,
        "oasis_full": oasis_full,
        "oasis_soft_projection": oasis_soft,
        "oasis_band_projection": oasis_band,
        "oasis_damped_guarded": guarded,
        "fresh": fresh,
    }
    aggressive_candidates = build_aggressive_candidates(
        observations=observations,
        stale=stale,
        isomer=isomer,
        oasis=oasis,
        oasis_full=oasis_full,
        guarded=guarded,
        args=args,
    )
    aggressive_choice, aggressive = choose_candidate_by_residual(
        aggressive_candidates,
        observations,
        recency_decay=None,
    )
    recency_choice, recency = choose_candidate_by_residual(
        aggressive_candidates,
        observations,
        recency_decay=args.recency_residual_decay,
    )
    method_boundaries["oasis_aggressive_hybrid"] = aggressive
    method_boundaries["oasis_recency_hybrid"] = recency
    hybrid_choice, hybrid = choose_residual_hybrid(method_boundaries, observations)
    method_boundaries["residual_hybrid"] = hybrid
    calibrated_choice, calibrated = choose_residual_hybrid(
        method_boundaries, observations,
        candidates=["stale", "isomer", "oasis_no_proj", "oasis_full",
                    "oasis_soft_projection", "oasis_band_projection"],
    )
    method_boundaries["calibrated_hybrid"] = calibrated

    metadata = {
        "accepted_damped": accept_damped,
        "aggressive_hybrid_choice": aggressive_choice,
        "recency_hybrid_choice": recency_choice,
        "residual_hybrid_choice": hybrid_choice,
        "calibrated_hybrid_choice": calibrated_choice,
        "full_residual_mean": full_mean,
        "damped_residual_mean": damped_mean,
        "full_residual_max": full_max,
        "damped_residual_max": damped_max,
    }
    return method_boundaries, metadata


def sample_paths(data_root: Path, q_values: Sequence[int], max_cases_per_q: int) -> List[Tuple[int, Path]]:
    paths: List[Tuple[int, Path]] = []
    for q_mods in q_values:
        q_dir = data_root / f"test_q{q_mods}"
        q_paths = sorted(q_dir.glob("*.json"))
        if max_cases_per_q > 0:
            q_paths = q_paths[:max_cases_per_q]
        paths.extend((q_mods, path) for path in q_paths)
    return paths


def generate_predicates(
    fresh_boundaries: Sequence[float],
    rng: random.Random,
    count: int,
    min_true_selectivity: float,
) -> List[dict]:
    levels = cdf_levels(fresh_boundaries)
    predicates: List[dict] = []
    attempts = 0
    while len(predicates) < count and attempts < count * 50:
        attempts += 1
        pred_type = rng.choices(["<=", ">=", "BETWEEN", "="], weights=[0.30, 0.30, 0.32, 0.08], k=1)[0]
        if pred_type == "BETWEEN":
            width = 10 ** rng.uniform(math.log10(0.01), math.log10(0.45))
            lo_p = rng.uniform(0.02, max(0.03, 0.98 - width))
            hi_p = min(0.99, lo_p + width)
            pred = {
                "predicate_type": "BETWEEN",
                "value": inverse_piecewise_cdf(fresh_boundaries, levels, lo_p),
                "value_upper": inverse_piecewise_cdf(fresh_boundaries, levels, hi_p),
            }
        elif pred_type == "<=":
            pred = {
                "predicate_type": "<=",
                "value": inverse_piecewise_cdf(fresh_boundaries, levels, rng.uniform(0.005, 0.95)),
                "value_upper": None,
            }
        elif pred_type == ">=":
            pred = {
                "predicate_type": ">=",
                "value": inverse_piecewise_cdf(fresh_boundaries, levels, rng.uniform(0.05, 0.995)),
                "value_upper": None,
            }
        else:
            pred = {
                "predicate_type": "=",
                "value": inverse_piecewise_cdf(fresh_boundaries, levels, rng.uniform(0.02, 0.98)),
                "value_upper": None,
            }
        if estimate_selectivity(fresh_boundaries, pred) >= min_true_selectivity:
            predicates.append(pred)
    return predicates


def scan_cost(choice: str, selectivity: float, table_rows: int, args: argparse.Namespace) -> float:
    rows = max(selectivity, 1e-9) * table_rows
    if choice == "index":
        return args.index_startup_cost + rows * args.index_tuple_cost
    return table_rows * args.seq_tuple_cost


def choose_scan(selectivity: float, table_rows: int, args: argparse.Namespace) -> str:
    return "index" if scan_cost("index", selectivity, table_rows, args) < scan_cost("seq", selectivity, table_rows, args) else "seq"


def join_cost(choice: str, selectivity: float, table_rows: int, args: argparse.Namespace) -> float:
    rows = max(selectivity, 1e-9) * table_rows
    if choice == "nested_loop":
        return scan_cost("index", selectivity, table_rows, args) + rows * args.nl_lookup_cost
    return (
        scan_cost("seq", selectivity, table_rows, args)
        + args.dim_rows * args.hash_build_tuple_cost
        + rows * args.hash_probe_tuple_cost
    )


def choose_join(selectivity: float, table_rows: int, args: argparse.Namespace) -> str:
    nested = join_cost("nested_loop", selectivity, table_rows, args)
    hashed = join_cost("hash_join", selectivity, table_rows, args)
    return "nested_loop" if nested < hashed else "hash_join"


def regrets(estimated: float, truth: float, table_rows: int, args: argparse.Namespace) -> Tuple[str, str, float, str, str, float]:
    scan_choice = choose_scan(estimated, table_rows, args)
    scan_optimal = choose_scan(truth, table_rows, args)
    scan_regret = scan_cost(scan_choice, truth, table_rows, args) / max(scan_cost(scan_optimal, truth, table_rows, args), 1e-12)
    join_choice = choose_join(estimated, table_rows, args)
    join_optimal = choose_join(truth, table_rows, args)
    join_regret = join_cost(join_choice, truth, table_rows, args) / max(join_cost(join_optimal, truth, table_rows, args), 1e-12)
    return scan_choice, scan_optimal, scan_regret, join_choice, join_optimal, join_regret


def aggregate(case_rows: Sequence[dict], predicate_rows: Sequence[dict], risk_threshold: float) -> List[dict]:
    case_by_method: Dict[str, List[dict]] = defaultdict(list)
    pred_by_method: Dict[str, List[dict]] = defaultdict(list)
    for row in case_rows:
        case_by_method[row["method"]].append(row)
    for row in predicate_rows:
        pred_by_method[row["method"]].append(row)

    stale_by_key = {
        (row["q_mods"], row["case_id"], row["predicate_id"]): row
        for row in predicate_rows
        if row["method"] == "stale"
    }
    fresh_by_key = {
        (row["q_mods"], row["case_id"], row["predicate_id"]): row
        for row in predicate_rows
        if row["method"] == "fresh"
    }

    summary: List[dict] = []
    for method in METHOD_ORDER:
        crows = case_by_method[method]
        prows = pred_by_method[method]
        risky = 0
        resolved = 0
        losses = 0
        fresh_join_match = 0
        for row in prows:
            key = (row["q_mods"], row["case_id"], row["predicate_id"])
            stale = stale_by_key[key]
            fresh = fresh_by_key[key]
            if stale["join_regret"] >= risk_threshold:
                risky += 1
                if row["join_regret"] < risk_threshold:
                    resolved += 1
            elif row["join_regret"] >= risk_threshold:
                losses += 1
            if row["join_choice"] == fresh["join_choice"]:
                fresh_join_match += 1

        summary.append({
            "method": method,
            "n_cases": len(crows),
            "n_predicates": len(prows),
            "selectivity_qerr_gm": geomean([row["selectivity_qerr"] for row in prows]),
            "selectivity_mae_mean": sum(row["selectivity_abs_error"] for row in prows) / max(len(prows), 1),
            "quantile_mae_mean": sum(row["quantile_mae"] for row in crows) / max(len(crows), 1),
            "feedback_residual_mean": sum(row["feedback_residual_mean"] for row in crows) / max(len(crows), 1),
            "feedback_residual_max": max((row["feedback_residual_max"] for row in crows), default=0.0),
            "scan_regret_gm": geomean([row["scan_regret"] for row in prows]),
            "join_regret_gm": geomean([row["join_regret"] for row in prows]),
            "join_optimal_match_frac": sum(row["join_choice"] == row["join_optimal"] for row in prows) / max(len(prows), 1),
            "join_fresh_match_frac": fresh_join_match / max(len(prows), 1),
            "risk_resolved_frac": resolved / max(risky, 1),
            "new_risk_loss_frac": losses / max(len(prows), 1),
        })
    return summary


def build_verdict(summary: Sequence[dict], choices: Dict[str, int], args: argparse.Namespace) -> dict:
    by_method = {row["method"]: row for row in summary}
    baseline = by_method["oasis_full"]
    candidate = by_method[args.verdict_candidate]
    verdict = {
        "candidate": args.verdict_candidate,
        "baseline": "oasis_full",
        "damped_accept_count": choices.get("accepted_damped", 0),
        "damped_reject_count": choices.get("rejected_damped", 0),
        "criteria": {
            "min_qerror_improvement_pct": args.min_qerror_improvement_pct,
            "max_feedback_residual_regression_pct": args.max_feedback_residual_regression_pct,
            "max_structural_regression_pct": args.max_structural_regression_pct,
            "max_join_regret_regression_pct": args.max_join_regret_regression_pct,
            "max_new_risk_loss_delta": args.max_new_risk_loss_delta,
        },
        "deltas": {
            "qerror_improvement_pct": pct_improvement(baseline["selectivity_qerr_gm"], candidate["selectivity_qerr_gm"]),
            "feedback_residual_delta_pct": pct_delta(baseline["feedback_residual_mean"], candidate["feedback_residual_mean"]),
            "selectivity_mae_delta_pct": pct_delta(baseline["selectivity_mae_mean"], candidate["selectivity_mae_mean"]),
            "quantile_mae_delta_pct": pct_delta(baseline["quantile_mae_mean"], candidate["quantile_mae_mean"]),
            "join_regret_delta_pct": pct_delta(baseline["join_regret_gm"], candidate["join_regret_gm"]),
            "new_risk_loss_delta": candidate["new_risk_loss_frac"] - baseline["new_risk_loss_frac"],
        },
    }
    deltas = verdict["deltas"]
    verdict["pass"] = (
        deltas["qerror_improvement_pct"] >= args.min_qerror_improvement_pct
        and deltas["feedback_residual_delta_pct"] <= args.max_feedback_residual_regression_pct
        and deltas["selectivity_mae_delta_pct"] <= args.max_structural_regression_pct
        and deltas["quantile_mae_delta_pct"] <= args.max_structural_regression_pct
        and deltas["join_regret_delta_pct"] <= args.max_join_regret_regression_pct
        and deltas["new_risk_loss_delta"] <= args.max_new_risk_loss_delta
    )
    return verdict


def write_outputs(
    output_dir: Path,
    case_rows: Sequence[dict],
    predicate_rows: Sequence[dict],
    summary: Sequence[dict],
    choices: Dict[str, int],
    verdict: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("case_rows", case_rows), ("predicate_rows", predicate_rows), ("summary", summary)]:
        with (output_dir / f"{name}.json").open("w", encoding="utf-8") as handle:
            json.dump(list(rows), handle, indent=2)
        if rows:
            with (output_dir / f"{name}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
    with (output_dir / "choices.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(choices), handle, indent=2)
    with (output_dir / "verdict.json").open("w", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=2)

    stale = next(row for row in summary if row["method"] == "stale")
    lines = [
        "OASIS accuracy/robustness smoke",
        "=" * 44,
        f"Candidate: {verdict['candidate']} against full OASIS baseline.",
        "",
        "Method                   SelQE  QEImp  SelMAE  QuantMAE  FeedMean  FeedMax  JoinReg  JoinOpt  NewRisk",
        "-" * 108,
    ]
    for row in summary:
        lines.append(
            f"{row['method']:<24s} {row['selectivity_qerr_gm']:5.3f}  "
            f"{pct_improvement(stale['selectivity_qerr_gm'], row['selectivity_qerr_gm']):5.1f}%  "
            f"{row['selectivity_mae_mean']:6.4f}  {row['quantile_mae_mean']:8.5f}  "
            f"{row['feedback_residual_mean']:8.5f}  {row['feedback_residual_max']:7.5f}  "
            f"{row['join_regret_gm']:7.4f}  {row['join_optimal_match_frac'] * 100:7.1f}%  "
            f"{row['new_risk_loss_frac'] * 100:6.2f}%"
        )
    lines.extend([
        "",
        f"Damped accepted: {choices.get('accepted_damped', 0)}, rejected: {choices.get('rejected_damped', 0)}",
        f"Aggressive-hybrid choices: {dict((key, value) for key, value in choices.items() if key.startswith('aggressive_'))}",
        f"Recency-hybrid choices: {dict((key, value) for key, value in choices.items() if key.startswith('recency_'))}",
        f"Residual-hybrid choices: {dict((key, value) for key, value in choices.items() if key.startswith('hybrid_'))}",
        "",
        f"Verdict pass={verdict['pass']}: {verdict['deltas']}",
    ])
    text = "\n".join(lines)
    (output_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


def run_smoke(args: argparse.Namespace) -> None:
    model = MlpHistogramModelV2.load(str(args.model_path))
    paths = sample_paths(args.data_root, args.q_values, args.max_cases_per_q)
    if not paths:
        raise FileNotFoundError(f"No test samples found under {args.data_root}")

    case_rows: List[dict] = []
    predicate_rows: List[dict] = []
    choices: Counter = Counter()

    for sample_index, (q_mods, path) in enumerate(paths):
        sample = load_feedback_sample(str(path))
        if sample.corrected_quantile_values is None:
            continue
        observations = observations_to_dicts(sample, args.max_observations)
        method_boundaries, metadata = build_method_boundaries(sample, model, args)
        if metadata["accepted_damped"]:
            choices["accepted_damped"] += 1
        else:
            choices["rejected_damped"] += 1
        choices[f"aggressive_{metadata['aggressive_hybrid_choice']}"] += 1
        choices[f"recency_{metadata['recency_hybrid_choice']}"] += 1
        choices[f"hybrid_{metadata['residual_hybrid_choice']}"] += 1
        choices[f"calibrated_{metadata['calibrated_hybrid_choice']}"] += 1

        fresh = method_boundaries["fresh"]
        rng = random.Random(args.seed + q_mods * 100_000 + sample_index)
        table_rows = int(10 ** rng.uniform(math.log10(args.min_table_rows), math.log10(args.max_table_rows)))
        predicates = generate_predicates(fresh, rng, args.predicates_per_case, args.min_true_selectivity)

        for method, boundaries in method_boundaries.items():
            residual_mean, residual_max = feedback_residuals(boundaries, observations)
            case_rows.append({
                "q_mods": q_mods,
                "case_id": path.stem,
                "method": method,
                "quantile_mae": quantile_mae(boundaries, fresh),
                "feedback_residual_mean": residual_mean,
                "feedback_residual_max": residual_max,
                "accepted_damped": bool(metadata["accepted_damped"]),
                "aggressive_hybrid_choice": metadata["aggressive_hybrid_choice"],
                "recency_hybrid_choice": metadata["recency_hybrid_choice"],
                "residual_hybrid_choice": metadata["residual_hybrid_choice"],
            })

        for pred_id, predicate in enumerate(predicates):
            truth = estimate_selectivity(fresh, predicate)
            for method, boundaries in method_boundaries.items():
                estimate = estimate_selectivity(boundaries, predicate)
                scan_choice, scan_optimal, scan_regret, join_choice, join_optimal, join_regret = regrets(
                    estimate, truth, table_rows, args
                )
                predicate_rows.append({
                    "q_mods": q_mods,
                    "case_id": path.stem,
                    "predicate_id": pred_id,
                    "predicate_type": predicate["predicate_type"],
                    "method": method,
                    "true_selectivity": truth,
                    "estimated_selectivity": estimate,
                    "selectivity_qerr": qerr(estimate, truth),
                    "selectivity_abs_error": abs(estimate - truth),
                    "scan_choice": scan_choice,
                    "scan_optimal": scan_optimal,
                    "scan_regret": scan_regret,
                    "join_choice": join_choice,
                    "join_optimal": join_optimal,
                    "join_regret": join_regret,
                })

    summary = aggregate(case_rows, predicate_rows, risk_threshold=args.risk_threshold)
    verdict = build_verdict(summary, choices, args)
    write_outputs(args.output_dir, case_rows, predicate_rows, summary, choices, verdict)


def parse_args() -> argparse.Namespace:
    root = _REPO_DIR / "experiments" / "results" / "synthetic_paper_suite_rerun_20260529"
    parser = argparse.ArgumentParser(description="Tiny OASIS accuracy/robustness smoke test over cached data")
    parser.add_argument("--data-root", type=Path, default=root / "compound_data")
    parser.add_argument("--model-path", type=Path, default=root / "models" / "oasis_k16.json")
    parser.add_argument("--output-dir", type=Path, default=_REPO_DIR / "experiments" / "results" / "oasis_accuracy_smoke_20260531")
    parser.add_argument("--q-values", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--max-cases-per-q", type=int, default=12)
    parser.add_argument("--predicates-per-case", type=int, default=16)
    parser.add_argument("--num-buckets", type=int, default=10)
    parser.add_argument("--max-observations", type=int, default=16)
    parser.add_argument("--projection-iters", type=int, default=200)
    parser.add_argument("--projection-tol", type=float, default=1e-4)
    parser.add_argument("--soft-projection-strength", type=float, default=30.0)
    parser.add_argument("--soft-projection-recency-decay", type=float, default=0.80)
    parser.add_argument("--soft-projection-target-blend", type=float, default=1.0)
    parser.add_argument("--soft-projection-window", type=int, default=0,
                        help="Use only the most recent N observations for soft projection; 0 uses the full window.")
    parser.add_argument("--soft-projection-iters", type=int, default=500)
    parser.add_argument("--soft-projection-lr", type=float, default=0.05)
    parser.add_argument("--soft-projection-tol", type=float, default=1e-9)
    parser.add_argument("--soft-projection-active-set", action="store_true",
                        help="Apply soft projection only to the latest hard-feasible feedback suffix.")
    parser.add_argument("--soft-projection-conflict-aware", action="store_true",
                        help="Down-weight feedback constraints contradicted by the most recent observations.")
    parser.add_argument("--soft-projection-conflict-ref-window", type=int, default=8,
                        help="Number of most-recent observations treated as the trusted conflict reference.")
    parser.add_argument("--soft-projection-conflict-tau", type=float, default=0.05,
                        help="Conflict bandwidth; smaller tau suppresses contradicted observations more aggressively.")
    parser.add_argument("--soft-projection-conflict-floor", type=float, default=0.0,
                        help="Minimum residual weight for a contradicted observation (0 fully removes it).")
    parser.add_argument("--band-kappa", type=float, default=0.04,
                        help="Banded projection confidence half-width scale (0 reproduces hard projection).")
    parser.add_argument("--band-floor", type=float, default=0.0,
                        help="Constant floor added to every banded-projection confidence half-width.")
    parser.add_argument("--band-conflict-aware", action="store_true",
                        help="Drop banded constraints contradicted by the most recent observations.")
    parser.add_argument("--band-conflict-ref-window", type=int, default=8,
                        help="Most-recent observations treated as the trusted reference for band conflict dropping.")
    parser.add_argument("--band-conflict-tau", type=float, default=0.05,
                        help="Unused weighting bandwidth placeholder kept for symmetry with soft projection.")
    parser.add_argument("--band-conflict-drop", type=float, default=0.10,
                        help="Residual against the recent reference above which a band constraint is dropped.")
    parser.add_argument("--damping-alpha", type=float, default=0.65)
    parser.add_argument("--damping-grid", type=float, nargs="+", default=[0.35, 0.50, 0.65, 0.80, 0.95])
    parser.add_argument("--recent-projection-windows", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--recency-residual-decay", type=float, default=0.70)
    parser.add_argument("--residual-guard-frac", type=float, default=0.30)
    parser.add_argument("--max-residual-guard-frac", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--min-true-selectivity", type=float, default=1e-4)
    parser.add_argument("--risk-threshold", type=float, default=1.05)
    parser.add_argument("--min-table-rows", type=float, default=100_000)
    parser.add_argument("--max-table-rows", type=float, default=10_000_000)

    parser.add_argument("--seq-tuple-cost", type=float, default=1.0)
    parser.add_argument("--index-startup-cost", type=float, default=100.0)
    parser.add_argument("--index-tuple-cost", type=float, default=8.0)
    parser.add_argument("--dim-rows", type=float, default=50_000.0)
    parser.add_argument("--hash-build-tuple-cost", type=float, default=1.0)
    parser.add_argument("--hash-probe-tuple-cost", type=float, default=0.20)
    parser.add_argument("--nl-lookup-cost", type=float, default=12.0)

    parser.add_argument("--min-qerror-improvement-pct", type=float, default=1.0)
    parser.add_argument("--verdict-candidate", type=str, default="oasis_aggressive_hybrid", choices=METHOD_ORDER)
    parser.add_argument("--max-feedback-residual-regression-pct", type=float, default=5.0)
    parser.add_argument("--max-structural-regression-pct", type=float, default=2.0)
    parser.add_argument("--max-join-regret-regression-pct", type=float, default=1.0)
    parser.add_argument("--max-new-risk-loss-delta", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    run_smoke(parse_args())


if __name__ == "__main__":
    main()
