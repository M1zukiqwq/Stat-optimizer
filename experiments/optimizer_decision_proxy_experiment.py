#!/usr/bin/env python3
"""
Generator-driven optimizer decision proxy experiment.

This experiment avoids wall-clock runtime claims. It asks a narrower question:
when generated ground truth gives the true predicate selectivity, do corrected
statistics improve optimizer-facing decisions under a transparent CBO-style
cost proxy?

For each synthetic histogram correction sample, the script:
  1. builds stale, ISOMER, OASIS, OASIS-Proj, Hybrid, and fresh marginals;
  2. generates future predicates from the fresh/ground-truth distribution;
  3. estimates selectivity under each statistics state;
  4. feeds those estimates to scan and join choice proxies;
  5. evaluates the chosen decisions using true selectivity.

The reported regret is not measured runtime. It is a deterministic decision
quality signal: true cost of the plan selected from estimated statistics divided
by the true optimal cost under the same proxy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
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
from modern_baselines import correct_isomer, correct_soft_isomer
from tensorizer import tensorize_sample
from copula_oasis_experiment import GaussianCopula
from factorjoin_oasis_experiment import choose_aggressive_marginal


METHOD_ORDER = [
    "stale",
    "isomer",
    "oasis",
    "oasis_projected",
    "oasis_soft_projection",
    "hybrid",
    "aggressive_hybrid",
    "calibrated_hybrid",
    "fresh",
]


@dataclass
class CostProxyConfig:
    seq_tuple_cost: float = 1.0
    index_startup_cost: float = 100.0
    index_tuple_cost: float = 8.0
    dim_rows: float = 50_000.0
    hash_build_tuple_cost: float = 1.0
    hash_probe_tuple_cost: float = 0.20
    nl_lookup_cost: float = 12.0


@dataclass
class DecisionRow:
    q_mods: int
    case_id: str
    predicate_id: int
    predicate_type: str
    value: float
    value_upper: Optional[float]
    table_rows: int
    true_selectivity: float
    method: str
    estimated_selectivity: float
    selectivity_qerr: float
    scan_choice: str
    scan_optimal: str
    scan_regret: float
    join_choice: str
    join_optimal: str
    join_regret: float


def qerr(estimate: float, truth: float) -> float:
    estimate = max(estimate, 1e-9)
    truth = max(truth, 1e-9)
    return max(estimate / truth, truth / estimate)


def geomean(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    return math.exp(sum(math.log(max(value, 1e-12)) for value in values) / len(values))


def pct_improvement(base: float, value: float) -> float:
    return (base - value) / max(base, 1e-12) * 100.0


def boundaries_from_quantiles(quantiles: Sequence[float]) -> List[float]:
    values = [clamp01(float(value)) for value in quantiles]
    values = project_monotonic(values)
    return [0.0] + values + [1.0]


def cdf_levels(boundaries: Sequence[float]) -> List[float]:
    buckets = max(len(boundaries) - 1, 1)
    return [idx / buckets for idx in range(len(boundaries))]


def estimate_selectivity(boundaries: Sequence[float], predicate: Dict[str, float]) -> float:
    levels = cdf_levels(boundaries)
    pred_type = predicate["predicate_type"]
    value = float(predicate["value"])

    if pred_type in {"<", "<="}:
        return max(evaluate_piecewise_cdf(boundaries, levels, value), 1e-9)
    if pred_type in {">", ">="}:
        return max(1.0 - evaluate_piecewise_cdf(boundaries, levels, value), 1e-9)
    if pred_type == "BETWEEN":
        upper = float(predicate["value_upper"])
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


def feedback_residual(boundaries: Sequence[float], observations: Sequence[dict]) -> float:
    if not observations:
        return float("inf")
    errors = []
    for obs in observations:
        pred = {
            "predicate_type": obs["predicate_type"],
            "value": obs["value"],
            "value_upper": obs.get("value_upper"),
        }
        errors.append(abs(estimate_selectivity(boundaries, pred) - float(obs["actual_sel"])))
    return sum(errors) / len(errors)


def observations_to_dicts(sample) -> List[dict]:
    result = []
    for obs in sample.observations:
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


def oasis_boundaries(sample, model: MlpHistogramModelV2, max_observations: int) -> List[float]:
    record = tensorize_sample(sample, max_observations=max_observations, teacher_fn=None, use_time_decay=False)
    prediction = model.predict([record.feature_tensor])[0]
    return boundaries_from_quantiles(prediction)


def isomer_boundaries(
    prior_boundaries: Sequence[float],
    observations: Sequence[dict],
    num_buckets: int,
    max_iter: int = 200,
    tol: float = 1e-4,
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


def soft_isomer_boundaries(
    prior_boundaries: Sequence[float],
    observations: Sequence[dict],
    num_buckets: int,
    strength: float = 30.0,
    recency_decay: float = 0.80,
    target_blend: float = 1.0,
    observation_window: int = 0,
    max_iter: int = 500,
    learning_rate: float = 0.05,
    tol: float = 1e-9,
    active_set: bool = False,
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
        corrected = correct_soft_isomer(
            float(prior_boundaries[0]),
            float(prior_boundaries[-1]),
            list(prior_boundaries[1:-1]),
            soft_observations,
            num_buckets=num_buckets,
            constraint_strength=strength,
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


def choose_hybrid(
    method_boundaries: Dict[str, List[float]],
    observations: Sequence[dict],
    candidates: Optional[Sequence[str]] = None,
) -> Tuple[str, List[float]]:
    if candidates is None:
        candidates = ["stale", "isomer", "oasis", "oasis_projected"]
    candidates = [m for m in candidates if m in method_boundaries]
    scores = {method: feedback_residual(method_boundaries[method], observations) for method in candidates}
    best_method = min(candidates, key=lambda method: scores[method])
    return best_method, method_boundaries[best_method]


def generate_predicates(
    fresh_boundaries: Sequence[float],
    rng: random.Random,
    count: int,
    min_true_selectivity: float,
) -> List[dict]:
    """Generate future predicates by sampling ranges in quantile space."""
    levels = cdf_levels(fresh_boundaries)
    predicates: List[dict] = []
    attempts = 0

    while len(predicates) < count and attempts < count * 40:
        attempts += 1
        pred_type = rng.choices(
            ["<=", ">=", "BETWEEN", "="],
            weights=[0.30, 0.30, 0.32, 0.08],
            k=1,
        )[0]

        if pred_type == "BETWEEN":
            width = 10 ** rng.uniform(math.log10(0.01), math.log10(0.45))
            lo_p = rng.uniform(0.02, max(0.03, 0.98 - width))
            hi_p = min(0.99, lo_p + width)
            lo = inverse_piecewise_cdf(fresh_boundaries, levels, lo_p)
            hi = inverse_piecewise_cdf(fresh_boundaries, levels, hi_p)
            pred = {"predicate_type": "BETWEEN", "value": lo, "value_upper": hi}
        elif pred_type == "<=":
            p = rng.uniform(0.005, 0.95)
            pred = {
                "predicate_type": "<=",
                "value": inverse_piecewise_cdf(fresh_boundaries, levels, p),
                "value_upper": None,
            }
        elif pred_type == ">=":
            p = rng.uniform(0.05, 0.995)
            pred = {
                "predicate_type": ">=",
                "value": inverse_piecewise_cdf(fresh_boundaries, levels, p),
                "value_upper": None,
            }
        else:
            p = rng.uniform(0.02, 0.98)
            pred = {
                "predicate_type": "=",
                "value": inverse_piecewise_cdf(fresh_boundaries, levels, p),
                "value_upper": None,
            }

        truth = estimate_selectivity(fresh_boundaries, pred)
        if truth >= min_true_selectivity:
            predicates.append(pred)

    return predicates


def scan_cost(choice: str, selectivity: float, table_rows: int, cfg: CostProxyConfig) -> float:
    rows = max(selectivity, 1e-9) * table_rows
    if choice == "index":
        return cfg.index_startup_cost + rows * cfg.index_tuple_cost
    return table_rows * cfg.seq_tuple_cost


def choose_scan(selectivity: float, table_rows: int, cfg: CostProxyConfig) -> str:
    index = scan_cost("index", selectivity, table_rows, cfg)
    seq = scan_cost("seq", selectivity, table_rows, cfg)
    return "index" if index < seq else "seq"


def join_cost(choice: str, selectivity: float, table_rows: int, cfg: CostProxyConfig) -> float:
    rows = max(selectivity, 1e-9) * table_rows
    if choice == "nested_loop":
        return scan_cost("index", selectivity, table_rows, cfg) + rows * cfg.nl_lookup_cost
    return (
        scan_cost("seq", selectivity, table_rows, cfg)
        + cfg.dim_rows * cfg.hash_build_tuple_cost
        + rows * cfg.hash_probe_tuple_cost
    )


def choose_join(selectivity: float, table_rows: int, cfg: CostProxyConfig) -> str:
    nested = join_cost("nested_loop", selectivity, table_rows, cfg)
    hashed = join_cost("hash_join", selectivity, table_rows, cfg)
    return "nested_loop" if nested < hashed else "hash_join"


def regret_for_scan(estimated_sel: float, true_sel: float, table_rows: int, cfg: CostProxyConfig) -> Tuple[str, str, float]:
    selected = choose_scan(estimated_sel, table_rows, cfg)
    optimal = choose_scan(true_sel, table_rows, cfg)
    selected_true_cost = scan_cost(selected, true_sel, table_rows, cfg)
    optimal_true_cost = scan_cost(optimal, true_sel, table_rows, cfg)
    return selected, optimal, selected_true_cost / max(optimal_true_cost, 1e-12)


def regret_for_join(estimated_sel: float, true_sel: float, table_rows: int, cfg: CostProxyConfig) -> Tuple[str, str, float]:
    selected = choose_join(estimated_sel, table_rows, cfg)
    optimal = choose_join(true_sel, table_rows, cfg)
    selected_true_cost = join_cost(selected, true_sel, table_rows, cfg)
    optimal_true_cost = join_cost(optimal, true_sel, table_rows, cfg)
    return selected, optimal, selected_true_cost / max(optimal_true_cost, 1e-12)


def build_method_boundaries(
    sample,
    model: MlpHistogramModelV2,
    num_buckets: int,
    max_observations: int,
    aggressive_damping_grid: Sequence[float] = (0.35, 0.50, 0.65, 0.80, 0.95),
    aggressive_recent_windows: Sequence[int] = (4, 8, 12),
    soft_projection_strength: float = 30.0,
    soft_projection_recency_decay: float = 0.80,
    soft_projection_target_blend: float = 1.0,
    soft_projection_window: int = 0,
    soft_projection_iters: int = 500,
    soft_projection_lr: float = 0.05,
    soft_projection_tol: float = 1e-9,
    soft_projection_active_set: bool = False,
    soft_projection_conflict_aware: bool = False,
    soft_projection_conflict_ref_window: int = 8,
    soft_projection_conflict_tau: float = 0.05,
    soft_projection_conflict_floor: float = 0.0,
    projection_iters: int = 200,
    projection_tol: float = 1e-4,
) -> Tuple[Dict[str, List[float]], str]:
    observations = observations_to_dicts(sample)
    stale = boundaries_from_quantiles(sample.prior.quantile_values)
    fresh = boundaries_from_quantiles(sample.corrected_quantile_values or sample.prior.quantile_values)
    isomer = isomer_boundaries(stale, observations, num_buckets, projection_iters, projection_tol)
    oasis = oasis_boundaries(sample, model, max_observations)
    oasis_projected = isomer_boundaries(oasis, observations, num_buckets, projection_iters, projection_tol)
    oasis_soft_projection = soft_isomer_boundaries(
        oasis,
        observations,
        num_buckets,
        strength=soft_projection_strength,
        recency_decay=soft_projection_recency_decay,
        target_blend=soft_projection_target_blend,
        observation_window=soft_projection_window,
        max_iter=soft_projection_iters,
        learning_rate=soft_projection_lr,
        tol=soft_projection_tol,
        active_set=soft_projection_active_set,
        conflict_aware=soft_projection_conflict_aware,
        conflict_ref_window=soft_projection_conflict_ref_window,
        conflict_tau=soft_projection_conflict_tau,
        conflict_floor=soft_projection_conflict_floor,
    )

    method_boundaries = {
        "stale": stale,
        "isomer": isomer,
        "oasis": oasis,
        "oasis_projected": oasis_projected,
        "oasis_soft_projection": oasis_soft_projection,
        "fresh": fresh,
    }
    hybrid_choice, hybrid = choose_hybrid(method_boundaries, observations)
    method_boundaries["hybrid"] = hybrid
    copula = GaussianCopula(num_buckets=num_buckets)
    aggressive, _, _ = choose_aggressive_marginal(
        copula=copula,
        stale=stale,
        isomer=isomer,
        oasis=oasis,
        projected=oasis_projected,
        hybrid=hybrid,
        observations=observations,
        num_buckets=num_buckets,
        damping_grid=aggressive_damping_grid,
        recent_windows=aggressive_recent_windows,
        projection_iters=projection_iters,
        projection_tol=projection_tol,
    )
    method_boundaries["aggressive_hybrid"] = aggressive
    _, calibrated = choose_hybrid(
        method_boundaries, observations,
        candidates=["stale", "isomer", "oasis", "oasis_projected", "oasis_soft_projection"],
    )
    method_boundaries["calibrated_hybrid"] = calibrated
    return method_boundaries, hybrid_choice


def iter_sample_paths(data_root: Path, q_values: Sequence[int], max_cases_per_q: int) -> List[Tuple[int, Path]]:
    paths = []
    for q in q_values:
        q_dir = data_root / f"test_q{q}"
        q_paths = sorted(q_dir.glob("*.json"))
        if max_cases_per_q > 0:
            q_paths = q_paths[:max_cases_per_q]
        paths.extend((q, path) for path in q_paths)
    return paths


def aggregate_rows(rows: Sequence[DecisionRow], risk_threshold: float) -> List[dict]:
    grouped: Dict[Tuple[str, Optional[int]], List[DecisionRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.method, None)].append(row)
        grouped[(row.method, row.q_mods)].append(row)

    summary = []
    for (method, q_mods), method_rows in sorted(grouped.items(), key=lambda item: (999 if item[0][1] is None else item[0][1], METHOD_ORDER.index(item[0][0]))):
        stale_rows = [row for row in rows if row.q_mods == q_mods] if q_mods is not None else list(rows)
        stale_by_key = {
            (row.q_mods, row.case_id, row.predicate_id): row
            for row in stale_rows
            if row.method == "stale"
        }
        fresh_by_key = {
            (row.q_mods, row.case_id, row.predicate_id): row
            for row in stale_rows
            if row.method == "fresh"
        }

        sel_qerr = geomean([row.selectivity_qerr for row in method_rows])
        scan_regret = geomean([row.scan_regret for row in method_rows])
        join_regret = geomean([row.join_regret for row in method_rows])
        scan_match = sum(row.scan_choice == row.scan_optimal for row in method_rows) / max(len(method_rows), 1)
        join_match = sum(row.join_choice == row.join_optimal for row in method_rows) / max(len(method_rows), 1)

        resolved = 0
        losses = 0
        risky = 0
        fresh_match = 0
        for row in method_rows:
            key = (row.q_mods, row.case_id, row.predicate_id)
            stale = stale_by_key.get(key)
            fresh = fresh_by_key.get(key)
            if stale is None:
                continue
            if stale.join_regret >= risk_threshold:
                risky += 1
                if row.join_regret < risk_threshold:
                    resolved += 1
            elif row.join_regret >= risk_threshold:
                losses += 1
            if fresh is not None and row.join_choice == fresh.join_choice:
                fresh_match += 1

        summary.append({
            "q_mods": "all" if q_mods is None else q_mods,
            "method": method,
            "n": len(method_rows),
            "selectivity_qerr_gm": sel_qerr,
            "scan_regret_gm": scan_regret,
            "join_regret_gm": join_regret,
            "scan_optimal_match_frac": scan_match,
            "join_optimal_match_frac": join_match,
            "join_fresh_match_frac": fresh_match / max(len(method_rows), 1),
            "risky_stale_cases": risky,
            "risk_resolved_frac": resolved / max(risky, 1),
            "new_risk_loss_frac": losses / max(len(method_rows), 1),
        })
    return summary


def write_outputs(output_dir: Path, rows: Sequence[DecisionRow], summary: Sequence[dict], hybrid_choices: Dict[str, int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "decision_rows.json", "w") as f:
        json.dump([asdict(row) for row in rows], f, indent=2)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(list(summary), f, indent=2)
    with open(output_dir / "hybrid_choices.json", "w") as f:
        json.dump(hybrid_choices, f, indent=2)

    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    with open(output_dir / "decision_rows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    write_latex_table(output_dir, summary)
    write_text_summary(output_dir, summary, hybrid_choices)


def write_latex_table(output_dir: Path, summary: Sequence[dict]) -> None:
    all_rows = [row for row in summary if row["q_mods"] == "all"]
    by_method = {row["method"]: row for row in all_rows}
    path = output_dir / "table_optimizer_decision_proxy.tex"
    with open(path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("  \\centering\n")
        f.write("  \\small\n")
        f.write("  \\caption{Generator-driven optimizer-decision proxy. Q-Error and regret are geometric means; regret is true proxy cost of the plan selected from estimated statistics divided by the true optimal proxy cost ($1.0$ is optimal). No query runtime is measured.}\n")
        f.write("  \\label{tab:optimizer_decision_proxy}\n")
        f.write("  \\setlength{\\tabcolsep}{4pt}\n")
        f.write("  \\begin{tabular}{lrrrrr}\n")
        f.write("    \\toprule\n")
        f.write("    Method & Sel. QE & Join Regret & Join Opt. & Fresh Match & Risk Resolved \\\\\n")
        f.write("    \\midrule\n")
        for method in METHOD_ORDER:
            row = by_method[method]
            label = {
                "stale": "Stale",
                "isomer": "ISOMER",
                "oasis": "OASIS",
                "oasis_projected": "OASIS-Proj",
                "oasis_soft_projection": "OASIS-Soft",
                "hybrid": "Hybrid",
                "aggressive_hybrid": "Aggressive",
                "calibrated_hybrid": "Calibrated",
                "fresh": "Fresh",
            }[method]
            f.write(
                f"    {label} & {row['selectivity_qerr_gm']:.3f} & "
                f"{row['join_regret_gm']:.3f} & "
                f"{row['join_optimal_match_frac'] * 100:.1f}\\% & "
                f"{row['join_fresh_match_frac'] * 100:.1f}\\% & "
                f"{row['risk_resolved_frac'] * 100:.1f}\\% \\\\\n"
            )
        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n")


def write_text_summary(output_dir: Path, summary: Sequence[dict], hybrid_choices: Dict[str, int]) -> None:
    all_rows = [row for row in summary if row["q_mods"] == "all"]
    by_method = {row["method"]: row for row in all_rows}
    stale = by_method["stale"]
    lines = []
    lines.append("Generator-driven optimizer decision proxy")
    lines.append("=" * 48)
    lines.append("No wall-clock runtime is measured. Regret is true proxy cost")
    lines.append("of the plan selected from estimated statistics divided by")
    lines.append("the true optimal proxy cost under the same cost model.")
    lines.append("")
    lines.append(f"Rows per method: {stale['n']}")
    lines.append("")
    lines.append("Method          SelQE  JoinReg  JoinOpt  FreshMatch  RiskResolved  QEImp  RegImp")
    lines.append("-" * 86)
    for method in METHOD_ORDER:
        row = by_method[method]
        qe_imp = pct_improvement(stale["selectivity_qerr_gm"], row["selectivity_qerr_gm"])
        regret_imp = pct_improvement(stale["join_regret_gm"], row["join_regret_gm"])
        lines.append(
            f"{method:<15s} {row['selectivity_qerr_gm']:5.3f}  "
            f"{row['join_regret_gm']:7.4f}  "
            f"{row['join_optimal_match_frac']*100:7.1f}%  "
            f"{row['join_fresh_match_frac']*100:10.1f}%  "
            f"{row['risk_resolved_frac']*100:12.1f}%  "
            f"{qe_imp:5.1f}%  {regret_imp:6.1f}%"
        )
    total_choices = sum(hybrid_choices.values())
    if total_choices:
        lines.append("")
        lines.append("Hybrid choices:")
        for method in sorted(hybrid_choices):
            lines.append(f"  {method}: {hybrid_choices[method] / total_choices * 100:.1f}%")

    text = "\n".join(lines)
    (output_dir / "summary.txt").write_text(text + "\n")
    print(text)


def run_experiment(args: argparse.Namespace) -> None:
    model = MlpHistogramModelV2.load(str(args.model_path))
    cfg = CostProxyConfig(
        seq_tuple_cost=args.seq_tuple_cost,
        index_startup_cost=args.index_startup_cost,
        index_tuple_cost=args.index_tuple_cost,
        dim_rows=args.dim_rows,
        hash_build_tuple_cost=args.hash_build_tuple_cost,
        hash_probe_tuple_cost=args.hash_probe_tuple_cost,
        nl_lookup_cost=args.nl_lookup_cost,
    )

    sample_paths = iter_sample_paths(args.data_root, args.q_values, args.max_cases_per_q)
    if not sample_paths:
        raise FileNotFoundError(f"No test samples found under {args.data_root}")

    rows: List[DecisionRow] = []
    hybrid_choices: Dict[str, int] = defaultdict(int)

    for sample_index, (q_mods, path) in enumerate(sample_paths):
        sample = load_feedback_sample(str(path))
        method_boundaries, hybrid_choice = build_method_boundaries(
            sample,
            model=model,
            num_buckets=args.num_buckets,
            max_observations=args.max_observations,
            aggressive_damping_grid=args.aggressive_damping_grid,
            aggressive_recent_windows=args.aggressive_recent_windows,
            soft_projection_strength=args.soft_projection_strength,
            soft_projection_recency_decay=args.soft_projection_recency_decay,
            soft_projection_target_blend=args.soft_projection_target_blend,
            soft_projection_window=args.soft_projection_window,
            soft_projection_iters=args.soft_projection_iters,
            soft_projection_lr=args.soft_projection_lr,
            soft_projection_tol=args.soft_projection_tol,
            soft_projection_active_set=args.soft_projection_active_set,
            soft_projection_conflict_aware=args.soft_projection_conflict_aware,
            soft_projection_conflict_ref_window=args.soft_projection_conflict_ref_window,
            soft_projection_conflict_tau=args.soft_projection_conflict_tau,
            soft_projection_conflict_floor=args.soft_projection_conflict_floor,
            projection_iters=args.projection_iters,
            projection_tol=args.projection_tol,
        )
        hybrid_choices[hybrid_choice] += 1

        fresh = method_boundaries["fresh"]
        rng = random.Random(args.seed + q_mods * 100_000 + sample_index)
        table_rows = int(10 ** rng.uniform(math.log10(args.min_table_rows), math.log10(args.max_table_rows)))
        predicates = generate_predicates(
            fresh,
            rng=rng,
            count=args.predicates_per_case,
            min_true_selectivity=args.min_true_selectivity,
        )

        case_id = path.stem
        for pred_id, predicate in enumerate(predicates):
            true_sel = estimate_selectivity(fresh, predicate)
            for method in METHOD_ORDER:
                estimated = estimate_selectivity(method_boundaries[method], predicate)
                scan_choice, scan_optimal, scan_regret = regret_for_scan(estimated, true_sel, table_rows, cfg)
                join_choice, join_optimal, join_regret = regret_for_join(estimated, true_sel, table_rows, cfg)
                rows.append(DecisionRow(
                    q_mods=q_mods,
                    case_id=case_id,
                    predicate_id=pred_id,
                    predicate_type=predicate["predicate_type"],
                    value=float(predicate["value"]),
                    value_upper=predicate.get("value_upper"),
                    table_rows=table_rows,
                    true_selectivity=true_sel,
                    method=method,
                    estimated_selectivity=estimated,
                    selectivity_qerr=qerr(estimated, true_sel),
                    scan_choice=scan_choice,
                    scan_optimal=scan_optimal,
                    scan_regret=scan_regret,
                    join_choice=join_choice,
                    join_optimal=join_optimal,
                    join_regret=join_regret,
                ))

        if (sample_index + 1) % 50 == 0:
            print(f"Processed {sample_index + 1}/{len(sample_paths)} samples")

    summary = aggregate_rows(rows, risk_threshold=args.risk_threshold)
    write_outputs(args.output_dir, rows, summary, dict(hybrid_choices))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generator-driven optimizer-decision proxy experiment")
    parser.add_argument("--data-root", type=Path,
                        default=_REPO_DIR / "experiments" / "results" / "synthetic_paper_suite_rerun_20260529" / "compound_data")
    parser.add_argument("--model-path", type=Path,
                        default=_REPO_DIR / "experiments" / "results" / "synthetic_paper_suite_rerun_20260529" / "models" / "oasis_k16.json")
    parser.add_argument("--output-dir", type=Path,
                        default=_REPO_DIR / "experiments" / "results" / "optimizer_decision_proxy_20260529")
    parser.add_argument("--q-values", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    parser.add_argument("--max-cases-per-q", type=int, default=128)
    parser.add_argument("--predicates-per-case", type=int, default=32)
    parser.add_argument("--num-buckets", type=int, default=10)
    parser.add_argument("--max-observations", type=int, default=16)
    parser.add_argument("--aggressive-damping-grid", type=float, nargs="+",
                        default=[0.35, 0.50, 0.65, 0.80, 0.95])
    parser.add_argument("--aggressive-recent-windows", type=int, nargs="+",
                        default=[4, 8, 12])
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
    parser.add_argument("--projection-iters", type=int, default=200)
    parser.add_argument("--projection-tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
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
    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
