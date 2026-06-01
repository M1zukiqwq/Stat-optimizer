#!/usr/bin/env python3
"""Stage-1 estimator swap: is the Stage-2 calibration layer estimator-agnostic?

This additive experiment reuses the cached drift data and the trained OASIS
checkpoint (no retraining). It plugs several *different* Stage-1 marginal
estimators into the pipeline---stale prior, LinInterp, FeedAvg, STHoles,
QuickSel-H, and the OASIS MLP---and, for each, applies three Stage-2 settings:

  - none   : the raw Stage-1 marginal;
  - hard   : hard feedback-consistency projection initialized from that marginal;
  - router : the residual-gated Router over
             {stale, ISOMER, this marginal, hard-proj(marginal),
              Soft(marginal)}.

The question is whether the Stage-2 calibration/router improves future
single-column accuracy and lowers feedback residual *regardless of which Stage-1
estimator produced the prior*. If so, the calibration layer is the reusable,
estimator-agnostic contribution rather than something specific to the OASIS MLP.
Metrics: future selectivity Q-error (geometric mean over generated predicates),
feedback residual on the observation window, and quantile MAE versus fresh.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_PIPELINE_DIR = _REPO_DIR / "cdf_kll_ml_pipeline"
for _p in (_PIPELINE_DIR, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from json_histogram_parser import load_feedback_sample
from mlp_histogram_model_v2 import MlpHistogramModelV2
from baselines import correct_linear_interp, correct_feedback_avg, correct_stholes_tree
from modern_baselines import correct_quicksel_h

import oasis_accuracy_smoke as smoke

# Stage-1 estimators that produce a corrected single-column marginal. Each takes
# (prior_min, prior_max, prior_quantiles, observations, num_buckets) and returns
# inner quantiles; the OASIS MLP is handled separately.
BASELINE_ESTIMATORS: Dict[str, Callable] = {
    "lininterp": correct_linear_interp,
    "feedavg": correct_feedback_avg,
    "stholes": correct_stholes_tree,
    "quicksel_h": correct_quicksel_h,
}
SOURCE_ORDER = ["stale", "lininterp", "feedavg", "stholes", "quicksel_h", "oasis_mlp"]
SOURCE_LABEL = {
    "stale": "Stale prior", "lininterp": "LinInterp", "feedavg": "FeedAvg",
    "stholes": "STHoles", "quicksel_h": "QuickSel-H", "oasis_mlp": "OASIS MLP",
}
VARIANTS = ["none", "hard", "router"]


class ProgressBar:
    """Tiny zero-dependency progress bar for long cached experiment runs."""

    def __init__(self, total: int, *, enabled: bool = True, label: str = "progress") -> None:
        self.total = max(int(total), 1)
        self.enabled = enabled
        self.label = label
        self.count = 0
        self.start = time.time()
        self.last_draw = 0.0

    def update(self, step: int = 1, status: str = "") -> None:
        self.count = min(self.total, self.count + step)
        if not self.enabled:
            return
        now = time.time()
        if self.count < self.total and now - self.last_draw < 0.20:
            return
        self.last_draw = now
        width = 30
        filled = int(width * self.count / self.total)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = max(now - self.start, 1e-9)
        rate = self.count / elapsed
        remaining = (self.total - self.count) / max(rate, 1e-9)
        message = (
            f"\r{self.label} [{bar}] {self.count}/{self.total} "
            f"({self.count / self.total * 100:5.1f}%) "
            f"elapsed {elapsed:6.1f}s eta {remaining:6.1f}s"
        )
        if status:
            message += f" | {status[:48]:<48}"
        sys.stderr.write(message)
        sys.stderr.flush()

    def close(self) -> None:
        if self.enabled:
            self.update(0)
            sys.stderr.write("\n")
            sys.stderr.flush()


def mean(values: Sequence[float], default: float = 0.0) -> float:
    if not values:
        return default
    return sum(values) / len(values)


def stage1_marginal(source: str, sample, model, args) -> Tuple[List[float], bool]:
    if source == "stale":
        return smoke.boundaries_from_quantiles(sample.prior.quantile_values), True
    if source == "oasis_mlp":
        return smoke.oasis_boundaries(sample, model, args.max_observations), True
    fn = BASELINE_ESTIMATORS[source]
    observations = smoke.observations_to_dicts(sample, args.max_observations)
    try:
        q = fn(0.0, 1.0, list(sample.prior.quantile_values), observations, num_buckets=args.num_buckets)
        return smoke.boundaries_from_quantiles(q), True
    except Exception:
        return smoke.boundaries_from_quantiles(sample.prior.quantile_values), False


def stage2_variants(
    prior: Sequence[float],
    stale: Sequence[float],
    isomer: Sequence[float],
    observations: Sequence[dict],
    args,
) -> Tuple[Dict[str, List[float]], Dict[str, str]]:
    """Return {none, hard, router} for a given Stage-1 marginal `prior`."""
    hard = smoke.project_boundaries(prior, observations, args.num_buckets,
                                    args.projection_iters, args.projection_tol)
    soft = smoke.soft_project_boundaries(
        prior, observations, args.num_buckets, args.soft_strength, 1.0, 1.0,
        args.soft_iters, args.soft_lr, args.soft_tol, False,
        conflict_aware=True, conflict_ref_window=args.conflict_ref_window,
        conflict_tau=args.conflict_tau, conflict_floor=0.0,
    )
    pool = {"stale": list(stale), "isomer": list(isomer), "noproj": list(prior),
            "hard": hard, "soft": soft}
    router_choice, router = smoke.choose_residual_hybrid(pool, observations, candidates=list(pool))
    return {"none": list(prior), "hard": hard, "router": router}, {"router": router_choice}


def run(args) -> None:
    model = MlpHistogramModelV2.load(str(args.model_path))
    paths = smoke.sample_paths(args.data_root, args.q_values, args.max_cases_per_q)
    if not paths:
        raise FileNotFoundError(f"No test samples found under {args.data_root}")

    qerrs: Dict[tuple, List[float]] = defaultdict(list)
    resids: Dict[tuple, List[float]] = defaultdict(list)
    qmaes: Dict[tuple, List[float]] = defaultdict(list)
    qerrs_by_q: Dict[tuple, List[float]] = defaultdict(list)
    resids_by_q: Dict[tuple, List[float]] = defaultdict(list)
    qmaes_by_q: Dict[tuple, List[float]] = defaultdict(list)
    case_counts: Counter = Counter()
    case_counts_by_q: Counter = Counter()
    router_choice_counts: Counter = Counter()
    router_choice_counts_by_q: Counter = Counter()
    fallback_counts: Counter = Counter()
    valid_cases = 0

    progress = ProgressBar(
        len(paths) * len(SOURCE_ORDER),
        enabled=not args.no_progress,
        label="stage1-estimator-swap",
    )

    for idx, (q_mods, path) in enumerate(paths):
        sample = load_feedback_sample(str(path))
        if sample.corrected_quantile_values is None:
            continue
        valid_cases += 1
        observations = smoke.observations_to_dicts(sample, args.max_observations)
        stale = smoke.boundaries_from_quantiles(sample.prior.quantile_values)
        fresh = smoke.boundaries_from_quantiles(sample.corrected_quantile_values)
        isomer = smoke.project_boundaries(stale, observations, args.num_buckets,
                                          args.projection_iters, args.projection_tol)
        rng = random.Random(args.seed + q_mods * 100_000 + idx)
        predicates = smoke.generate_predicates(fresh, rng, args.predicates_per_case, args.min_true_selectivity)
        truths = [smoke.estimate_selectivity(fresh, p) for p in predicates]

        for source in SOURCE_ORDER:
            prior, ok = stage1_marginal(source, sample, model, args)
            if not ok:
                fallback_counts[source] += 1
            variants, choices = stage2_variants(prior, stale, isomer, observations, args)
            router_choice_counts[(source, choices["router"])] += 1
            router_choice_counts_by_q[(q_mods, source, choices["router"])] += 1
            for variant, bounds in variants.items():
                key = (source, variant)
                key_q = (q_mods, source, variant)
                est = [smoke.estimate_selectivity(bounds, p) for p in predicates]
                pred_qerrs = [smoke.qerr(e, t) for e, t in zip(est, truths)]
                resid = smoke.feedback_residuals(bounds, observations)[0]
                qmae = smoke.quantile_mae(bounds, fresh)
                qerrs[key].extend(pred_qerrs)
                resids[key].append(resid)
                qmaes[key].append(qmae)
                qerrs_by_q[key_q].extend(pred_qerrs)
                resids_by_q[key_q].append(resid)
                qmaes_by_q[key_q].append(qmae)
                case_counts[key] += 1
                case_counts_by_q[key_q] += 1
            progress.update(status=f"q={q_mods} source={source}")

    progress.close()

    write_outputs(
        args.output_dir,
        qerrs,
        resids,
        qmaes,
        qerrs_by_q,
        resids_by_q,
        qmaes_by_q,
        case_counts,
        case_counts_by_q,
        router_choice_counts,
        router_choice_counts_by_q,
        fallback_counts,
        valid_cases,
        args,
    )


def build_summary_rows(sources, variants, qerrs, resids, qmaes, case_counts, key_prefix=()) -> List[dict]:
    rows = []
    for source in sources:
        row = {"source": source, "label": SOURCE_LABEL[source]}
        max_cases = 0
        for variant in variants:
            key = tuple(key_prefix) + (source, variant)
            max_cases = max(max_cases, case_counts.get(key, 0))
            row[f"{variant}_qerr"] = smoke.geomean(qerrs.get(key, []))
            row[f"{variant}_resid"] = mean(resids.get(key, []))
            row[f"{variant}_qmae"] = mean(qmaes.get(key, []))
        row["cases"] = max_cases
        rows.append(row)
    return rows


def write_choice_counts(
    output_dir: Path,
    q_values: Sequence[int],
    router_choice_counts: Counter,
    router_choice_counts_by_q: Counter,
) -> None:
    choices = ["stale", "isomer", "noproj", "hard", "soft"]
    rows = []
    for source in SOURCE_ORDER:
        total = sum(router_choice_counts.get((source, choice), 0) for choice in choices)
        for choice in choices:
            count = router_choice_counts.get((source, choice), 0)
            rows.append({
                "q": "all",
                "source": source,
                "label": SOURCE_LABEL[source],
                "router_choice": choice,
                "count": count,
                "fraction": count / max(total, 1),
            })
    for q_mods in q_values:
        for source in SOURCE_ORDER:
            total = sum(router_choice_counts_by_q.get((q_mods, source, choice), 0) for choice in choices)
            for choice in choices:
                count = router_choice_counts_by_q.get((q_mods, source, choice), 0)
                rows.append({
                    "q": q_mods,
                    "source": source,
                    "label": SOURCE_LABEL[source],
                    "router_choice": choice,
                    "count": count,
                    "fraction": count / max(total, 1),
                })

    with (output_dir / "router_choices.csv").open("w", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "router_choices.json").open("w", encoding="utf-8") as h:
        json.dump(rows, h, indent=2)


def write_latex_table(output_dir: Path, rows: Sequence[dict]) -> None:
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        "  \\caption{Stage-1 prior-source swap under a fixed Stage-2 calibration protocol. "
        "Each row supplies a different single-column marginal prior to Stage 2; "
        "lower Q-error and feedback residual are better.}",
        "  \\label{tab:stage1_estimator_swap}",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\begin{tabular}{lrrrrrr}",
        "    \\toprule",
        "    Stage-1 source & Raw QE & +Hard QE & +Router QE & Router gain & Raw resid. & Router resid. \\\\",
        "    \\midrule",
    ]
    for row in rows:
        raw = row["none_qerr"]
        router = row["router_qerr"]
        gain = (raw - router) / max(raw, 1e-12) * 100.0
        lines.append(
            f"    {row['label']} & {raw:.3f} & {row['hard_qerr']:.3f} & "
            f"{router:.3f} & {gain:.1f}\\% & {row['none_resid']:.4f} & "
            f"{row['router_resid']:.4f} \\\\"
        )
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ]
    (output_dir / "table_stage1_estimator_swap.tex").write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    output_dir: Path,
    qerrs,
    resids,
    qmaes,
    qerrs_by_q,
    resids_by_q,
    qmaes_by_q,
    case_counts,
    case_counts_by_q,
    router_choice_counts,
    router_choice_counts_by_q,
    fallback_counts,
    valid_cases: int,
    args,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_summary_rows(SOURCE_ORDER, VARIANTS, qerrs, resids, qmaes, case_counts)
    by_q_rows = []
    for q_mods in args.q_values:
        for row in build_summary_rows(
            SOURCE_ORDER, VARIANTS, qerrs_by_q, resids_by_q, qmaes_by_q,
            case_counts_by_q, key_prefix=(q_mods,),
        ):
            by_q_rows.append({"q": q_mods, **row})

    with (output_dir / "estimator_swap_overall.csv").open("w", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "estimator_swap_overall.json").open("w", encoding="utf-8") as h:
        json.dump(rows, h, indent=2)
    with (output_dir / "estimator_swap_by_q.csv").open("w", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(by_q_rows[0].keys()))
        writer.writeheader()
        writer.writerows(by_q_rows)
    with (output_dir / "estimator_swap_by_q.json").open("w", encoding="utf-8") as h:
        json.dump(by_q_rows, h, indent=2)
    # Backward-compatible filenames for existing notes/scripts.
    with (output_dir / "estimator_swap.csv").open("w", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "estimator_swap.json").open("w", encoding="utf-8") as h:
        json.dump(rows, h, indent=2)

    write_choice_counts(output_dir, args.q_values, router_choice_counts, router_choice_counts_by_q)
    write_latex_table(output_dir, rows)

    run_config = {
        "q_values": list(args.q_values),
        "max_cases_per_q": args.max_cases_per_q,
        "predicates_per_case": args.predicates_per_case,
        "valid_cases": valid_cases,
        "num_buckets": args.num_buckets,
        "max_observations": args.max_observations,
        "projection_iters": args.projection_iters,
        "projection_tol": args.projection_tol,
        "soft_strength": args.soft_strength,
        "soft_iters": args.soft_iters,
        "soft_lr": args.soft_lr,
        "soft_tol": args.soft_tol,
        "conflict_ref_window": args.conflict_ref_window,
        "conflict_tau": args.conflict_tau,
        "seed": args.seed,
        "data_root": str(args.data_root),
        "model_path": str(args.model_path),
        "fallback_counts": dict(fallback_counts),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as h:
        json.dump(run_config, h, indent=2)

    lines = [
        "Stage-1 estimator swap: is Stage-2 calibration estimator-agnostic?",
        "=" * 68,
        f"Cached full-grid cases: {valid_cases} "
        f"(q={','.join(str(q) for q in args.q_values)}, "
        f"max_cases_per_q={args.max_cases_per_q}, "
        f"predicates_per_case={args.predicates_per_case})",
        "",
        "Future selectivity Q-error (lower is better) per Stage-1 estimator:",
        "",
        f"{'Stage-1 estimator':<16}{'raw':>9}{'+hard':>9}{'+router':>9}{'router gain':>13}",
        "-" * 56,
    ]
    all_improve = True
    router_best = True
    for row in rows:
        raw, hard, rtr = row["none_qerr"], row["hard_qerr"], row["router_qerr"]
        gain = (raw - rtr) / max(raw, 1e-12) * 100.0
        lines.append(f"{row['label']:<16}{raw:>9.3f}{hard:>9.3f}{rtr:>9.3f}{gain:>11.1f}%")
        if not (rtr <= raw + 1e-9):
            all_improve = False
        if not (rtr <= hard + 1e-9):
            router_best = False
    lines += [
        "",
        "Feedback residual (lower is better): raw / +hard / +router",
    ]
    for row in rows:
        lines.append(f"  {row['label']:<16}{row['none_resid']:.4f} / {row['hard_resid']:.4f} / {row['router_resid']:.4f}")
    lines += [
        "",
        f"Router reduces future Q-error vs the raw Stage-1 marginal for every estimator: {all_improve}",
        f"Router is <= hard projection for every estimator: {router_best}",
        "",
        "Router choice mix (aggregate):",
    ]
    for source in SOURCE_ORDER:
        total = sum(router_choice_counts.get((source, choice), 0)
                    for choice in ["stale", "isomer", "noproj", "hard", "soft"])
        parts = []
        for choice in ["stale", "isomer", "noproj", "hard", "soft"]:
            count = router_choice_counts.get((source, choice), 0)
            if count:
                parts.append(f"{choice} {count / max(total, 1) * 100:.1f}%")
        lines.append(f"  {SOURCE_LABEL[source]:<16}{', '.join(parts) if parts else 'n/a'}")
    if fallback_counts:
        lines += ["", f"Stage-1 fallback counts: {dict(fallback_counts)}"]
    text = "\n".join(lines)
    (output_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


def parse_args() -> argparse.Namespace:
    root = _REPO_DIR / "experiments" / "results" / "synthetic_paper_suite_rerun_20260529"
    p = argparse.ArgumentParser(description="Stage-1 estimator swap experiment")
    p.add_argument("--data-root", type=Path, default=root / "compound_data")
    p.add_argument("--model-path", type=Path, default=root / "models" / "oasis_k16.json")
    p.add_argument("--output-dir", type=Path,
                   default=_REPO_DIR / "experiments" / "results" / "stage1_estimator_swap_20260531")
    p.add_argument("--q-values", type=int, nargs="+", default=[1, 3, 5, 10, 15, 20, 25, 30])
    p.add_argument("--max-cases-per-q", type=int, default=128)
    p.add_argument("--predicates-per-case", type=int, default=32)
    p.add_argument("--num-buckets", type=int, default=10)
    p.add_argument("--max-observations", type=int, default=16)
    p.add_argument("--projection-iters", type=int, default=200)
    p.add_argument("--projection-tol", type=float, default=1e-4)
    p.add_argument("--soft-strength", type=float, default=30.0)
    p.add_argument("--soft-iters", type=int, default=500)
    p.add_argument("--soft-lr", type=float, default=0.05)
    p.add_argument("--soft-tol", type=float, default=1e-9)
    p.add_argument("--conflict-ref-window", type=int, default=8)
    p.add_argument("--conflict-tau", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=20260531)
    p.add_argument("--min-true-selectivity", type=float, default=1e-4)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
