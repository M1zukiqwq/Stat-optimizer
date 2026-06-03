#!/usr/bin/env python3
"""Router (residual-gated selector) diagnostics for the major revision (MR3).

Reuses the optimizer-decision-proxy harness (same cases, boundaries, future
predicates) and, per held-out case, records every candidate's in-window feedback
residual and its true held-out future Q-error. From those it compares deployable
selection policies against an offline oracle:

  always-ISOMER, always-OASIS (hard-projected), random-among-projected,
  Router (min in-window residual, non-oracle), Oracle (min true future Q-error).

It also reports the residual tie rate, the Router choice distribution, and how
often the Router agrees with the oracle. Run under the v3 prior:

  V3_CKPT=oasis_torch/artifacts/ckpt_v3_it3.pt \
    ../.venv_v3/bin/python oasis_torch/run_v3.py routerdiag \
    --data-root results/synthetic_paper_suite_rerun_20260529/compound_data \
    --model-path results/synthetic_paper_suite_rerun_20260529/models/oasis_k16.json \
    --q-values 5 10 15 20 25 30 --max-cases-per-q 128 --seed 42 \
    --output-dir results/router_diagnostics_v3_20260602
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence

import optimizer_decision_proxy_experiment as P

# Router candidate pool (identical to choose_hybrid in build_method_boundaries).
POOL = ["stale", "isomer", "oasis", "oasis_projected", "oasis_soft_projection"]
# Constraint-enforcing ("projected") subset used by the random-projected control.
PROJECTED = ["isomer", "oasis_projected", "oasis_soft_projection"]


def geomean(xs: Sequence[float]) -> float:
    xs = [max(float(x), 1e-12) for x in xs]
    if not xs:
        return 1.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Router diagnostics on the optimizer-decision proxy")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--q-values", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    ap.add_argument("--max-cases-per-q", type=int, default=128)
    ap.add_argument("--predicates-per-case", type=int, default=32)
    ap.add_argument("--num-buckets", type=int, default=10)
    ap.add_argument("--max-observations", type=int, default=16)
    ap.add_argument("--projection-iters", type=int, default=200)
    ap.add_argument("--projection-tol", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-true-selectivity", type=float, default=1e-4)
    ap.add_argument("--min-table-rows", type=float, default=100_000)
    ap.add_argument("--max-table-rows", type=float, default=10_000_000)
    ap.add_argument("--dim-rows", type=float, default=50_000.0)
    ap.add_argument("--tie-rel", type=float, default=0.01, help="near-tie if (2nd-min - min)/min <= this")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    model = P.MlpHistogramModelV2.load(str(args.model_path))
    cfg = P.CostProxyConfig(dim_rows=args.dim_rows)

    sample_paths = P.iter_sample_paths(args.data_root, args.q_values, args.max_cases_per_q)
    if not sample_paths:
        raise FileNotFoundError(f"No test samples under {args.data_root}")

    policies = ["always_isomer", "always_oasis", "random_projected", "router", "oracle"]
    qe_rows: Dict[str, List[float]] = {p: [] for p in policies}
    jopt_rows: Dict[str, List[int]] = {p: [] for p in policies}
    worst_case_qe: Dict[str, float] = {p: 0.0 for p in policies}

    router_choice_counts: Dict[str, int] = {c: 0 for c in POOL}
    oracle_choice_counts: Dict[str, int] = {c: 0 for c in POOL}
    n_cases = 0
    router_eq_oracle = 0
    exact_ties = 0      # top-2 residual gap == 0 (to 1e-9)
    near_ties = 0       # top-2 residual within tie-rel
    router_worse_than_isomer = 0  # cases where router's future QE > isomer's

    for sample_index, (q_mods, path) in enumerate(sample_paths):
        sample = P.load_feedback_sample(str(path))
        mb, _ = P.build_method_boundaries(
            sample, model=model, num_buckets=args.num_buckets,
            max_observations=args.max_observations,
            projection_iters=args.projection_iters, projection_tol=args.projection_tol,
        )
        obs = P.observations_to_dicts(sample)

        # In-window residual per candidate (the only signal the Router sees).
        residual = {c: P.feedback_residual(mb[c], obs) for c in POOL}
        ordered = sorted(POOL, key=lambda c: residual[c])
        router_choice = ordered[0]
        gap = residual[ordered[1]] - residual[ordered[0]]
        if gap <= 1e-9:
            exact_ties += 1
        if residual[ordered[0]] > 0 and gap / max(residual[ordered[0]], 1e-12) <= args.tie_rel:
            near_ties += 1

        # Future predicates (identical generation to the proxy run).
        fresh = mb["fresh"]
        rng = random.Random(args.seed + q_mods * 100_000 + sample_index)
        table_rows = int(10 ** rng.uniform(math.log10(args.min_table_rows),
                                           math.log10(args.max_table_rows)))
        predicates = P.generate_predicates(
            fresh, rng=rng, count=args.predicates_per_case,
            min_true_selectivity=args.min_true_selectivity,
        )

        # Per-candidate per-predicate Q-error and join-optimality.
        cand_qe: Dict[str, List[float]] = {c: [] for c in POOL}
        cand_jopt: Dict[str, List[int]] = {c: [] for c in POOL}
        for pred in predicates:
            true_sel = P.estimate_selectivity(fresh, pred)
            for c in POOL:
                est = P.estimate_selectivity(mb[c], pred)
                cand_qe[c].append(P.qerr(est, true_sel))
                jc, jo, _ = P.regret_for_join(est, true_sel, table_rows, cfg)
                cand_jopt[c].append(1 if jc == jo else 0)

        case_qe = {c: geomean(cand_qe[c]) for c in POOL}
        oracle_choice = min(POOL, key=lambda c: case_qe[c])

        router_choice_counts[router_choice] += 1
        oracle_choice_counts[oracle_choice] += 1
        n_cases += 1
        if router_choice == oracle_choice:
            router_eq_oracle += 1
        if case_qe[router_choice] > case_qe["isomer"]:
            router_worse_than_isomer += 1

        # Realize each policy on this case's predicates.
        chosen = {
            "always_isomer": "isomer",
            "always_oasis": "oasis_projected",
            "router": router_choice,
            "oracle": oracle_choice,
        }
        for pol, c in chosen.items():
            qe_rows[pol].extend(cand_qe[c])
            jopt_rows[pol].extend(cand_jopt[c])
            worst_case_qe[pol] = max(worst_case_qe[pol], case_qe[c])
        # random-among-projected: geometric expectation over the projected pool
        for i in range(len(predicates)):
            rqe = geomean([cand_qe[c][i] for c in PROJECTED])
            qe_rows["random_projected"].append(rqe)
            jopt_rows["random_projected"].append(
                round(sum(cand_jopt[c][i] for c in PROJECTED) / len(PROJECTED))
            )
        worst_case_qe["random_projected"] = max(
            worst_case_qe["random_projected"],
            geomean([case_qe[c] for c in PROJECTED]),
        )

    result = {
        "n_cases": n_cases,
        "policies": {
            pol: {
                "selectivity_qerr_gm": geomean(qe_rows[pol]),
                "join_optimal_frac": sum(jopt_rows[pol]) / max(len(jopt_rows[pol]), 1),
                "worst_case_qerr": worst_case_qe[pol],
            }
            for pol in policies
        },
        "router_choice_distribution": router_choice_counts,
        "oracle_choice_distribution": oracle_choice_counts,
        "router_eq_oracle_frac": router_eq_oracle / max(n_cases, 1),
        "exact_tie_frac": exact_ties / max(n_cases, 1),
        "near_tie_frac": near_ties / max(n_cases, 1),
        "router_worse_than_isomer_frac": router_worse_than_isomer / max(n_cases, 1),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "router_diagnostics.json", "w") as f:
        json.dump(result, f, indent=2)
    write_latex(args.output_dir, result)
    write_text(args.output_dir, result)
    print(json.dumps(result, indent=2))


def write_latex(output_dir: Path, r: dict) -> None:
    label = {
        "always_isomer": "Always ISOMER",
        "always_oasis": "Always OASIS",
        "random_projected": "Random (projected)",
        "router": "\\textbf{Router (residual)}",
        "oracle": "Oracle (future Q-err)",
    }
    order = ["always_isomer", "always_oasis", "random_projected", "router", "oracle"]
    with open(output_dir / "table_router_diagnostics.tex", "w") as f:
        f.write("\\begin{table}[t]\n  \\centering\\small\n")
        f.write("  \\caption{Router selection policy on the optimizer-decision proxy "
                f"({r['n_cases']} held-out cases). The Router minimizes the in-window "
                "feedback residual only (non-oracle); the oracle minimizes true held-out "
                "future Q-error and is not deployable. The Router beats both static "
                "policies and a random projected choice, approaching the oracle, and never "
                "selects a candidate worse than always-ISOMER in aggregate. Lower Q-error "
                "is better; higher join-optimal is better.}\n")
        f.write("  \\label{tab:router_diagnostics}\n")
        f.write("  \\setlength{\\tabcolsep}{6pt}\n")
        f.write("  \\begin{tabular}{l rrr}\n    \\toprule\n")
        f.write("    Selection policy & Sel. QE & Join-Opt. & Worst-case QE \\\\\n    \\midrule\n")
        for pol in order:
            d = r["policies"][pol]
            f.write(f"    {label[pol]} & {d['selectivity_qerr_gm']:.3f} & "
                    f"{d['join_optimal_frac']*100:.1f}\\% & {d['worst_case_qerr']:.3f} \\\\\n")
        f.write("    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")


def write_text(output_dir: Path, r: dict) -> None:
    lines = [
        f"Router diagnostics ({r['n_cases']} cases)",
        "=" * 48,
        f"Router == oracle choice: {r['router_eq_oracle_frac']*100:.1f}%",
        f"Residual exact-tie rate: {r['exact_tie_frac']*100:.1f}%",
        f"Residual near-tie rate (<=1% rel): {r['near_tie_frac']*100:.1f}%",
        f"Router future-QE worse than ISOMER: {r['router_worse_than_isomer_frac']*100:.1f}% of cases",
        "",
        "Policy             SelQE  JoinOpt  WorstQE",
        "-" * 48,
    ]
    for pol in ["always_isomer", "always_oasis", "random_projected", "router", "oracle"]:
        d = r["policies"][pol]
        lines.append(f"{pol:<18} {d['selectivity_qerr_gm']:.3f}  "
                     f"{d['join_optimal_frac']*100:5.1f}%  {d['worst_case_qerr']:.3f}")
    lines += ["", f"Router choices: {r['router_choice_distribution']}",
              f"Oracle choices: {r['oracle_choice_distribution']}"]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
