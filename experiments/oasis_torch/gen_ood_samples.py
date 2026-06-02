"""Generate OOD-drift-family samples in the gate-compatible JSON schema.

The v3 model is trained only on compound drift; this produces held-out
deployment-inspired drift families (batch load, range shift, skew evolution,
outlier burst, multimodal, seasonal) so gate_eval can test family-level
generalization. Output mirrors generate_synthetic_json_dataset.py's schema.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "cdf_kll_ml_pipeline"))

from extended_drift_generators import ExtendedMemoryTable
from histogram_math import evaluate_piecewise_cdf, project_monotonic

PREDICATES = ["<", "<=", ">", ">=", "BETWEEN"]
FAMILIES = ["batch_load", "range_shift", "skew_evol", "outlier", "multimodal", "seasonal"]


def boundaries(table, B):
    levels = [i / B for i in range(B + 1)]
    bx = table.get_quantiles(levels)
    bx[0], bx[-1] = 0.0, 1.0
    inner = project_monotonic([min(max(v, 0.0), 1.0) for v in bx[1:-1]])
    return [0.0] + inner + [1.0]


def piecewise(bx):
    B = len(bx) - 1
    return bx, [i / B for i in range(B + 1)]


def apply_family(table, fam, rng, q):
    if fam == "batch_load":
        table.apply_batch_load_drift(rng, q)
    elif fam == "range_shift":
        table.apply_range_shift(rng, q)
    elif fam == "skew_evol":
        table.apply_skew_evolution(rng, q)
    elif fam == "outlier":
        table.apply_outlier_burst(rng, q)
    elif fam == "multimodal":
        table.apply_multimodal_drift(rng, q)
    elif fam == "seasonal":
        table.apply_seasonal_drift(rng, q)


def build_case(seed, fam, q, B, n0=4000):
    rng = random.Random(seed)
    np.random.seed(seed % (2**31))
    init = [min(max(np.random.normal(0.5, 0.16), 0.0), 1.0) for _ in range(n0)]
    table = ExtendedMemoryTable(list(init), 0, 0.0, 1.0)
    prior_b = boundaries(table, B)
    px, pp = piecewise(prior_b)
    apply_family(table, fam, rng, q)
    true_b = boundaries(table, B)
    obs = []
    for k in range(16):
        pred = rng.choice(PREDICATES)
        v = rng.uniform(0.02, 0.98); vu = None
        if pred == "BETWEEN":
            v2 = rng.uniform(0.02, 0.98); v, vu = sorted((v, v2))
            est = max(0.0, evaluate_piecewise_cdf(px, pp, vu) - evaluate_piecewise_cdf(px, pp, v))
            act = table.query_conditional_sel("BETWEEN", v, vu)
        elif pred in ("<", "<="):
            est = evaluate_piecewise_cdf(px, pp, v); act = table.query_conditional_sel(pred, v)
        else:
            est = 1.0 - evaluate_piecewise_cdf(px, pp, v); act = table.query_conditional_sel(pred, v)
        o = {"predicate_type": pred, "value": round(v, 6),
             "estimated_sel": round(min(max(est, 0), 1), 6),
             "actual_sel": round(min(max(act, 0), 1), 6),
             "timestamp": "2026-01-01T00:00:00Z"}
        if vu is not None:
            o["value_upper"] = round(vu, 6)
        obs.append(o)
    levels = [round(i / B, 6) for i in range(1, B)]
    return {
        "prior_kll": {"type": "double", "k": 1024, "min": 0.0, "max": 1.0, "null_fraction": 0.0,
                      "quantile_levels": levels, "quantile_values": [round(v, 6) for v in prior_b[1:-1]],
                      "bucket_boundaries": [round(v, 6) for v in prior_b]},
        "observations": obs,
        "corrected_kll": {"type": "double", "k": 1024, "quantile_levels": levels,
                          "quantile_values": [round(v, 6) for v in true_b[1:-1]],
                          "bucket_boundaries": [round(v, 6) for v in true_b]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--q-values", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--cases-per-family", type=int, default=22)
    ap.add_argument("--num-buckets", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--split", choices=["train", "test"], default="test")
    a = ap.parse_args()
    fams = a.families
    total = 0
    for q in a.q_values:
        d = os.path.join(a.out, f"{a.split}_q{q}")
        os.makedirs(d, exist_ok=True)
        idx = 0
        for fam in fams:
            for c in range(a.cases_per_family):
                seed = a.seed + q * 10007 + hash(fam) % 1000 + c
                try:
                    case = build_case(seed, fam, q, a.num_buckets)
                except Exception as e:
                    continue
                json.dump(case, open(os.path.join(d, f"sim_case_{idx:04d}.json"), "w"))
                idx += 1; total += 1
    print(f"generated {total} OOD-family samples under {a.out}")


if __name__ == "__main__":
    main()
