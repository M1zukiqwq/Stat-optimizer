"""Gate evaluation for OASIS Stage-1 v3.

Apples-to-apples with Exp 3: apply the REAL correct_isomer projection on the v3
model prior, compare held-out future-predicate geomean Q-error against ISOMER
(project from stale), STHoles-init, QuickSel-init, across feedback caps K.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for p in [str(_REPO / "cdf_kll_ml_pipeline"), str(_REPO / "experiments")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from json_histogram_parser import load_feedback_sample
from baselines import correct_stholes_tree
from modern_baselines import correct_quicksel_h
from optimizer_decision_proxy_experiment import (
    boundaries_from_quantiles, isomer_boundaries, observations_to_dicts,
    estimate_selectivity, generate_predicates, qerr, geomean)

from model import OasisTorchV3, boundaries_from_logits

PREDICATE_ORDER = ["<", "<=", ">", ">=", "BETWEEN", "="]
PIDX = {p: i for i, p in enumerate(PREDICATE_ORDER)}
N_PRED = len(PREDICATE_ORDER)
OBS_FEAT_DIM = N_PRED + 6


def pred_interval(ptype, value, value_upper):
    if ptype in ("<", "<="):
        return 0.0, float(value)
    if ptype in (">", ">="):
        return float(value), 1.0
    if ptype == "BETWEEN":
        lo, hi = sorted((float(value), float(value_upper)))
        return lo, hi
    return max(0.0, float(value) - 0.005), min(1.0, float(value) + 0.005)


def model_prior(model, obs_subset, stale_boundaries, B, K, device):
    feat = np.zeros((1, K, OBS_FEAT_DIM), np.float32)
    mask = np.zeros((1, K), np.float32)
    for i, o in enumerate(obs_subset[:K]):
        pt = o["predicate_type"]; v = float(o["value"])
        vu = o.get("value_upper"); has_up = 1.0 if vu is not None else 0.0
        est = float(o.get("estimated_sel", 0.0)); act = float(o.get("actual_sel", 0.0))
        feat[0, i, PIDX.get(pt, 1)] = 1.0
        feat[0, i, N_PRED:] = [v, (vu if vu is not None else v), est, act, est - act, has_up]
        mask[0, i] = 1.0
    sb = torch.tensor(np.array([stale_boundaries], np.float32), device=device)
    with torch.no_grad():
        logits = model(torch.tensor(feat, device=device), torch.tensor(mask, device=device), sb)
        bx = boundaries_from_logits(logits)[0].cpu().numpy().tolist()
    return boundaries_from_quantiles(bx[1:-1])


def gmean_qerr(methods_bounds, predicates, fresh_bounds):
    out = {}
    for name, bounds in methods_bounds.items():
        qs = []
        for pred in predicates:
            true = estimate_selectivity(fresh_bounds, pred)
            est = estimate_selectivity(bounds, pred)
            qs.append(qerr(est, true))
        out[name] = geomean(qs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--q-values", type=int, nargs="+", default=[1, 3, 5, 10, 15, 20, 25, 30])
    ap.add_argument("--max-cases-per-q", type=int, default=128)
    ap.add_argument("--kcaps", type=int, nargs="+", default=[2, 6, 16])
    ap.add_argument("--preds-per-case", type=int, default=64)
    ap.add_argument("--num-buckets", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(a.ckpt, map_location=device); cfg = ck["config"]
    model = OasisTorchV3(num_buckets=cfg["num_buckets"], max_obs=cfg["max_obs"],
                         d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                         n_layers=cfg["n_layers"], residual_prior=cfg["residual_prior"]).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    B = a.num_buckets

    # acc[(kcap, q)][method] -> list of per-case geomean qerr
    acc = defaultdict(lambda: defaultdict(list))
    for q in a.q_values:
        files = sorted(glob.glob(os.path.join(a.data_root, f"test_q{q}", "*.json")))[:a.max_cases_per_q]
        for ci, path in enumerate(files):
            sample = load_feedback_sample(path)
            observations = observations_to_dicts(sample)
            stale = boundaries_from_quantiles(sample.prior.quantile_values)
            fresh = boundaries_from_quantiles(sample.corrected_quantile_values or sample.prior.quantile_values)
            stale_inner = list(stale[1:-1])
            rng = random.Random(a.seed + q * 100003 + ci)
            preds = generate_predicates(fresh, rng, a.preds_per_case, 1e-4)
            for kcap in a.kcaps:
                obs = observations[:kcap]
                if not obs:
                    continue
                isomer = isomer_boundaries(stale, obs, B)
                try:
                    sth = boundaries_from_quantiles(correct_stholes_tree(0.0, 1.0, stale_inner, obs, num_buckets=B))
                    sth_proj = isomer_boundaries(sth, obs, B)
                except Exception:
                    sth_proj = isomer
                try:
                    qs = boundaries_from_quantiles(correct_quicksel_h(0.0, 1.0, stale_inner, obs, num_buckets=B))
                    qs_proj = isomer_boundaries(qs, obs, B)
                except Exception:
                    qs_proj = isomer
                mp = model_prior(model, obs, stale, B, cfg["max_obs"], device)
                mp_proj = isomer_boundaries(mp, obs, B)
                methods = {"stale": stale, "isomer": isomer, "stholes": sth_proj,
                           "quicksel": qs_proj, "v3_proj": mp_proj, "v3_raw": mp, "fresh": fresh}
                g = gmean_qerr(methods, preds, fresh)
                for k, v in g.items():
                    acc[(kcap, q)][k].append(v)

    methods = ["stale", "isomer", "stholes", "quicksel", "v3_raw", "v3_proj", "fresh"]
    print("\n==== GATE EVAL: future-predicate geomean Q-error (real projection) ====")
    for kcap in a.kcaps:
        print(f"\n--- feedback K={kcap} ---")
        print("q     " + "  ".join(f"{m:>8s}" for m in methods) + "   v3vsISO")
        allcase = defaultdict(list)
        for q in a.q_values:
            row = acc[(kcap, q)]
            if not row:
                continue
            vals = {m: geomean(row[m]) for m in methods}
            for m in methods:
                allcase[m].extend(row[m])
            d = (vals["isomer"] - vals["v3_proj"]) / vals["isomer"] * 100
            print(f"{q:<5d} " + "  ".join(f"{vals[m]:8.3f}" for m in methods) + f"   {d:+6.1f}%")
        agg = {m: geomean(allcase[m]) for m in methods}
        d = (agg["isomer"] - agg["v3_proj"]) / agg["isomer"] * 100
        winS = (agg["stholes"] - agg["v3_proj"]) / agg["stholes"] * 100
        winQ = (agg["quicksel"] - agg["v3_proj"]) / agg["quicksel"] * 100
        print(f"ALL   " + "  ".join(f"{agg[m]:8.3f}" for m in methods) + f"   {d:+6.1f}%")
        print(f"  v3_proj vs ISOMER {d:+.1f}% | vs STHoles {winS:+.1f}% | vs QuickSel {winQ:+.1f}%")
        passed = agg["v3_proj"] < agg["isomer"] and agg["v3_proj"] <= min(agg["stholes"], agg["quicksel"]) + 1e-9
        print(f"  GATE(K={kcap}) primary: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
