"""Preprocess cached OASIS drift JSON into stacked tensors for v3 training.

Produces, per sample: stale/fresh equi-depth boundaries (B+1), padded feedback
observation features + interval constraints (type, lo, hi, target), and a set of
held-out future predicates drawn from the FRESH distribution (lo, hi, true_sel).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from typing import List, Tuple

import numpy as np
import torch

PREDICATE_ORDER = ["<", "<=", ">", ">=", "BETWEEN", "="]
PIDX = {p: i for i, p in enumerate(PREDICATE_ORDER)}
N_PRED = len(PREDICATE_ORDER)
OBS_FEAT_DIM = N_PRED + 6


def _mono01(vals: List[float]) -> List[float]:
    out = [min(max(float(v), 0.0), 1.0) for v in vals]
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def boundaries(qvals: List[float]) -> np.ndarray:
    return np.array([0.0] + _mono01(list(qvals)) + [1.0], dtype=np.float64)


def cdf_np(bx: np.ndarray, levels: np.ndarray, x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    if x <= bx[0]:
        return float(levels[0])
    if x >= bx[-1]:
        return float(levels[-1])
    i = int(np.searchsorted(bx, x, side="right"))
    i = min(max(i, 1), len(bx) - 1)
    x0, x1 = bx[i - 1], bx[i]
    p0, p1 = levels[i - 1], levels[i]
    if x1 == x0:
        return float(p1)
    return float(p0 + (x - x0) / (x1 - x0) * (p1 - p0))


def inv_cdf_np(bx: np.ndarray, levels: np.ndarray, p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    if p <= levels[0]:
        return float(bx[0])
    if p >= levels[-1]:
        return float(bx[-1])
    i = int(np.searchsorted(levels, p, side="left"))
    i = min(max(i, 1), len(levels) - 1)
    p0, p1 = levels[i - 1], levels[i]
    x0, x1 = bx[i - 1], bx[i]
    if p1 == p0:
        return float(x1)
    return float(x0 + (p - p0) / (p1 - p0) * (x1 - x0))


def pred_interval(ptype: str, value: float, value_upper):
    if ptype in ("<", "<="):
        return 0.0, float(value)
    if ptype in (">", ">="):
        return float(value), 1.0
    if ptype == "BETWEEN":
        lo, hi = sorted((float(value), float(value_upper)))
        return lo, hi
    return max(0.0, float(value) - 0.005), min(1.0, float(value) + 0.005)


def make_future(fresh_bx: np.ndarray, levels: np.ndarray, n: int, rng: random.Random):
    los, his, trues = [], [], []
    tries = 0
    while len(los) < n and tries < n * 40:
        tries += 1
        t = rng.choices(["<=", ">=", "BETWEEN"], weights=[0.34, 0.34, 0.32])[0]
        if t == "BETWEEN":
            w = 10 ** rng.uniform(np.log10(0.01), np.log10(0.45))
            lp = rng.uniform(0.02, max(0.03, 0.98 - w)); hp = min(0.99, lp + w)
            lo = inv_cdf_np(fresh_bx, levels, lp); hi = inv_cdf_np(fresh_bx, levels, hp)
        elif t == "<=":
            lo = 0.0; hi = inv_cdf_np(fresh_bx, levels, rng.uniform(0.005, 0.95))
        else:
            lo = inv_cdf_np(fresh_bx, levels, rng.uniform(0.05, 0.995)); hi = 1.0
        true = cdf_np(fresh_bx, levels, hi) - cdf_np(fresh_bx, levels, lo)
        if true >= 1e-4:
            los.append(lo); his.append(hi); trues.append(true)
    while len(los) < n:  # pad by repeat
        los.append(los[-1] if los else 0.0); his.append(his[-1] if his else 1.0)
        trues.append(trues[-1] if trues else 1.0)
    return los, his, trues


def build(files: List[str], num_buckets: int, max_obs: int, n_future: int, seed: int):
    levels = np.linspace(0, 1, num_buckets + 1)
    rng = random.Random(seed)
    S = {k: [] for k in ["stale", "fresh", "obs_feat", "obs_mask", "obs_lo", "obs_hi",
                         "obs_pidx", "obs_tgt", "fut_lo", "fut_hi", "fut_true"]}
    kept = 0
    for path in files:
        try:
            d = json.load(open(path))
        except Exception:
            continue
        prior = d.get("prior_kll", {}); corr = d.get("corrected_kll", {})
        if "quantile_values" not in prior or "quantile_values" not in corr:
            continue
        stale = boundaries(prior["quantile_values"])
        fresh = boundaries(corr["quantile_values"])
        if len(stale) != num_buckets + 1 or len(fresh) != num_buckets + 1:
            continue
        obs = d.get("observations", [])[:max_obs]
        feat = np.zeros((max_obs, OBS_FEAT_DIM), np.float32)
        mask = np.zeros((max_obs,), np.float32)
        olo = np.zeros((max_obs,), np.float32); ohi = np.ones((max_obs,), np.float32)
        opidx = np.zeros((max_obs,), np.int64); otgt = np.zeros((max_obs,), np.float32)
        for i, o in enumerate(obs):
            pt = o["predicate_type"]; v = float(o["value"])
            vu = o.get("value_upper"); has_up = 1.0 if vu is not None else 0.0
            est = float(o.get("estimated_sel", 0.0)); act = float(o.get("actual_sel", 0.0))
            lo, hi = pred_interval(pt, v, vu if vu is not None else v)
            feat[i, PIDX.get(pt, 1)] = 1.0
            feat[i, N_PRED:] = [v, (vu if vu is not None else v), est, act, est - act, has_up]
            mask[i] = 1.0; olo[i] = lo; ohi[i] = hi
            opidx[i] = PIDX.get(pt, 1); otgt[i] = min(max(act, 1e-4), 1 - 1e-4)
        flo, fhi, ftrue = make_future(fresh, levels, n_future, rng)
        S["stale"].append(stale); S["fresh"].append(fresh)
        S["obs_feat"].append(feat); S["obs_mask"].append(mask)
        S["obs_lo"].append(olo); S["obs_hi"].append(ohi)
        S["obs_pidx"].append(opidx); S["obs_tgt"].append(otgt)
        S["fut_lo"].append(flo); S["fut_hi"].append(fhi); S["fut_true"].append(ftrue)
        kept += 1
    T = {
        "stale": torch.tensor(np.array(S["stale"]), dtype=torch.float32),
        "fresh": torch.tensor(np.array(S["fresh"]), dtype=torch.float32),
        "obs_feat": torch.tensor(np.array(S["obs_feat"]), dtype=torch.float32),
        "obs_mask": torch.tensor(np.array(S["obs_mask"]), dtype=torch.float32),
        "obs_lo": torch.tensor(np.array(S["obs_lo"]), dtype=torch.float32),
        "obs_hi": torch.tensor(np.array(S["obs_hi"]), dtype=torch.float32),
        "obs_tgt": torch.tensor(np.array(S["obs_tgt"]), dtype=torch.float32),
        "fut_lo": torch.tensor(np.array(S["fut_lo"]), dtype=torch.float32),
        "fut_hi": torch.tensor(np.array(S["fut_hi"]), dtype=torch.float32),
        "fut_true": torch.tensor(np.array(S["fut_true"]), dtype=torch.float32),
    }
    return T, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-buckets", type=int, default=10)
    ap.add_argument("--max-obs", type=int, default=16)
    ap.add_argument("--n-future", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra-train-roots", nargs="*", default=[],
                    help="Extra roots (e.g. OOD-family dirs) whose train_q*/ and test_q*/ "
                         "samples are all added to TRAINING (domain randomization).")
    a = ap.parse_args()
    train_files = sorted(glob.glob(os.path.join(a.data_root, "train_q*", "*.json")))
    val_files = sorted(glob.glob(os.path.join(a.data_root, "test_q*", "*.json")))
    for root in a.extra_train_roots:
        train_files += sorted(glob.glob(os.path.join(root, "train_q*", "*.json")))
        train_files += sorted(glob.glob(os.path.join(root, "test_q*", "*.json")))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tr, ntr = build(train_files, a.num_buckets, a.max_obs, a.n_future, a.seed)
    va, nva = build(val_files, a.num_buckets, a.max_obs, a.n_future, a.seed + 1)
    torch.save({"train": tr, "val": va,
                "meta": {"num_buckets": a.num_buckets, "max_obs": a.max_obs}}, a.out)
    print(f"saved {a.out}: train={ntr} val={nva}")


if __name__ == "__main__":
    main()
