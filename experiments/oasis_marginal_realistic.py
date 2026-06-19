#!/usr/bin/env python3
"""Confirmatory gate with FAITHFUL marginal error (no synthetic noise).

The synthetic i.i.d. log-normal sweep (oasis_marginal_robustness.py) showed
marg800 collapsing once the marginal Q-error exceeds ~1.05. A reviewer could
object that i.i.d. per-bin noise is harsher than the structured error real OASIS
marginals carry. Here the marginals are produced by the *actual* OASIS
single-column mechanism: a 1D feedback (ISOMER) projection from a uniform stale
prior onto K observed interval masses -- the same maintenance the paper measures
in Table tab:real_singlecol. Both methods then consume these realistic marginals.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from collections import defaultdict

import numpy as np

import crosscol_feedback_experiment as base
from oasis_local_iteration import gm, ipf_with_marginals


def ipf_1d(prior, intervals, targets, n_iter=300):
    m = prior.copy()
    for _ in range(n_iter):
        for (a, b), t in zip(intervals, targets):
            cur = m[a:b].sum()
            if cur > 1e-12:
                m[a:b] *= t / cur
        s = m.sum()
        if s > 1e-12:
            m /= s
    return m


def realistic_marginal(true_m, g, K, rng, n_iter=300):
    """OASIS single-column repair: 1D ISOMER projection from uniform prior."""
    intervals, targets = [], []
    while len(intervals) < K:
        a, b = sorted(rng.sample(range(g + 1), 2))
        intervals.append((a, b))
        targets.append(float(true_m[a:b].sum()))
    return ipf_1d(np.ones(g) / g, intervals, targets, n_iter=n_iter)


def col_qerr(true_m, est_m, g, rng, n=256, min_true=1e-3):
    qs = []
    for _ in range(n):
        a, b = sorted(rng.sample(range(g + 1), 2))
        t = float(true_m[a:b].sum())
        if t >= min_true:
            qs.append(base.qerr(float(est_m[a:b].sum()), t))
    return gm(qs) if qs else float("nan")


def load_pairs(g):
    out = []
    for pi, (name, path, sep, skip, ca, cb) in enumerate(base.PAIRS):
        try:
            x, y = base.read_pair(path, sep, skip, ca, cb)
        except FileNotFoundError:
            continue
        if len(x) < 500:
            continue
        P = base.joint_pmf(x, y, g)
        out.append((pi, name, P, P.sum(1), P.sum(0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--k", type=int, default=16)            # joint feedback rectangles
    ap.add_argument("--marg-k", type=int, nargs="+", default=[16, 8, 4, 2])  # per-column feedback
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--marg-iter", type=int, default=800)
    ap.add_argument("--ipf-iter", type=int, default=40)
    ap.add_argument("--min-true", type=float, default=1e-3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results/oasis_marginal_realistic"))
    a = ap.parse_args()
    g = a.grid
    pairs = load_pairs(g)
    os.makedirs(a.out, exist_ok=True)

    pooled = defaultdict(lambda: defaultdict(list))
    colq = defaultdict(list)
    per_pair = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for mk in a.marg_k:
        for seed in a.seeds:
            for pi, name, P, px, py in pairs:
                mrng = random.Random(seed * 7 + pi * 13 + mk * 101)
                pxr = realistic_marginal(px, g, mk, mrng)
                pyr = realistic_marginal(py, g, mk, mrng)
                colq[mk].append(col_qerr(px, pxr, g, mrng))
                colq[mk].append(col_qerr(py, pyr, g, mrng))

                avi_true = px[:, None] @ py[None, :]; avi_true /= avi_true.sum()
                avi_real = pxr[:, None] @ pyr[None, :]; avi_real /= avi_real.sum()
                pxr2, pyr2 = pxr[:, None], pyr[None, :]

                for t in range(a.trials):
                    trng = random.Random(seed + 1000 * t + 7919 * pi)
                    fb = base.rects(g, a.k, trng)
                    fb_t = [base.mass(P, rr) for rr in fb]
                    ipf = base.ipf2d(avi_real, fb, fb_t, n_iter=a.ipf_iter)
                    marg = ipf_with_marginals(avi_real, fb, fb_t, pxr2, pyr2, n_iter=a.marg_iter)
                    for rr in base.rects(g, a.n_eval, trng):
                        true = base.mass(P, rr)
                        if true < a.min_true:
                            continue
                        for m, est in (("avi_true", avi_true), ("avi_real", avi_real),
                                       ("ipf40", ipf), ("marg800", marg)):
                            qe = base.qerr(base.mass(est, rr), true)
                            pooled[mk][m].append(qe)
                            per_pair[mk][m][name].append(qe)

    methods = ["avi_true", "avi_real", "ipf40", "marg800"]
    print(f"{'marg_K':>6} {'col_Qerr':>8} | " + " ".join(f"{m:>10}" for m in methods)
          + f" | {'marg_vs_ipf':>11}")
    print("-" * 78)
    rows = []
    for mk in a.marg_k:
        cq = gm([v for v in colq[mk] if v == v])
        gms = {m: gm(pooled[mk][m]) for m in methods}
        delta = (gms["ipf40"] - gms["marg800"]) / gms["ipf40"] * 100
        verdict = "marg WINS" if gms["marg800"] < gms["ipf40"] else "marg LOSES"
        print(f"{mk:>6d} {cq:>8.3f} | " + " ".join(f"{gms[m]:>10.4f}" for m in methods)
              + f" | {delta:>+9.2f}%  {verdict}")
        rows.append({"marg_k": mk, "col_qerr": cq, **gms, "marg_vs_ipf_pct": delta})

    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["marg_k", "col_qerr"] + methods + ["marg_vs_ipf_pct"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nwrote {a.out}/summary.csv")


if __name__ == "__main__":
    main()
