#!/usr/bin/env python3
"""Decisive gate: does marginal-preserving IPF survive *realistic* marginal error?

The cross-column win of `ipf_with_marginals` (marg800) over the published
`ipf2d` (oasis_ipf40) was measured with the *true* single-column marginals as the
hard constraint. In deployment the marginals are maintained by OASIS single-column
repair and carry their own Q-error (paper Table tab:real_singlecol: ~1.10-1.18 at
K=16, up to ~1.36-1.47 at K=2). This script injects multiplicative log-normal noise
into the marginals, reports the resulting *marginal* Q-error (so we can locate the
realistic band), and asks whether marg800 still beats ipf40 when BOTH consume the
SAME noisy marginals. Feedback rectangle masses stay exact (they are real observed
counts). If marg800 crosses above ipf40 inside the realistic band, the method
cannot replace the current one.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

import crosscol_feedback_experiment as base
from oasis_local_iteration import gm, ipf_with_marginals


def perturb(p_flat, sigma, rng):
    if sigma <= 0:
        return p_flat.copy()
    q = p_flat * np.exp(rng.normal(0.0, sigma, size=p_flat.shape))
    q = np.clip(q, 1e-12, None)
    return q / q.sum()


def marg_qerr(p_true, p_noisy, g, rng, n=256, min_true=1e-3):
    qs = []
    for _ in range(n):
        a, b = sorted(rng.choice(g + 1, size=2, replace=False))
        t = float(p_true[a:b].sum())
        if t < min_true:
            continue
        qs.append(base.qerr(float(p_noisy[a:b].sum()), t))
    return gm(qs) if qs else float("nan")


def load_pairs(g):
    out = []
    for pi, (name, path, sep, skip, ca, cb) in enumerate(base.PAIRS):
        try:
            x, y = base.read_pair(path, sep, skip, ca, cb)
        except FileNotFoundError:
            print(f"{name:24} (file missing)")
            continue
        if len(x) < 500:
            continue
        P = base.joint_pmf(x, y, g)
        px = P.sum(1)            # (g,) true row marginal
        py = P.sum(0)            # (g,) true col marginal
        out.append((pi, name, P, px, py))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8])
    ap.add_argument("--marg-iter", type=int, default=800)
    ap.add_argument("--ipf-iter", type=int, default=40)
    ap.add_argument("--min-true", type=float, default=1e-3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "results/oasis_marginal_robustness"))
    a = ap.parse_args()
    g = a.grid
    pairs = load_pairs(g)
    os.makedirs(a.out, exist_ok=True)

    # pooled q-errors keyed by (sigma, method); marginal q-error keyed by sigma
    pooled = defaultdict(lambda: defaultdict(list))
    marg_q = defaultdict(list)
    # per-pair pooled for the survivability breakdown, keyed by (sigma, method, pair)
    per_pair = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for sigma in a.sigmas:
        for seed in a.seeds:
            for pi, name, P, px, py in pairs:
                nrng = np.random.default_rng(seed * 100003 + pi * 31 + int(sigma * 1000))
                pxn = perturb(px, sigma, nrng)
                pyn = perturb(py, sigma, nrng)
                marg_q[sigma].append(marg_qerr(px, pxn, g, nrng))
                marg_q[sigma].append(marg_qerr(py, pyn, g, nrng))

                avi_true = (px[:, None] @ py[None, :]); avi_true /= avi_true.sum()
                avi_noisy = (pxn[:, None] @ pyn[None, :]); avi_noisy /= avi_noisy.sum()
                pxn2, pyn2 = pxn[:, None], pyn[None, :]

                import random as _r
                for t in range(a.trials):
                    trng = _r.Random(seed + 1000 * t + 7919 * pi)
                    fb = base.rects(g, a.k, trng)
                    fb_t = [base.mass(P, rr) for rr in fb]
                    # CURRENT method: ipf seeded from realistic (noisy) marginals
                    ipf = base.ipf2d(avi_noisy, fb, fb_t, n_iter=a.ipf_iter)
                    # NEW method: same noisy marginals as seed AND hard constraint
                    marg = ipf_with_marginals(avi_noisy, fb, fb_t, pxn2, pyn2, n_iter=a.marg_iter)
                    ev = base.rects(g, a.n_eval, trng)
                    for rr in ev:
                        true = base.mass(P, rr)
                        if true < a.min_true:
                            continue
                        for m, est in (("avi_true", avi_true), ("avi_noisy", avi_noisy),
                                       ("ipf40", ipf), ("marg800", marg)):
                            qe = base.qerr(base.mass(est, rr), true)
                            pooled[sigma][m].append(qe)
                            per_pair[sigma][m][name].append(qe)

    methods = ["avi_true", "avi_noisy", "ipf40", "marg800"]
    print(f"{'sigma':>6} {'marg_Qerr':>9} | " + " ".join(f"{m:>10}" for m in methods)
          + f" | {'marg_vs_ipf':>11}")
    print("-" * 78)
    rows = []
    for sigma in a.sigmas:
        mq = gm([v for v in marg_q[sigma] if v == v])
        gms = {m: gm(pooled[sigma][m]) for m in methods}
        delta = (gms["ipf40"] - gms["marg800"]) / gms["ipf40"] * 100
        verdict = "marg WINS" if gms["marg800"] < gms["ipf40"] else "marg LOSES"
        print(f"{sigma:>6.2f} {mq:>9.3f} | " + " ".join(f"{gms[m]:>10.4f}" for m in methods)
              + f" | {delta:>+9.2f}%  {verdict}")
        rows.append({"sigma": sigma, "marg_qerr": mq, **gms,
                     "marg_vs_ipf_pct": delta})

    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["sigma", "marg_qerr"] + methods + ["marg_vs_ipf_pct"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # per-pair survivability at each sigma: does marg ever lose to ipf on a pair?
    with open(os.path.join(a.out, "per_pair.csv"), "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["sigma", "pair", "avi_true", "ipf40", "marg800", "marg_beats_ipf"])
        for sigma in a.sigmas:
            for _, name, *_ in pairs:
                av = gm(per_pair[sigma]["avi_true"][name])
                ip = gm(per_pair[sigma]["ipf40"][name])
                mg = gm(per_pair[sigma]["marg800"][name])
                w.writerow([f"{sigma:.2f}", name, f"{av:.4f}", f"{ip:.4f}",
                            f"{mg:.4f}", int(mg < ip)])
    print(f"\nwrote {a.out}/summary.csv and per_pair.csv")


if __name__ == "__main__":
    main()
