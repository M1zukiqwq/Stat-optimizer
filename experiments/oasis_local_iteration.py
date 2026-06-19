#!/usr/bin/env python3
"""Local-only OASIS cross-column iteration harness.

This script keeps the published cross-column experiment untouched and tries
OASIS/IPF variants that use only the same legal inputs: the AVI seed, optimizer
single-column marginals, and the observed feedback rectangle masses.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

import crosscol_feedback_experiment as base


class Progress:
    def __init__(self, total, label="", enabled=True):
        self.total = max(int(total), 1)
        self.label = label
        self.enabled = enabled
        self.done = 0
        self.start = time.monotonic()
        self.last = 0.0

    def step(self):
        if not self.enabled:
            return
        self.done += 1
        now = time.monotonic()
        if self.done < self.total and now - self.last < 0.5:
            return
        self.last = now
        frac = min(self.done / self.total, 1.0)
        width = 28
        filled = int(width * frac)
        rate = self.done / max(now - self.start, 1e-9)
        eta = (self.total - self.done) / max(rate, 1e-9)
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write(
            f"\r{self.label} [{bar}] {self.done}/{self.total} "
            f"{frac * 100:5.1f}% {rate:5.1f}/s ETA {eta:5.1f}s"
        )
        sys.stderr.flush()
        if self.done >= self.total:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _scale_marginals_inplace(P, px, py, n_iter=1):
    tx = px.reshape(-1)
    ty = py.reshape(-1)
    for _ in range(n_iter):
        rs = P.sum(axis=1)
        nz = rs > 1e-12
        P[nz, :] *= (tx[nz] / rs[nz])[:, None]
        cs = P.sum(axis=0)
        nz = cs > 1e-12
        P[:, nz] *= (ty[nz] / cs[nz])[None, :]
        s = P.sum()
        if s > 1e-12:
            P /= s
    return P


def scale_marginals(P, px, py, n_iter=1):
    return _scale_marginals_inplace(P.copy(), px, py, n_iter=n_iter)


def ipf_relaxed(P0, fb_rects, fb_targets, n_iter=40, alpha=1.0):
    P = P0.copy()
    for _ in range(n_iter):
        for r, t in zip(fb_rects, fb_targets):
            i0, i1, j0, j1 = r
            cur = P[i0:i1, j0:j1].sum()
            if cur > 1e-12:
                P[i0:i1, j0:j1] *= (t / cur) ** alpha
        s = P.sum()
        if s > 1e-12:
            P /= s
    return P


def ipf_with_marginals(P0, fb_rects, fb_targets, px, py, n_iter=40, alpha=1.0, marginal_passes=1):
    P = P0.copy()
    for _ in range(n_iter):
        for r, t in zip(fb_rects, fb_targets):
            i0, i1, j0, j1 = r
            cur = P[i0:i1, j0:j1].sum()
            if cur > 1e-12:
                P[i0:i1, j0:j1] *= (t / cur) ** alpha
        _scale_marginals_inplace(P, px, py, n_iter=marginal_passes)
    return P


def linear_mix(A, B, w):
    P = w * A + (1.0 - w) * B
    s = P.sum()
    return P / s if s > 1e-12 else P


def gm(xs):
    return float(np.exp(np.mean(np.log(xs))))


def load_pairs(g):
    out = []
    for pi, (name, path, sep, skip, ca, cb) in enumerate(base.PAIRS):
        try:
            x, y = base.read_pair(path, sep, skip, ca, cb)
        except FileNotFoundError:
            print(f"{name:24} (file missing)")
            continue
        if len(x) < 500:
            print(f"{name:24} (too few rows: {len(x)})")
            continue
        P = base.joint_pmf(x, y, g)
        px, py = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
        avi = px @ py
        avi /= max(avi.sum(), 1e-12)
        out.append((pi, name, len(x), float(np.corrcoef(x, y)[0, 1]), P, px, py, avi))
    return out


def build_tasks(pairs, args, seed):
    tasks = []
    for pi, name, n, corr, P, px, py, avi in pairs:
        for t in range(args.trials):
            trng = random.Random(seed + 1000 * t + 7919 * pi)
            fb = base.rects(args.grid, args.k, trng)
            fb_t = [base.mass(P, rr) for rr in fb]
            ev = base.rects(args.grid, args.n_eval, trng)
            evals = []
            for rr in ev:
                true = base.mass(P, rr)
                if true >= args.min_true:
                    evals.append((rr, true))
            tasks.append((name, P, px, py, avi, fb, fb_t, evals))
    return tasks


def base_specs():
    return [("avi", lambda avi, fb, fb_t, px, py: avi),
            ("oasis_ipf40", lambda avi, fb, fb_t, px, py: base.ipf2d(avi, fb, fb_t))]


def focused_specs():
    return base_specs() + [
        ("marg160", lambda avi, fb, fb_t, px, py: ipf_with_marginals(avi, fb, fb_t, px, py, n_iter=160)),
        ("marg320", lambda avi, fb, fb_t, px, py: ipf_with_marginals(avi, fb, fb_t, px, py, n_iter=320)),
        ("marg800", lambda avi, fb, fb_t, px, py: ipf_with_marginals(avi, fb, fb_t, px, py, n_iter=800)),
    ]


def variant_specs(args):
    specs = []
    for n_iter in [10, 20, 40, 80, 160]:
        specs.append((f"ipf{n_iter}", lambda avi, fb, fb_t, px, py, n=n_iter: base.ipf2d(avi, fb, fb_t, n_iter=n)))
    for alpha in [0.25, 0.5, 0.75, 1.25, 1.5]:
        specs.append((f"soft_a{alpha:g}", lambda avi, fb, fb_t, px, py, a=alpha: ipf_relaxed(avi, fb, fb_t, args.iter, a)))
    for n_iter in [20, 40, 80, 160]:
        specs.append((f"marg{n_iter}", lambda avi, fb, fb_t, px, py, n=n_iter: ipf_with_marginals(avi, fb, fb_t, px, py, n_iter=n)))
    for alpha in [0.5, 0.75, 1.0, 1.25]:
        specs.append((f"marg_a{alpha:g}", lambda avi, fb, fb_t, px, py, a=alpha: ipf_with_marginals(avi, fb, fb_t, px, py, args.iter, a)))
    for passes in [2, 4]:
        specs.append((f"marg_pass{passes}", lambda avi, fb, fb_t, px, py, p=passes: ipf_with_marginals(avi, fb, fb_t, px, py, args.iter, 1.0, p)))
    for w in [0.25, 0.5, 0.75]:
        specs.append((f"mix_ipf_avi_w{w:g}", lambda avi, fb, fb_t, px, py, w=w: linear_mix(base.ipf2d(avi, fb, fb_t), avi, w)))
    for w in [0.25, 0.5, 0.75]:
        specs.append((f"mix_marg_avi_w{w:g}", lambda avi, fb, fb_t, px, py, w=w: linear_mix(ipf_with_marginals(avi, fb, fb_t, px, py, args.iter), avi, w)))
    return specs


def evaluate(tasks, specs, progress=False, label=""):
    scores = defaultdict(list)
    per_pair = defaultdict(lambda: defaultdict(list))
    residuals = defaultdict(list)
    meter = Progress(len(tasks) * len(specs), label=label, enabled=progress)
    for name, P, px, py, avi, fb, fb_t, evals in tasks:
        for method, fn in specs:
            est = fn(avi, fb, fb_t, px, py)
            for rr, true in evals:
                qe = base.qerr(base.mass(est, rr), true)
                scores[method].append(qe)
                per_pair[method][name].append(qe)
            residuals[method].append(max(abs(base.mass(est, rr) - tt) for rr, tt in zip(fb, fb_t)))
            meter.step()
    rows = []
    for method in sorted(scores, key=lambda k: gm(scores[k])):
        rows.append({
            "method": method,
            "qerr_gm": gm(scores[method]),
            "max_feedback_resid": max(residuals[method]),
            "mean_feedback_resid": float(np.mean(residuals[method])),
            "per_pair": {name: gm(vals) for name, vals in per_pair[method].items()},
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--min-true", type=float, default=1e-3)
    ap.add_argument("--iter", type=int, default=80)
    ap.add_argument("--profile", choices=["broad", "focused"], default="broad")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results/oasis_local_iteration"))
    args = ap.parse_args()

    pairs = load_pairs(args.grid)
    specs = focused_specs() if args.profile == "focused" else base_specs() + variant_specs(args)
    os.makedirs(args.out, exist_ok=True)

    all_rows = []
    for seed in args.seeds:
        rows = evaluate(build_tasks(pairs, args, seed), specs,
                        progress=not args.no_progress, label=f"seed={seed}")
        best = rows[0]
        print(f"\nseed={seed} best={best['method']} qerr={best['qerr_gm']:.4f} "
              f"resid_max={best['max_feedback_resid']:.4g}")
        for row in rows[:12]:
            print(f"  {row['method']:<18} qerr={row['qerr_gm']:.4f} "
                  f"resid_max={row['max_feedback_resid']:.4g} resid_mean={row['mean_feedback_resid']:.4g}")
            all_rows.append({"seed": seed, **{k: v for k, v in row.items() if k != "per_pair"}})

        path = os.path.join(args.out, f"per_pair_seed{seed}.csv")
        with open(path, "w", newline="") as h:
            w = csv.writer(h)
            w.writerow(["method"] + [p[1] for p in pairs])
            for row in rows:
                w.writerow([row["method"]] + [f"{row['per_pair'].get(p[1], math.nan):.6f}" for p in pairs])

    summary_path = os.path.join(args.out, "summary.csv")
    with open(summary_path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["seed", "method", "qerr_gm", "max_feedback_resid", "mean_feedback_resid"])
        w.writeheader()
        for row in all_rows:
            w.writerow({
                "seed": row["seed"],
                "method": row["method"],
                "qerr_gm": f"{row['qerr_gm']:.6f}",
                "max_feedback_resid": f"{row['max_feedback_resid']:.6g}",
                "mean_feedback_resid": f"{row['mean_feedback_resid']:.6g}",
            })
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
