#!/usr/bin/env python3
"""Cross-column feedback repair vs the independence assumption (AVI).

The optimizer's #1 selectivity error is the attribute-value-independence assumption:
sel(A AND B) = sel(A)*sel(B). For correlated columns this is badly wrong. This experiment
tests whether *query feedback* on conjunctive predicates can repair the joint distribution
and beat AVI -- the regime where (unlike single-column) max-entropy is NOT already optimal.

For each real correlated column pair: build the true joint on a GxG grid; the AVI baseline
is the outer product of the TRUE marginals (best case for independence -- isolates the
correlation effect); the repaired joint is 2D IPF (the max-entropy / ISOMER analog in 2D)
seeded from AVI and projected onto K observed rectangle masses (the feedback). Both are
scored on held-out conjunctive rectangle predicates by Q-error. Reports per-pair Pearson r,
AVI Q-error, repaired Q-error, and % improvement.
"""
from __future__ import annotations

import argparse
import csv
import os
import random

import numpy as np

# (name, csv, sep, skip_header, colA, colB)
PAIRS = [
    ("bike:temp~atemp",        "data/real/hour.csv",              ",", True, 10, 11),
    ("bike:casual~cnt",        "data/real/hour.csv",              ",", True, 14, 16),
    ("wwhite:freeSO2~totSO2",  "data/real/winequality-white.csv", ";", True, 5, 6),
    ("wred:fixedAcid~pH",      "data/real/winequality-red.csv",   ";", True, 0, 8),
    ("wred:density~alcohol",   "data/real/winequality-red.csv",   ";", True, 7, 10),
    ("forest:hill9~hill3",     "data/real/covtype.data",          ",", False, 6, 8),
    ("forest:elev~roadway",    "data/real/covtype.data",          ",", False, 0, 5),
    ("census:age~hours",       "data/real/adult.data",            ",", False, 0, 12),
]


def read_pair(path, sep, skip, ca, cb):
    xs, ys = [], []
    with open(path) as f:
        if skip:
            next(f, None)
        for line in f:
            p = line.rstrip("\n").split(sep)
            if max(ca, cb) >= len(p):
                continue
            a, b = p[ca].strip(), p[cb].strip()
            if a in ("", "?", "NA", "nan") or b in ("", "?", "NA", "nan"):
                continue
            try:
                xs.append(float(a)); ys.append(float(b))
            except ValueError:
                continue
    return np.asarray(xs), np.asarray(ys)


def joint_pmf(x, y, g):
    def norm(v):
        lo, hi = np.percentile(v, [0.5, 99.5]); hi = max(hi, lo + 1e-9)
        return np.clip((v - lo) / (hi - lo), 0, 1)
    h, _, _ = np.histogram2d(norm(x), norm(y), bins=[g, g], range=[[0, 1], [0, 1]])
    return h / max(h.sum(), 1.0)


def rects(g, n, rng):
    out = []
    while len(out) < n:
        i0, i1 = sorted(rng.sample(range(g + 1), 2))
        j0, j1 = sorted(rng.sample(range(g + 1), 2))
        out.append((i0, i1, j0, j1))
    return out


def mass(P, r):
    i0, i1, j0, j1 = r
    return float(P[i0:i1, j0:j1].sum())


def ipf2d(P0, fb_rects, fb_targets, n_iter=40):
    P = P0.copy()
    for _ in range(n_iter):
        for r, t in zip(fb_rects, fb_targets):
            i0, i1, j0, j1 = r
            cur = P[i0:i1, j0:j1].sum()
            if cur > 1e-12:
                P[i0:i1, j0:j1] *= (t / cur)
        s = P.sum()
        if s > 1e-12:
            P /= s
    return P


def qerr(est, true):
    e, t = max(est, 1e-6), max(true, 1e-6)
    return max(e / t, t / e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-true", type=float, default=1e-3)
    ap.add_argument("--out", default="results/crosscol_feedback_v1")
    a = ap.parse_args()
    g = a.grid
    rng = random.Random(a.seed)

    print(f"{'pair':24} {'N':>8} {'pearson_r':>9} | {'AVI_QErr':>8} {'FB_QErr':>8} {'improve%':>8}")
    print("-" * 78)
    agg_avi, agg_fb = [], []
    rows = []
    for pi, (name, path, sep, skip, ca, cb) in enumerate(PAIRS):
        try:
            x, y = read_pair(path, sep, skip, ca, cb)
        except FileNotFoundError:
            print(f"{name:24} (file missing)"); continue
        if len(x) < 500:
            print(f"{name:24} (too few rows: {len(x)})"); continue
        r = float(np.corrcoef(x, y)[0, 1])
        P = joint_pmf(x, y, g)
        px, py = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
        avi = px @ py  # independence from TRUE marginals
        avi /= max(avi.sum(), 1e-12)
        pair_avi, pair_fb = [], []
        for t in range(a.trials):
            trng = random.Random(a.seed + 1000 * t + 7919 * pi)
            fb = rects(g, a.k, trng)
            fb_t = [mass(P, rr) for rr in fb]
            rep = ipf2d(avi, fb, fb_t)
            ev = rects(g, a.n_eval, trng)
            for rr in ev:
                true = mass(P, rr)
                if true < a.min_true:
                    continue
                pair_avi.append(qerr(mass(avi, rr), true))
                pair_fb.append(qerr(mass(rep, rr), true))
        gm_avi = float(np.exp(np.mean(np.log(pair_avi))))
        gm_fb = float(np.exp(np.mean(np.log(pair_fb))))
        impr = (gm_avi - gm_fb) / gm_avi * 100
        agg_avi += pair_avi; agg_fb += pair_fb
        rows.append((name, len(x), r, gm_avi, gm_fb, impr))
        print(f"{name:24} {len(x):>8} {r:>9.3f} | {gm_avi:>8.3f} {gm_fb:>8.3f} {impr:>7.1f}%")
    GA = float(np.exp(np.mean(np.log(agg_avi))))
    GF = float(np.exp(np.mean(np.log(agg_fb))))
    print("-" * 78)
    print(f"{'AGGREGATE':24} {'':>8} {'':>9} | {GA:>8.3f} {GF:>8.3f} {(GA-GF)/GA*100:>7.1f}%")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["pair", "n_rows", "pearson_r", "avi_qerr", "feedback_qerr", "improve_pct"])
        for nm, n, rr, av, fb, im in rows:
            w.writerow([nm, n, f"{rr:.4f}", f"{av:.4f}", f"{fb:.4f}", f"{im:.2f}"])
        w.writerow(["AGGREGATE", "", "", f"{GA:.4f}", f"{GF:.4f}", f"{(GA-GF)/GA*100:.2f}"])
    lines = [
        r"\begin{table}[t]", r"  \centering\small",
        r"  \caption{Feedback-driven joint repair versus the independence assumption (AVI) on real",
        r"  correlated column pairs. AVI uses the true marginals (best case for independence); the",
        r"  feedback estimate is a 2D max-entropy (IPF) projection onto $K{=}16$ observed conjunctive",
        r"  rectangle masses. Geometric-mean conjunctive-predicate Q-error on held-out predicates;",
        r"  the gain over AVI is concentrated on the correlated pairs (negligible at $r{\approx}0$).}",
        r"  \label{tab:crosscol_feedback}", r"  \setlength{\tabcolsep}{6pt}",
        r"  \begin{tabular}{l r r r r}", r"    \toprule",
        r"    Real column pair & Pearson $r$ & AVI & Feedback & Improv. \\", r"    \midrule",
    ]
    for nm, n, rr, av, fb, im in sorted(rows, key=lambda z: abs(z[2])):
        disp = nm.replace("_", r"\_").replace("~", "/").replace("&", r"\&")
        lines.append(f"    \\texttt{{{disp}}} & ${rr:+.2f}$ & {av:.3f} & {fb:.3f} & ${im:+.1f}\\%$ \\\\")
    lines += [
        r"    \midrule",
        f"    \\textbf{{Aggregate}} & & {GA:.3f} & {GF:.3f} & ${(GA-GF)/GA*100:+.1f}\\%$ \\\\",
        r"    \bottomrule", r"  \end{tabular}", r"\end{table}",
    ]
    with open(os.path.join(a.out, "table_crosscol_feedback.tex"), "w") as h:
        h.write("\n".join(lines) + "\n")
    print(f"wrote {a.out}/summary.csv and table_crosscol_feedback.tex")


if __name__ == "__main__":
    main()
