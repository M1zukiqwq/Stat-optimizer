#!/usr/bin/env python3
"""Cross-column feedback repair vs the independence assumption (AVI).

The optimizer's #1 selectivity error is the attribute-value-independence assumption:
sel(A AND B) = sel(A)*sel(B). For correlated columns this is badly wrong. This experiment
tests whether *query feedback* on conjunctive predicates can repair the joint distribution
and beat AVI -- the regime where (unlike single-column) max-entropy is NOT already optimal.

For each real correlated column pair: build the true joint on a GxG grid; the AVI baseline
is the outer product of the TRUE marginals (best case for independence -- isolates the
correlation effect); STHoles is learned from the feedback only, starting from a uniform
root bucket; the repaired joint is 2D IPF (the max-entropy / ISOMER analog in 2D) seeded
from AVI and projected onto K observed rectangle masses (the feedback). All estimates are
scored on held-out conjunctive rectangle predicates by Q-error. Reports per-pair Pearson r,
AVI Q-error, STHoles Q-error, repaired Q-error, and % improvement.
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


def _area(r):
    i0, i1, j0, j1 = r
    return max(0, i1 - i0) * max(0, j1 - j0)


def _intersect(a, b):
    ai0, ai1, aj0, aj1 = a
    bi0, bi1, bj0, bj1 = b
    r = (max(ai0, bi0), min(ai1, bi1), max(aj0, bj0), min(aj1, bj1))
    return r if _area(r) > 0 else None


def _split_bucket(rect, freq, cut):
    hit = _intersect(rect, cut)
    if hit is None or hit == rect:
        return [(rect, freq)]
    i0, i1, j0, j1 = rect
    hi0, hi1, hj0, hj1 = hit
    pieces = [hit]
    if i0 < hi0:
        pieces.append((i0, hi0, j0, j1))
    if hi1 < i1:
        pieces.append((hi1, i1, j0, j1))
    if j0 < hj0:
        pieces.append((hi0, hi1, j0, hj0))
    if hj1 < j1:
        pieces.append((hi0, hi1, hj1, j1))
    a0 = _area(rect)
    return [(p, freq * _area(p) / a0) for p in pieces if _area(p) > 0]


def _render_buckets(g, buckets):
    P = np.zeros((g, g), dtype=float)
    for r, freq in buckets:
        i0, i1, j0, j1 = r
        a = _area(r)
        if a > 0 and freq > 0:
            P[i0:i1, j0:j1] += freq / a
    s = P.sum()
    if s > 1e-12:
        P /= s
    return P


def _merge_penalty(a, b):
    ra, fa = a
    rb, fb = b
    ai0, ai1, aj0, aj1 = ra
    bi0, bi1, bj0, bj1 = rb
    ui0, ui1 = min(ai0, bi0), max(ai1, bi1)
    uj0, uj1 = min(aj0, bj0), max(aj1, bj1)
    u = (ui0, ui1, uj0, uj1)
    aa, ab, au = _area(ra), _area(rb), _area(u)
    if aa + ab != au:
        return None
    da, db, du = fa / aa, fb / ab, (fa + fb) / au
    return aa * (da - du) ** 2 + ab * (db - du) ** 2, u


def _merge_to_budget(buckets, budget):
    buckets = list(buckets)
    while len(buckets) > budget:
        best = None
        for i in range(len(buckets)):
            for j in range(i + 1, len(buckets)):
                cand = _merge_penalty(buckets[i], buckets[j])
                if cand is None:
                    continue
                pen, u = cand
                if best is None or pen < best[0]:
                    best = (pen, i, j, u)
        if best is None:
            break
        _, i, j, u = best
        freq = buckets[i][1] + buckets[j][1]
        keep = [b for k, b in enumerate(buckets) if k not in (i, j)]
        keep.append((u, freq))
        buckets = keep
    return buckets


def _fit_feedback_buckets(buckets, fb_rects, fb_targets, n_iter=100):
    buckets = [(r, max(float(f), 0.0)) for r, f in buckets]
    for _ in range(n_iter):
        for q, target in zip(fb_rects, fb_targets):
            inside = [i for i, (r, _) in enumerate(buckets) if _intersect(r, q) == r]
            cur = sum(buckets[i][1] for i in inside)
            if not inside:
                continue
            if cur > 1e-12:
                scale = target / cur
                for i in inside:
                    r, f = buckets[i]
                    buckets[i] = (r, f * scale)
            else:
                share = target / len(inside)
                for i in inside:
                    r, _ = buckets[i]
                    buckets[i] = (r, share)
        s = sum(f for _, f in buckets)
        if s > 1e-12:
            buckets = [(r, f / s) for r, f in buckets]
    return buckets


def sthole2d(g, fb_rects, fb_targets, budget=96):
    budget = max(1, int(budget))
    buckets = [((0, g, 0, g), 1.0)]
    for pos, (q, target) in enumerate(zip(fb_rects, fb_targets)):
        split = []
        for r, f in buckets:
            split.extend(_split_bucket(r, f, q))
        buckets = split
        inside = [i for i, (r, _) in enumerate(buckets) if _intersect(r, q) == r]
        cur = sum(buckets[i][1] for i in inside)
        out = max(1.0 - cur, 0.0)
        for i, (r, f) in enumerate(buckets):
            if i in inside:
                nf = f * target / cur if cur > 1e-12 else target / max(len(inside), 1)
            else:
                nf = f * (1.0 - target) / out if out > 1e-12 else 0.0
            buckets[i] = (r, max(nf, 0.0))
        buckets = _fit_feedback_buckets(buckets, fb_rects[:pos + 1], fb_targets[:pos + 1], n_iter=4)
        buckets = _merge_to_budget(buckets, budget)
    buckets = _fit_feedback_buckets(buckets, fb_rects, fb_targets)
    return _render_buckets(g, buckets)


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
    ap.add_argument("--sth-budget", type=int, default=96)
    ap.add_argument("--sth-check-tol", type=float, default=2e-2)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results/crosscol_feedback_v1"))
    a = ap.parse_args()
    g = a.grid
    rng = random.Random(a.seed)

    print(f"{'pair':24} {'N':>8} {'pearson_r':>9} | {'AVI_QErr':>8} {'STH_QErr':>8} {'FB_QErr':>8} {'improve%':>8}")
    print("-" * 89)
    agg_avi, agg_sth, agg_fb = [], [], []
    rows = []
    max_sth_resid = 0.0
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
        pair_avi, pair_sth, pair_fb = [], [], []
        for t in range(a.trials):
            trng = random.Random(a.seed + 1000 * t + 7919 * pi)
            fb = rects(g, a.k, trng)
            fb_t = [mass(P, rr) for rr in fb]
            rep = ipf2d(avi, fb, fb_t)
            sth = sthole2d(g, fb, fb_t, budget=a.sth_budget)
            max_sth_resid = max(max_sth_resid, max(abs(mass(sth, rr) - tt) for rr, tt in zip(fb, fb_t)))
            ev = rects(g, a.n_eval, trng)
            for rr in ev:
                true = mass(P, rr)
                if true < a.min_true:
                    continue
                pair_avi.append(qerr(mass(avi, rr), true))
                pair_sth.append(qerr(mass(sth, rr), true))
                pair_fb.append(qerr(mass(rep, rr), true))
        gm_avi = float(np.exp(np.mean(np.log(pair_avi))))
        gm_sth = float(np.exp(np.mean(np.log(pair_sth))))
        gm_fb = float(np.exp(np.mean(np.log(pair_fb))))
        impr = (gm_avi - gm_fb) / gm_avi * 100
        agg_avi += pair_avi; agg_sth += pair_sth; agg_fb += pair_fb
        rows.append((name, len(x), r, gm_avi, gm_sth, gm_fb, impr))
        print(f"{name:24} {len(x):>8} {r:>9.3f} | {gm_avi:>8.3f} {gm_sth:>8.3f} {gm_fb:>8.3f} {impr:>7.1f}%")
    GA = float(np.exp(np.mean(np.log(agg_avi))))
    GS = float(np.exp(np.mean(np.log(agg_sth))))
    GF = float(np.exp(np.mean(np.log(agg_fb))))
    print("-" * 89)
    print(f"{'AGGREGATE':24} {'':>8} {'':>9} | {GA:>8.3f} {GS:>8.3f} {GF:>8.3f} {(GA-GF)/GA*100:>7.1f}%")
    print(f"max STHoles feedback residual: {max_sth_resid:.4g}")
    if max_sth_resid > a.sth_check_tol:
        print(f"warning: STHoles residual exceeds --sth-check-tol={a.sth_check_tol:g}; budgeted merges can relax old feedback constraints")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["pair", "n_rows", "pearson_r", "avi_qerr", "sthole_qerr", "feedback_qerr", "improve_pct"])
        for nm, n, rr, av, sth, fb, im in rows:
            w.writerow([nm, n, f"{rr:.4f}", f"{av:.4f}", f"{sth:.4f}", f"{fb:.4f}", f"{im:.2f}"])
        w.writerow(["AGGREGATE", "", "", f"{GA:.4f}", f"{GS:.4f}", f"{GF:.4f}", f"{(GA-GF)/GA*100:.2f}"])
    lines = [
        r"\begin{table}[t]", r"  \centering\small",
        r"  \caption{Feedback-driven joint repair versus the independence assumption (AVI) and",
        r"  a 2D STHoles baseline on real correlated column pairs. AVI uses the true marginals",
        r"  (best case for independence); STHoles starts from a uniform root bucket and consumes",
        r"  the same $K{=}16$ observed rectangle masses as Feedback/OASIS, which is the 2D",
        r"  max-entropy (IPF) projection seeded from AVI. Geometric-mean conjunctive-predicate",
        r"  Q-error is measured on the same held-out predicates.}",
        r"  \label{tab:crosscol_feedback}", r"  \setlength{\tabcolsep}{6pt}",
        r"  \begin{tabular}{l r r r r r}", r"    \toprule",
        r"    Real column pair & Pearson $r$ & AVI & STHoles & Feedback & Improv. \\", r"    \midrule",
    ]
    for nm, n, rr, av, sth, fb, im in sorted(rows, key=lambda z: abs(z[2])):
        disp = nm.replace("_", r"\_").replace("~", "/").replace("&", r"\&")
        lines.append(f"    \\texttt{{{disp}}} & ${rr:+.2f}$ & {av:.3f} & {sth:.3f} & {fb:.3f} & ${im:+.1f}\\%$ \\\\")
    lines += [
        r"    \midrule",
        f"    \\textbf{{Aggregate}} & & {GA:.3f} & {GS:.3f} & {GF:.3f} & ${(GA-GF)/GA*100:+.1f}\\%$ \\\\",
        r"    \bottomrule", r"  \end{tabular}", r"\end{table}",
    ]
    with open(os.path.join(a.out, "table_crosscol_feedback.tex"), "w") as h:
        h.write("\n".join(lines) + "\n")
    print(f"wrote {a.out}/summary.csv and table_crosscol_feedback.tex")


if __name__ == "__main__":
    main()
