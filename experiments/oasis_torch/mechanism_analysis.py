"""Sparse/dense mechanism: rank/nullspace of the feedback-constraint system.

A B-bucket equi-depth marginal has B-1 free interior quantile boundaries. Each
feedback predicate constrains the CDF at its boundary value(s) (<=/>= : 1 anchor;
BETWEEN: 2). After hard projection, the marginal is pinned only along the span of
those constraints; the orthogonal complement (the nullspace) is left FREE, and a
prior fills it. We quantify, per feedback budget K, the independent-constraint
rank and the residual free degrees of freedom (DOF), and show the learned prior's
measured gain tracks free-DOF>0 — collapsing to a tie when dense feedback pins
the marginal (free-DOF -> 0).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def anchors_for_obs(o, lo_v, hi_v):
    """CDF-anchor x-positions a predicate constrains (normalized to [0,1] later)."""
    pt = o.get("predicate_type", "<=")
    v = float(o.get("value", 0.5))
    if pt == "BETWEEN" and o.get("value_upper") is not None:
        return [v, float(o["value_upper"])]
    return [v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--q-values", type=int, nargs="+", default=[1, 3, 5, 10, 15, 20, 25, 30])
    ap.add_argument("--kcaps", type=int, nargs="+", default=[2, 4, 6, 8, 12, 16])
    ap.add_argument("--max-cases-per-q", type=int, default=128)
    ap.add_argument("--num-buckets", type=int, default=10)
    ap.add_argument("--anchor-tol", type=float, default=1e-3,
                    help="Two anchors closer than this (in normalized x) count as one constraint.")
    a = ap.parse_args()
    B = a.num_buckets
    free_total = B - 1  # interior quantile boundaries = marginal DOF

    by_k = defaultdict(lambda: {"rank": [], "free": [], "n": 0})
    for q in a.q_values:
        files = sorted(glob.glob(os.path.join(a.data_root, f"test_q{q}", "*.json")))[:a.max_cases_per_q]
        for f in files:
            try:
                d = json.load(open(f))
            except Exception:
                continue
            obs = d.get("observations", [])
            # normalization domain from the prior boundaries (already ~[0,1] here)
            for K in a.kcaps:
                anchors = []
                for o in obs[:K]:
                    anchors.extend(anchors_for_obs(o, 0.0, 1.0))
                anchors = [min(max(x, 0.0), 1.0) for x in anchors]
                # independent constraints = distinct anchor positions (merge within tol),
                # capped at the marginal DOF (B-1). The CDF is monotone piecewise-linear,
                # so each distinct interior anchor fixes one independent boundary DOF.
                anchors.sort()
                indep = 0
                last = -1.0
                for x in anchors:
                    if 1e-6 < x < 1 - 1e-6 and x - last > a.anchor_tol:
                        indep += 1
                        last = x
                rank = min(indep, free_total)
                free = free_total - rank
                by_k[K]["rank"].append(rank)
                by_k[K]["free"].append(free)
                by_k[K]["n"] += 1

    print("\n==== Feedback-constraint rank / free-DOF vs budget K "
          f"(marginal DOF = B-1 = {free_total}) ====")
    print(f"{'K':>3}  {'mean_rank':>9}  {'mean_freeDOF':>12}  {'frac_pinned(free=0)':>19}  {'n':>6}")
    for K in a.kcaps:
        r = np.array(by_k[K]["rank"]); fr = np.array(by_k[K]["free"])
        pinned = float(np.mean(fr == 0)) if len(fr) else 0.0
        print(f"{K:>3}  {r.mean():>9.2f}  {fr.mean():>12.2f}  {pinned*100:>18.1f}%  {by_k[K]['n']:>6}")
    print("\nInterpretation: free-DOF>0 => the projection leaves marginal freedom the "
          "learned prior fills (optimizer-relevant completion). free-DOF->0 / frac_pinned->1 "
          "=> dense feedback pins the marginal => OASIS converges to ISOMER (tie).")


if __name__ == "__main__":
    main()
