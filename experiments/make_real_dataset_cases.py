#!/usr/bin/env python3
"""Generate OASIS drift cases (synthetic-suite JSON schema) from a REAL dataset column.

Drift model -- "sliding window since the last refresh". A numeric column is read in
file order (chronological for Power, spatially contiguous for Forest/Covtype, so the
marginal genuinely drifts along the stream). For each case the STALE snapshot is a
window of W rows [s, s+W) -- the table at the last ANALYZE -- and the FRESH snapshot
is an equally sized window [s+off, s+off+W) shifted forward by off = round(r*stride)
rows, modelling a retention/sliding-window table whose contents have moved on. Larger
r = larger shift = more drift (the analogue of the synthetic suite's drift intensity).
B=10 equi-depth deciles of each give prior_kll / corrected_kll; feedback predicates
get estimated_sel from the stale decile CDF and actual_sel from the fresh decile CDF
-- the identical protocol the synthetic suite uses, over real-data distributions.

The emitted test_q<tag>/*.json are byte-compatible with json_histogram_parser and
are consumed unchanged by run_v3.py (stage1swap / proj / odp), so the canonical
projection / router / Q-error code -- and the pretrained v3 prior -- are reused
verbatim and the numbers are directly comparable to the synthetic tables.

Example:
  python make_real_dataset_cases.py --csv data/real/household_power_consumption.txt \
      --col 2 --sep ';' --skip-header --name power --window 20000 --stride 400000 \
      --out data/real_cases/power --intensities 0.1 0.3 0.5 1.0 --cases-per-intensity 128
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random

import numpy as np

_TS_BASE = datetime.datetime(2026, 1, 1, 0, 0, 0)

LEVELS9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # B=10 interior deciles


def read_column(path: str, col: int, sep: str, skip_header: bool) -> np.ndarray:
    vals = []
    with open(path) as f:
        if skip_header:
            next(f, None)
        for line in f:
            parts = line.rstrip("\n").split(sep)
            if col >= len(parts):
                continue
            tok = parts[col].strip()
            if tok in ("", "?", "NA", "NaN", "nan"):
                continue
            try:
                vals.append(float(tok))
            except ValueError:
                continue
    return np.asarray(vals, dtype=np.float64)


def deciles(x: np.ndarray) -> list:
    q = np.quantile(x, LEVELS9)
    q = np.maximum.accumulate(q)  # enforce monotone for the piecewise-linear CDF
    return [float(v) for v in q]


def piecewise_cdf(qv: list, x: float) -> float:
    bx = [0.0] + list(qv) + [1.0]
    lv = np.linspace(0.0, 1.0, len(bx))
    return float(np.interp(min(max(x, 0.0), 1.0), bx, lv))


def interval_sel(qv: list, lo: float, hi: float) -> float:
    return max(0.0, piecewise_cdf(qv, hi) - piecewise_cdf(qv, lo))


def make_kll(qv: list, full: bool) -> dict:
    d = {
        "type": "double", "k": 1024,
        "quantile_levels": list(LEVELS9),
        "quantile_values": [float(v) for v in qv],
        "bucket_boundaries": [0.0] + [float(v) for v in qv] + [1.0],
    }
    if full:
        d.update({"min": 0.0, "max": 1.0, "null_fraction": 0.0})
    return d


def gen_observations(stale_qv: list, fresh_qv: list, k: int, rng: random.Random) -> list:
    obs = []
    for i in range(k):
        t = rng.choice(["<=", ">=", "BETWEEN"])
        if t == "BETWEEN":
            a, b = sorted((rng.random(), rng.random()))
            lo, hi, val, vup = a, b, a, b
        elif t == "<=":
            v = rng.random(); lo, hi, val, vup = 0.0, v, v, None
        else:
            v = rng.random(); lo, hi, val, vup = v, 1.0, v, None
        o = {
            "predicate_type": t, "value": float(val),
            "estimated_sel": float(interval_sel(stale_qv, lo, hi)),
            "actual_sel": float(interval_sel(fresh_qv, lo, hi)),
            "timestamp": (_TS_BASE + datetime.timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if vup is not None:
            o["value_upper"] = float(vup)
        obs.append(o)
    return obs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--col", type=int, required=True)
    ap.add_argument("--sep", default=",")
    ap.add_argument("--skip-header", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=20000)
    ap.add_argument("--stride", type=int, default=200000,
                    help="fresh window is shifted forward by round(intensity*stride) rows")
    ap.add_argument("--intensities", type=float, nargs="+", default=[0.1, 0.3, 0.5, 1.0])
    ap.add_argument("--cases-per-intensity", type=int, default=128)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test", choices=["train", "test"],
                    help="emit <split>_q<tag>/ dirs (train pool vs test/val)")
    a = ap.parse_args()

    col = read_column(a.csv, a.col, a.sep, a.skip_header)
    lo, hi = np.percentile(col, [0.5, 99.5])  # robust range to tame outliers
    hi = max(hi, lo + 1e-9)
    coln = np.clip((col - lo) / (hi - lo), 0.0, 1.0)
    n = len(coln)
    rng = random.Random(a.seed)

    total = 0
    for r in a.intensities:
        tag = int(round(r * 10))
        d = os.path.join(a.out, f"{a.split}_q{tag}")
        os.makedirs(d, exist_ok=True)
        w = a.window
        off = int(round(r * a.stride))
        hi_start = n - off - w - 1
        if hi_start <= 0:
            raise SystemExit(f"window {w}+offset {off} too large for {a.name} (N={n}, r={r})")
        for ci in range(a.cases_per_intensity):
            s = rng.randint(0, hi_start)
            stale = coln[s:s + w]
            fresh = coln[s + off:s + off + w]
            sq, fq = deciles(stale), deciles(fresh)
            case = {
                "prior_kll": make_kll(sq, full=True),
                "observations": gen_observations(sq, fq, a.k, rng),
                "corrected_kll": make_kll(fq, full=False),
            }
            with open(os.path.join(d, f"real_{a.name}_{ci:04d}.json"), "w") as h:
                json.dump(case, h)
            total += 1
    print(f"[{a.name}] N={n} value_range=[{lo:.4g},{hi:.4g}] window={a.window} stride={a.stride} "
          f"-> {a.out} intensities={a.intensities} cases/int={a.cases_per_intensity} total={total}")


if __name__ == "__main__":
    main()
