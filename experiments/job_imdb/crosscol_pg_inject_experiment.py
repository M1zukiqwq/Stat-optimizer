#!/usr/bin/env python3
"""A + D: PostgreSQL's REAL conjunctive cardinality error on correlated JOB/IMDB columns,
and how much a feedback-repaired joint recovers -- compared against PostgreSQL's own
scan-built multi-column statistics.

For each correlated pair (T, A, B), on real conjunctive predicates (type-aware: equality on
categorical columns like kind_id, ranges on numeric columns), we score row Q-error of:
  * PG-default : PostgreSQL EXPLAIN with only per-column stats (independence / AVI).        [A]
  * PG-extstat : PostgreSQL EXPLAIN after CREATE STATISTICS (mcv,dependencies) -- PG's own
                 multi-column estimate, built from a full scan.
  * OASIS      : the feedback-repaired joint (2D IPF from K observed predicate masses, no
                 scan) used as the conjunctive estimate.                                      [D]
all against true COUNT(*). Shows the real optimizer's AVI error, and that feedback repair
removes most of it -- matching PG's scan-built extended statistics with no scan.
"""
from __future__ import annotations
import argparse, json, os, random, subprocess
import numpy as np

PAIRS = [
    ("title",           "production_year", "kind_id"),
    ("aka_title",       "production_year", "kind_id"),
    ("cast_info",       "nr_order",        "role_id"),
    ("title",           "season_nr",       "episode_nr"),
    ("movie_companies", "company_type_id", "company_id"),
    ("person_info",     "info_type_id",    "person_id"),   # near-independent control (person_id is an ID)
]
PSQL = None
CAT_MAX = 256        # <= this many distinct -> treat as categorical (per-value bins)
GNUM = 16            # equi-depth bins for numeric columns


def q(sql):
    o = subprocess.run(PSQL + ["-tAF", "\t", "-c", sql], capture_output=True, text=True)
    if o.returncode != 0:
        raise RuntimeError(o.stderr.strip()[:300] + " | SQL: " + sql[:160])
    return [ln.split("\t") for ln in o.stdout.splitlines() if ln != ""]


def scalar(sql):
    r = q(sql)
    return r[0][0] if r and r[0] else None


def col_spec(t, c):
    """Return ('cat', sorted_values) or ('num', boundary_edges[g+1])."""
    nd = int(scalar(f"SELECT count(distinct {c}) FROM {t} WHERE {c} IS NOT NULL"))
    if nd <= CAT_MAX:
        vals = [float(r[0]) for r in q(f"SELECT DISTINCT {c}::double precision FROM {t} "
                                       f"WHERE {c} IS NOT NULL ORDER BY 1")]
        return ("cat", vals)
    lv = [i / GNUM for i in range(1, GNUM)]
    arr = "ARRAY[" + ",".join(f"{x:.6f}" for x in lv) + "]"
    inner = [float(x) for x in scalar(
        f"SELECT percentile_cont({arr}) WITHIN GROUP (ORDER BY {c}::double precision) "
        f"FROM {t} WHERE {c} IS NOT NULL").strip("{}").split(",")]
    lo = float(scalar(f"SELECT min({c}::double precision) FROM {t}"))
    hi = float(scalar(f"SELECT max({c}::double precision) FROM {t}"))
    edges = sorted(set([lo] + inner + [hi]))
    return ("num", edges)


def nbins(spec):
    return len(spec[1]) if spec[0] == "cat" else len(spec[1]) - 1


def build_joint(t, a, b, sa, sb):
    """True joint counts over (binA, binB)."""
    ea = (f"width_bucket({a}::double precision, ARRAY[" +
          ",".join(f"{x:.6f}" for x in sa[1][1:-1]) + "])") if sa[0] == "num" else f"{a}::double precision"
    eb = (f"width_bucket({b}::double precision, ARRAY[" +
          ",".join(f"{x:.6f}" for x in sb[1][1:-1]) + "])") if sb[0] == "num" else f"{b}::double precision"
    rows = q(f"SELECT {ea} ka, {eb} kb, count(*) FROM {t} "
             f"WHERE {a} IS NOT NULL AND {b} IS NOT NULL GROUP BY 1,2")
    na, nb = nbins(sa), nbins(sb)
    catidx_a = {v: i for i, v in enumerate(sa[1])} if sa[0] == "cat" else None
    catidx_b = {v: i for i, v in enumerate(sb[1])} if sb[0] == "cat" else None
    P = np.zeros((na, nb))
    for ka, kb, c in rows:
        ia = catidx_a[float(ka)] if sa[0] == "cat" else min(max(int(float(ka)), 1), na) - 1
        ib = catidx_b[float(kb)] if sb[0] == "cat" else min(max(int(float(kb)), 1), nb) - 1
        P[ia, ib] += float(c)
    return P


def render(col, spec, i0, i1):
    """SQL condition for bin range [i0,i1) of a column, and whether non-empty."""
    if spec[0] == "cat":
        vals = spec[1][i0:i1]
        if not vals:
            return None
        ints = all(v == int(v) for v in vals)
        vs = ",".join(str(int(v)) if ints else repr(v) for v in vals)
        return f"{col} IN ({vs})" if len(vals) > 1 else f"{col} = {vs}"
    e = spec[1]
    lo, hi = e[i0], e[min(i1, len(e) - 1)]
    if hi <= lo:
        return None
    return f"{col} BETWEEN {lo} AND {hi}"


def rects(na, nb, n, rng):
    out = []
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        i0, i1 = sorted(rng.sample(range(na + 1), 2))
        j0, j1 = sorted(rng.sample(range(nb + 1), 2))
        out.append((i0, i1, j0, j1))
    return out


def ipf2d(P0, fb, ft, it=40):
    P = P0.copy()
    for _ in range(it):
        for (i0, i1, j0, j1), t in zip(fb, ft):
            cur = P[i0:i1, j0:j1].sum()
            if cur > 1e-12:
                P[i0:i1, j0:j1] *= t / cur
        s = P.sum()
        if s > 1e-12:
            P /= s
    return P


def qerr(e, t):
    e, t = max(e, 1e-9), max(t, 1e-9)
    return max(e / t, t / e)


def explain_rows(t, where):
    o = subprocess.run(PSQL + ["-tAc", f"EXPLAIN (FORMAT JSON) SELECT * FROM {t} WHERE {where}"],
                       capture_output=True, text=True)
    if o.returncode != 0:
        raise RuntimeError(o.stderr.strip()[:200])
    return float(json.loads(o.stdout.strip())[0]["Plan"]["Plan Rows"])


def run_pair(t, a, b, args, rng):
    sa, sb = col_spec(t, a), col_spec(t, b)
    na, nb = nbins(sa), nbins(sb)
    N = int(scalar(f"SELECT count(*) FROM {t} WHERE {a} IS NOT NULL AND {b} IS NOT NULL"))
    rr = scalar(f"SELECT corr({a}::double precision,{b}::double precision) FROM {t} "
                f"WHERE {a} IS NOT NULL AND {b} IS NOT NULL")
    P = build_joint(t, a, b, sa, sb)
    Pn = P / P.sum()
    avi = (P.sum(1, keepdims=True) @ P.sum(0, keepdims=True)) / (P.sum() ** 2)

    # feedback (for IPF) and eval rectangles (disjoint draws)
    fb = rects(na, nb, args.k, rng)
    ft = [Pn[i0:i1, j0:j1].sum() for (i0, i1, j0, j1) in fb]
    rep = ipf2d(avi, fb, ft)

    # PG-extended stats (built from a scan): create, ANALYZE
    q(f"DROP STATISTICS IF EXISTS oas_ext_{t}")
    q(f"CREATE STATISTICS oas_ext_{t} (mcv, dependencies, ndistinct) ON {a}, {b} FROM {t}")
    q(f"ANALYZE {t}")  # whole-table ANALYZE: required to (re)build extended statistics

    qdef, qext, qoas, qavi = [], [], [], []
    used = 0
    for (i0, i1, j0, j1) in rects(na, nb, args.eval * 4, rng):
        if used >= args.eval:
            break
        ca = render(a, sa, i0, i1); cb = render(b, sb, j0, j1)
        if ca is None or cb is None:
            continue
        where = f"{ca} AND {cb}"
        true = float(scalar(f"SELECT count(*) FROM {t} WHERE {where}"))
        if true < 1:
            continue
        # PG-extended (stats currently present)
        ext = explain_rows(t, where)
        oas = float(rep[i0:i1, j0:j1].sum()) * N
        av = float(avi[i0:i1, j0:j1].sum()) * N    # independence with TRUE marginals
        qext.append(qerr(ext, true)); qoas.append(qerr(oas, true)); qavi.append(qerr(av, true))
        used += 1
        # stash predicate for the default pass
        qdef.append((where, true))

    # PG-default (drop stats, re-EXPLAIN same predicates)
    q(f"DROP STATISTICS IF EXISTS oas_ext_{t}")
    q(f"ANALYZE {t}")  # whole-table ANALYZE: required to (re)build extended statistics
    qd = []
    for where, true in qdef:
        qd.append(qerr(explain_rows(t, where), true))

    gm = lambda xs: float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")
    return {"pair": f"{t}.{a}/{b}", "N": N, "r": float(rr), "n_pred": used,
            "pg_default": gm(qd), "pg_extstat": gm(qext), "avi_true": gm(qavi),
            "oasis": gm(qoas), "kindA": sa[0], "kindB": sb[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--eval", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_pg_inject/summary.json")
    ap.add_argument("--psql", default=os.environ.get("PSQL_BIN", "psql"), help="psql binary")
    ap.add_argument("--host", default=os.environ.get("PGHOST", ""))
    ap.add_argument("--port", default=os.environ.get("PGPORT", ""))
    ap.add_argument("--db", default=os.environ.get("PGDATABASE", "imdb"))
    ap.add_argument("--user", default=os.environ.get("PGUSER", ""))
    a = ap.parse_args()
    global PSQL
    PSQL = [a.psql, "-d", a.db]
    if a.host: PSQL += ["-h", a.host]
    if a.port: PSQL += ["-p", str(a.port)]
    if a.user: PSQL += ["-U", a.user]
    print(f"{'pair':34} {'N':>10} {'r':>6} {'np':>3} | {'PG-def':>7} {'PG-ext':>7} {'AVI*':>7} {'OASIS':>7} | def->oasis  AVI*->oasis")
    print("-" * 110)
    res = []
    for t, ca, cb in PAIRS:
        try:
            r = run_pair(t, ca, cb, a, random.Random(a.seed))
            res.append(r)
            impr = (r["pg_default"] - r["oasis"]) / r["pg_default"] * 100
            impr2 = (r["avi_true"] - r["oasis"]) / r["avi_true"] * 100
            print(f"{r['pair']:34} {r['N']:>10} {r['r']:>6.2f} {r['n_pred']:>3} | "
                  f"{r['pg_default']:>7.2f} {r['pg_extstat']:>7.2f} {r['avi_true']:>7.2f} {r['oasis']:>7.2f} | "
                  f"{impr:>6.1f}%  {impr2:>6.1f}%")
        except Exception as e:
            print(f"{t}.{ca}/{cb}  ERROR: {str(e)[:160]}")
    if res:
        import os
        os.makedirs(a.out.rsplit("/", 1)[0], exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
