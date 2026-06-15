#!/usr/bin/env python3
"""Cross-column feedback repair vs the independence assumption on REAL JOB/IMDB data,
measured against a live PostgreSQL planner.

This extends the paper's cross-column result (Wine/Bike/Forest/Census) to the Join Order
Benchmark's real IMDB tables. For each correlated column pair (A,B) of a table T:

  * true joint  : binned g x g histogram computed in SQL (equi-depth bins via percentile_cont).
  * AVI baseline: outer product of the TRUE marginals -- best case for independence, isolates
                  the correlation error (exactly as in the paper).
  * feedback    : 2D max-entropy (IPF) projection from AVI onto K observed conjunctive
                  rectangle masses (the feedback), the 2D analogue of the ISOMER projection.
  * PG-actual   : PostgreSQL's OWN estimate for the same conjunctive predicates, via
                  EXPLAIN (FORMAT JSON) -> Plan Rows (it uses per-column stats + independence).

Reports, per pair: Pearson r, AVI Q-error, feedback Q-error, %improvement, and PostgreSQL's
own conjunctive Q-error -- showing the real optimizer suffers from independence and feedback
repair removes most of it, with no table rescan.
"""
from __future__ import annotations
import argparse, json, os, random, subprocess
import numpy as np

# (table, colA, colB) -- numeric column pairs of IMDB tables that JOB filters/joins on.
PAIRS = [
    ("title",          "production_year", "kind_id"),
    ("title",          "season_nr",       "episode_nr"),
    ("aka_title",      "production_year", "kind_id"),
    ("aka_title",      "season_nr",       "episode_nr"),
    ("cast_info",      "nr_order",        "role_id"),
    ("movie_info_idx", "info_type_id",    "movie_id"),
    ("movie_companies","company_type_id", "company_id"),
    ("person_info",    "info_type_id",    "person_id"),
]

PSQL = None  # set in main


def q(sql):
    """Run SQL, return list of rows (each a list of string fields)."""
    out = subprocess.run(PSQL + ["-tAF", "\t", "-c", sql], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"psql error:\n{out.stderr.strip()}\nSQL: {sql[:200]}")
    rows = [ln.split("\t") for ln in out.stdout.splitlines() if ln != ""]
    return rows


def scalar(sql):
    r = q(sql)
    return r[0][0] if r and r[0] else None


def edges(table, col, g):
    """g-1 interior equi-depth thresholds for width_bucket -> g buckets."""
    levels = [i / g for i in range(1, g)]
    arr = "ARRAY[" + ",".join(f"{x:.6f}" for x in levels) + "]"
    sql = (f"SELECT percentile_cont({arr}) WITHIN GROUP (ORDER BY {col}::double precision) "
           f"FROM {table} WHERE {col} IS NOT NULL")
    val = scalar(sql)            # postgres array literal like {1980,1995,...}
    nums = [float(x) for x in val.strip("{}").split(",") if x != ""]
    # dedupe (ties collapse buckets) but keep sorted unique
    uniq = sorted(set(nums))
    return uniq


def joint(table, a, b, ea, eb, g):
    aarr = "ARRAY[" + ",".join(f"{x:.6f}" for x in ea) + "]"
    barr = "ARRAY[" + ",".join(f"{x:.6f}" for x in eb) + "]"
    sql = (f"SELECT width_bucket({a}::double precision,{aarr}) ba, "
           f"width_bucket({b}::double precision,{barr}) bb, count(*) "
           f"FROM {table} WHERE {a} IS NOT NULL AND {b} IS NOT NULL GROUP BY 1,2")
    P = np.zeros((g, g), dtype=np.float64)
    for ba, bb, c in q(sql):
        i, j = int(ba), int(bb)            # width_bucket -> 1..g (0 if below first edge handled as 1)
        i = min(max(i, 1), g) - 1
        j = min(max(j, 1), g) - 1
        P[i, j] += float(c)
    s = P.sum()
    return P / s if s > 0 else P


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


def ipf2d(P0, fb, ft, n_iter=40):
    P = P0.copy()
    for _ in range(n_iter):
        for r, t in zip(fb, ft):
            i0, i1, j0, j1 = r
            cur = P[i0:i1, j0:j1].sum()
            if cur > 1e-12:
                P[i0:i1, j0:j1] *= (t / cur)
        s = P.sum()
        if s > 1e-12:
            P /= s
    return P


def qerr(e, t):
    e, t = max(e, 1e-9), max(t, 1e-9)
    return max(e / t, t / e)


def pg_card_qerr(table, a, b, ea, eb, g, rng, n_pred, total_rows):
    """PostgreSQL's own conjunctive-predicate Q-error (EXPLAIN est vs true COUNT)."""
    # full value range edges (with -inf/+inf as table min/max)
    amin = float(scalar(f"SELECT min({a}::double precision) FROM {table}"))
    amax = float(scalar(f"SELECT max({a}::double precision) FROM {table}"))
    bmin = float(scalar(f"SELECT min({b}::double precision) FROM {table}"))
    bmax = float(scalar(f"SELECT max({b}::double precision) FROM {table}"))
    av = [amin] + list(ea) + [amax]
    bv = [bmin] + list(eb) + [bmax]
    qs = []
    tried = 0
    for r in rects(g, n_pred * 3, rng):
        if tried >= n_pred:
            break
        i0, i1, j0, j1 = r
        a0, a1 = av[min(i0, len(av) - 1)], av[min(i1, len(av) - 1)]
        b0, b1 = bv[min(j0, len(bv) - 1)], bv[min(j1, len(bv) - 1)]
        if a1 <= a0 or b1 <= b0:
            continue
        where = (f"{a} BETWEEN {a0} AND {a1} AND {b} BETWEEN {b0} AND {b1}")
        try:
            ej = scalar(f"EXPLAIN (FORMAT JSON) SELECT * FROM {table} WHERE {where}")
            est = float(json.loads(ej)[0]["Plan"]["Plan Rows"])
            true = float(scalar(f"SELECT count(*) FROM {table} WHERE {where}"))
        except Exception:
            continue
        if true < 1:
            continue
        qs.append(qerr(est, true))
        tried += 1
    return float(np.exp(np.mean(np.log(qs)))) if qs else float("nan"), len(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-true", type=float, default=1e-3)
    ap.add_argument("--pg-pred", type=int, default=24, help="EXPLAIN predicates per pair (0=skip)")
    ap.add_argument("--pg-pred-maxrows", type=int, default=5_000_000,
                    help="skip PG count(*) anchor for tables larger than this")
    ap.add_argument("--out", default="results_crosscol_job")
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
    g = a.grid

    print(f"{'table.pair':34} {'N':>10} {'r':>7} | {'AVI':>7} {'FB':>7} {'impr%':>7} | {'PG':>8}({'n':>3})")
    print("-" * 92)
    rows, agg_avi, agg_fb = [], [], []
    for table, ca, cb in PAIRS:
        try:
            n = int(scalar(f"SELECT count(*) FROM {table} WHERE {ca} IS NOT NULL AND {cb} IS NOT NULL"))
            if n < 1000:
                print(f"{table}.{ca}~{cb:>10} too few ({n})"); continue
            r = scalar(f"SELECT corr({ca}::double precision,{cb}::double precision) FROM {table} "
                       f"WHERE {ca} IS NOT NULL AND {cb} IS NOT NULL")
            r = float(r) if r not in (None, "") else float("nan")
            ea, eb = edges(table, ca, g), edges(table, cb, g)
            if len(ea) < 2 or len(eb) < 2:
                print(f"{table}.{ca}~{cb} degenerate bins"); continue
            P = joint(table, ca, cb, ea, eb, g)
            px, py = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
            avi = px @ py
            avi /= max(avi.sum(), 1e-12)
            pa, pf = [], []
            for t in range(a.trials):
                trng = random.Random(a.seed + 1000 * t)
                fb = rects(g, a.k, trng); ft = [mass(P, rr) for rr in fb]
                rep = ipf2d(avi, fb, ft)
                for rr in rects(g, a.n_eval, trng):
                    tv = mass(P, rr)
                    if tv < a.min_true:
                        continue
                    pa.append(qerr(mass(avi, rr), tv)); pf.append(qerr(mass(rep, rr), tv))
            gm_a = float(np.exp(np.mean(np.log(pa)))); gm_f = float(np.exp(np.mean(np.log(pf))))
            impr = (gm_a - gm_f) / gm_a * 100
            pg_q, pg_n = (float("nan"), 0)
            if a.pg_pred and n <= a.pg_pred_maxrows:
                pg_q, pg_n = pg_card_qerr(table, ca, cb, ea, eb, g, random.Random(a.seed + 7), a.pg_pred, n)
            agg_avi += pa; agg_fb += pf
            rows.append((f"{table}.{ca}/{cb}", n, r, gm_a, gm_f, impr, pg_q, pg_n))
            print(f"{table}.{ca}~{cb:<12}"[:34].ljust(34) +
                  f" {n:>10} {r:>7.3f} | {gm_a:>7.3f} {gm_f:>7.3f} {impr:>6.1f}% | {pg_q:>8.2f}({pg_n:>3})")
        except Exception as e:
            print(f"{table}.{ca}~{cb}  ERROR: {str(e)[:120]}")
    if agg_avi:
        GA = float(np.exp(np.mean(np.log(agg_avi)))); GF = float(np.exp(np.mean(np.log(agg_fb))))
        print("-" * 92)
        print(f"{'AGGREGATE':34} {'':>10} {'':>7} | {GA:>7.3f} {GF:>7.3f} {(GA-GF)/GA*100:>6.1f}% |")
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "summary.json"), "w") as h:
            json.dump({"pairs": rows, "agg_avi": GA, "agg_fb": GF,
                       "agg_impr_pct": (GA - GF) / GA * 100}, h, indent=2)
        print(f"wrote {a.out}/summary.json")


if __name__ == "__main__":
    main()
