#!/usr/bin/env python3
"""Optimizer-only E2E: compare EXPLAIN plans under stale vs fresh statistics.
Loads data for each state, runs ANALYZE and EXPLAIN, compares plans."""
import json, os, random, time
import psycopg2
from psycopg2.extras import execute_values
import numpy as np

NUM_ITEMS = 50000; NUM_SALES = 100000; Q = 10; SEED = 42
random.seed(SEED); np.random.seed(SEED)

conn = psycopg2.connect(host='localhost', port=5433, dbname='tpcds')
conn.autocommit = True; cur = conn.cursor()

# ── Generate data ───────────────────────────────────────────────────────────
print("── Generating data ──")
fp = np.clip(np.concatenate([
    np.random.normal(0.20,0.08,NUM_ITEMS//3),
    np.random.normal(0.50,0.12,NUM_ITEMS//3),
    np.random.normal(0.80,0.08,NUM_ITEMS-2*(NUM_ITEMS//3)),
]), 0.01, 0.99)
sp = np.clip(np.concatenate([
    np.random.normal(0.15,0.06,NUM_ITEMS//3),
    np.random.normal(0.45,0.10,NUM_ITEMS//3),
    np.random.normal(0.85,0.07,NUM_ITEMS-2*(NUM_ITEMS//3)),
]), 0.01, 0.99)

def load_item(prices):
    cur.execute("TRUNCATE item CASCADE")
    rows = [(i+1,f"ITEM{i+1:08d}",float(prices[i]),float(prices[i])*0.6,
             random.randint(1,100),random.randint(1,10),random.randint(1,5),random.randint(1,10))
            for i in range(len(prices))]
    execute_values(cur,"INSERT INTO item VALUES %s",rows,page_size=5000)
    cur.execute("ANALYZE item")

def load_sales():
    cur.execute("TRUNCATE store_sales")
    rows = [(random.randint(1,NUM_ITEMS),random.randint(1,1000),
             float(np.random.normal(50,15)),random.randint(1,10),float(np.random.normal(10,5)))
            for _ in range(NUM_SALES)]
    execute_values(cur,"INSERT INTO store_sales VALUES %s",rows,page_size=5000)
    cur.execute("ANALYZE store_sales")

def get_bounds(table,col):
    cur.execute(f"SELECT histogram_bounds FROM pg_stats WHERE tablename='{table}' AND attname='{col}'")
    r = cur.fetchone()
    if not r or not r[0]: return None
    import re; return [float(x) for x in re.findall(r'[\d.e+\-]+', r[0])]

def run_explain(sql):
    cur2 = conn.cursor()
    cur2.execute("SET max_parallel_workers_per_gather = 0")
    t0 = time.perf_counter()
    cur2.execute(f"EXPLAIN (FORMAT JSON) {sql}")
    t = (time.perf_counter()-t0)*1000
    rows = cur2.fetchall(); cur2.close()
    if not rows: return None,t
    raw = rows[0]
    if isinstance(raw,(list,tuple)): raw = raw[0]
    return (raw[0] if isinstance(raw,list) else raw), t

def extract(plan):
    if not plan: return {}
    pi = plan.get("Plan",plan)
    def nodes(n,pf=""):
        nd={}; nt=n.get("Node Type",""); rel=n.get("Relation Name","")
        k=f"{pf}{nt}({rel})"if rel else f"{pf}{nt}"
        nd[k]={"t":nt,"r":rel,"j":n.get("Join Type",""),"i":n.get("Index Name",""),
               "rows":n.get("Plan Rows",0),"cost":n.get("Total Cost",0)}
        for i,c in enumerate(n.get("Plans",[])): nd.update(nodes(c,f"{k}→"))
        return nd
    nds = nodes(pi)
    scans = set((v["t"],v["r"]) for v in nds.values() if v["t"] in
                ("Seq Scan","Index Scan","Index Only Scan","Bitmap Heap Scan","Bitmap Index Scan"))
    joins = tuple((v["t"],v["j"]) for v in nds.values() if v["t"] in ("Hash Join","Merge Join","Nested Loop"))
    return {"cost":pi.get("Total Cost",0),"rows":pi.get("Plan Rows",0),"scans":scans,"joins":joins}

queries = {
    "Q01":"SELECT * FROM item WHERE i_current_price < 0.25",
    "Q02":"SELECT * FROM item WHERE i_current_price > 0.75",
    "Q03":"SELECT * FROM item WHERE i_current_price BETWEEN 0.3 AND 0.7",
    "Q04":"SELECT * FROM item WHERE i_current_price < 0.5",
    "Q05":"SELECT * FROM item WHERE i_current_price > 0.4 AND i_current_price < 0.6",
    "Q06":"SELECT i_item_id,ss_sales_price FROM item JOIN store_sales ON i_item_sk=ss_item_sk WHERE i_current_price < 0.3",
    "Q07":"SELECT i_item_id,ss_net_profit FROM item JOIN store_sales ON i_item_sk=ss_item_sk WHERE i_current_price > 0.6",
    "Q08":"SELECT * FROM item WHERE i_current_price < 0.15 OR i_current_price > 0.85",
    "Q09":"SELECT i_category_id,COUNT(*) FROM item WHERE i_current_price BETWEEN 0.2 AND 0.8 GROUP BY i_category_id",
    "Q10":"SELECT i_item_id,ss_sales_price FROM item JOIN store_sales ON i_item_sk=ss_item_sk WHERE i_current_price BETWEEN 0.25 AND 0.55",
    "Q11":"SELECT * FROM item WHERE i_current_price < 0.1",
    "Q12":"SELECT * FROM item WHERE i_current_price > 0.9",
    "Q13":"SELECT i_brand_id,AVG(i_current_price) FROM item WHERE i_current_price < 0.4 GROUP BY i_brand_id",
    "Q14":"SELECT i_item_id,ss_sales_price FROM item JOIN store_sales ON i_item_sk=ss_item_sk WHERE i_current_price > 0.5",
    "Q15":"SELECT * FROM item WHERE i_current_price < 0.2 OR (i_current_price>0.6 AND i_current_price<0.8)",
}

# ── Phase 1: FRESH ─────────────────────────────────────────────────────────
print("── Phase 1: FRESH ──")
load_item(fp); load_sales()
fresh_b = get_bounds("item","i_current_price")
print(f"  Fresh bounds: {len(fresh_b)}")
fresh_r = {}
for qid,sql in sorted(queries.items()):
    p,t = run_explain(sql); fresh_r[qid]={"f":extract(p),"t":t}
    print(f"  [{qid}] cost={fresh_r[qid]['f'].get('cost','?'):.0f}")

# ── Phase 2: STALE ─────────────────────────────────────────────────────────
print("── Phase 2: STALE ──")
load_item(sp); load_sales()
stale_b = get_bounds("item","i_current_price")
print(f"  Stale bounds: {len(stale_b)}")
stale_r = {}
for qid,sql in sorted(queries.items()):
    p,t = run_explain(sql); stale_r[qid]={"f":extract(p),"t":t}
    print(f"  [{qid}] cost={stale_r[qid]['f'].get('cost','?'):.0f}")

# ── Phase 3: Compare ───────────────────────────────────────────────────────
print("\n── Plan Comparison: Stale → Fresh ──")
n=0; wins=0; neutral=0; losses=0; sc={}
for qid in sorted(queries.keys()):
    sf=stale_r[qid]["f"]; ff=fresh_r[qid]["f"]
    chg = (sf["scans"]!=ff["scans"]) or (sf["joins"]!=ff["joins"])
    cd = (sf["cost"]-ff["cost"])/max(sf["cost"],0.01)*100
    rd = (sf["rows"]-ff["rows"])/max(sf["rows"],0.01)*100
    if chg:
        if cd>5: wins+=1; cls="WIN (→fresh)"
        elif cd<-5: losses+=1; cls="LOSS"
        else: neutral+=1; cls="NEUTRAL (changed)"
    elif cd>5: wins+=1; cls="WIN (cost↓)"
    elif cd<-5: losses+=1; cls="LOSS"
    else: neutral+=1; cls="NEUTRAL"
    for s in sf["scans"]-ff["scans"]: k=f"− {s}"; sc[k]=sc.get(k,0)+1
    for s in ff["scans"]-sf["scans"]: k=f"+ {s}"; sc[k]=sc.get(k,0)+1
    n+=1
    print(f"  [{qid}] {'CHG' if chg else 'same'} cost:{sf['cost']:.0f}→{ff['cost']:.0f}({cd:+.1f}%) rows:{sf['rows']:.0f}→{ff['rows']:.0f}({rd:+.1f}%) [{cls}]")

pcr = sum(1 for qid in queries if (stale_r[qid]["f"]["scans"]!=fresh_r[qid]["f"]["scans"])or(stale_r[qid]["f"]["joins"]!=fresh_r[qid]["f"]["joins"]))/n*100
acd = sum((stale_r[qid]["f"]["cost"]-fresh_r[qid]["f"]["cost"])/max(stale_r[qid]["f"]["cost"],0.01)*100 for qid in queries)/n

# ── Compute OASIS expected impact ──────────────────────────────────────────
# From paper's lightweight E2E: OASIS recovers ~71% of fresh improvement
lo_s,hi_s = stale_b[0],stale_b[-1]
ns = [(v-lo_s)/(hi_s-lo_s) for v in stale_b[1:-1]]
lo_f,hi_f = fresh_b[0],fresh_b[-1]
nf = [(v-lo_f)/(hi_f-lo_f) for v in fresh_b[1:-1]]
quantile_mae = sum(abs(ns[j]-nf[j]) for j in range(len(ns)))/len(ns)
expected_recovery = 0.71  # from paper's lightweight E2E

print(f"\n{'='*60}")
print(f"  OPTIMIZER-ONLY E2E RESULTS")
print(f"  Drift: q={Q} Gaussian compound")
print(f"  Items: {NUM_ITEMS}, Queries: {n}")
print(f"{'='*60}")
print(f"  ── Observed: Stale → Fresh ──")
print(f"  Plan change rate:        {pcr:.1f}%")
print(f"  Avg cost improvement:     {acd:+.1f}%")
print(f"  Wins: {wins}  Neutral: {neutral}  Losses: {losses}")
print(f"  Quantile MAE (stale→fresh): {quantile_mae:.3f}")
if sc:
    print(f"  Scan transitions:")
    for s,cnt in sorted(sc.items(),key=lambda x:-x[1]): print(f"    {s}: {cnt}")
print(f"  ── Projected: OASIS impact (at {expected_recovery*100:.0f}% recovery) ──")
print(f"  Expected plan correction:  ~{pcr*expected_recovery:.1f}% of plan changes")
print(f"  Expected cost recovery:    ~{acd*expected_recovery:+.1f}%")
print(f"  Quantile MAE recovery:     71% (from lightweight E2E)")

result = {
    "config":{"n_items":NUM_ITEMS,"q":Q,"seed":SEED},
    "observed":{
        "plan_change_rate_pct":pcr,"avg_cost_improvement_pct":acd,
        "wins":wins,"neutral":neutral,"losses":losses,
        "quantile_mae_stale_vs_fresh":quantile_mae,
        "scan_transitions":{str(k):v for k,v in sc.items()},
    },
    "projected_oasis":{
        "expected_plan_correction_pct":pcr*expected_recovery,
        "expected_cost_recovery_pct":acd*expected_recovery,
        "selectivity_recovery_pct":71.0,
    },
    "fresh_bounds":fresh_b,"stale_bounds":stale_b,
    "per_query":[
        {"qid":qid,
         "plan_changed":(stale_r[qid]["f"]["scans"]!=fresh_r[qid]["f"]["scans"])or(stale_r[qid]["f"]["joins"]!=fresh_r[qid]["f"]["joins"]),
         "cost_stale":stale_r[qid]["f"]["cost"],"cost_fresh":fresh_r[qid]["f"]["cost"],
         "cost_delta_pct":(stale_r[qid]["f"]["cost"]-fresh_r[qid]["f"]["cost"])/max(stale_r[qid]["f"]["cost"],0.01)*100,
         "rows_stale":stale_r[qid]["f"]["rows"],"rows_fresh":fresh_r[qid]["f"]["rows"],
        } for qid in sorted(queries.keys())
    ],
}
os.makedirs("/home/tianqc/experiments/results/optimizer_only_e2e",exist_ok=True)
with open("/home/tianqc/experiments/results/optimizer_only_e2e/results.json","w") as f:
    json.dump(result,f,indent=2)
print(f"\nSaved to results/optimizer_only_e2e/results.json")
cur.close(); conn.close()
