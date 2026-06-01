#!/bin/bash
# Experiment B setup: TPC-H SF10 load + RF-style DML drift on the remote PG.
# Run AFTER Experiment A finishes (heavy CPU/IO; would disturb A's timing).
# Persists pre-drift quantiles so the runtime experiment can reconstruct the
# "stale" histogram after the drift has been applied.
set -euo pipefail
B=/home/tianqc/deps/install/bin
export LD_LIBRARY_PATH=/home/tianqc/deps/install/lib
PORT=55432
DB=tpch
SOCK=/home/tianqc/oasis_pg/run
KIT=/home/tianqc/tpch-kit
DATA=/home/tianqc/oasis_runtime/tpch_data
SF=${SF:-10}
PSQL="$B/psql -h $SOCK -p $PORT -U tianqc -d $DB -v ON_ERROR_STOP=1"

echo "=== [1/6] generate TPC-H SF$SF (dbgen) $(date) ==="
mkdir -p "$DATA"
if [ ! -f "$DATA/lineitem.tbl" ]; then
  cd "$KIT"
  ./dbgen -s "$SF" -f
  mv ./*.tbl "$DATA"/
fi
ls -la "$DATA"/*.tbl | awk '{print $5, $9}'

echo "=== [2/6] create schema $(date) ==="
$PSQL <<'SQL'
DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;
CREATE TABLE region   (r_regionkey int, r_name char(25), r_comment varchar(152));
CREATE TABLE nation   (n_nationkey int, n_name char(25), n_regionkey int, n_comment varchar(152));
CREATE TABLE part     (p_partkey int, p_name varchar(55), p_mfgr char(25), p_brand char(10), p_type varchar(25), p_size int, p_container char(10), p_retailprice numeric(15,2), p_comment varchar(23));
CREATE TABLE supplier (s_suppkey int, s_name char(25), s_address varchar(40), s_nationkey int, s_phone char(15), s_acctbal numeric(15,2), s_comment varchar(101));
CREATE TABLE partsupp (ps_partkey int, ps_suppkey int, ps_availqty int, ps_supplycost numeric(15,2), ps_comment varchar(199));
CREATE TABLE customer (c_custkey int, c_name varchar(25), c_address varchar(40), c_nationkey int, c_phone char(15), c_acctbal numeric(15,2), c_mktsegment char(10), c_comment varchar(117));
CREATE TABLE orders   (o_orderkey bigint, o_custkey int, o_orderstatus char(1), o_totalprice numeric(15,2), o_orderdate date, o_orderpriority char(15), o_clerk char(15), o_shippriority int, o_comment varchar(79));
CREATE TABLE lineitem (l_orderkey bigint, l_partkey int, l_suppkey int, l_linenumber int, l_quantity numeric(15,2), l_extendedprice numeric(15,2), l_discount numeric(15,2), l_tax numeric(15,2), l_returnflag char(1), l_linestatus char(1), l_shipdate date, l_commitdate date, l_receiptdate date, l_shipinstruct char(25), l_shipmode char(10), l_comment varchar(44));
SQL

echo "=== [3/6] load (.tbl trailing pipe stripped) $(date) ==="
for t in region nation part supplier partsupp customer orders lineitem; do
  echo "  load $t"
  $PSQL -c "COPY $t FROM PROGRAM 'sed -e ''s/|\$//'' $DATA/$t.tbl' WITH (FORMAT text, DELIMITER '|', NULL '')"
done

echo "=== [4/6] keys + indexes + ANALYZE (pre-drift) $(date) ==="
$PSQL <<'SQL'
ALTER TABLE region   ADD PRIMARY KEY (r_regionkey);
ALTER TABLE nation   ADD PRIMARY KEY (n_nationkey);
ALTER TABLE part     ADD PRIMARY KEY (p_partkey);
ALTER TABLE supplier ADD PRIMARY KEY (s_suppkey);
ALTER TABLE partsupp ADD PRIMARY KEY (ps_partkey, ps_suppkey);
ALTER TABLE customer ADD PRIMARY KEY (c_custkey);
ALTER TABLE orders   ADD PRIMARY KEY (o_orderkey);
ALTER TABLE lineitem ADD PRIMARY KEY (l_orderkey, l_linenumber);
CREATE INDEX l_shipdate_idx   ON lineitem(l_shipdate);
CREATE INDEX l_orderkey_idx   ON lineitem(l_orderkey);
CREATE INDEX l_partkey_idx    ON lineitem(l_partkey);
CREATE INDEX o_orderdate_idx  ON orders(o_orderdate);
CREATE INDEX o_custkey_idx    ON orders(o_custkey);
ALTER TABLE lineitem ALTER COLUMN l_shipdate  SET STATISTICS 100;
ALTER TABLE orders   ALTER COLUMN o_orderdate SET STATISTICS 100;
ANALYZE;
SQL

echo "=== [5/6] persist pre-drift quantiles (stale prior) $(date) ==="
$PSQL <<'SQL'
DROP TABLE IF EXISTS stale_quantiles;
CREATE TABLE stale_quantiles (col text, q double precision, v date);
INSERT INTO stale_quantiles
SELECT 'lineitem.l_shipdate', q,
       percentile_disc(q) WITHIN GROUP (ORDER BY l_shipdate)
FROM lineitem, unnest(ARRAY[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) AS q
GROUP BY q;
INSERT INTO stale_quantiles
SELECT 'orders.o_orderdate', q,
       percentile_disc(q) WITHIN GROUP (ORDER BY o_orderdate)
FROM orders, unnest(ARRAY[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) AS q
GROUP BY q;
SELECT col, min(v), max(v) FROM stale_quantiles GROUP BY col;
SQL

echo "=== [6/6] RF-style DML drift (insert recent + delete old) $(date) ==="
# RF1: insert new orders + their lineitems, shifted +900 days into a NEW date band
#      the stale histogram does not cover; offset keys by 10^10 to avoid collisions.
# RF2: delete the oldest orders (and their lineitems).
$PSQL <<'SQL'
\timing on
-- RF1 inserts (orders first, then matching lineitems -> referential structure kept)
INSERT INTO orders
SELECT o_orderkey + 10000000000, o_custkey, o_orderstatus, o_totalprice,
       (o_orderdate + INTERVAL '900 days')::date, o_orderpriority, o_clerk, o_shippriority, o_comment
FROM orders
WHERE o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1997-12-31';

INSERT INTO lineitem
SELECT l_orderkey + 10000000000, l_partkey, l_suppkey, l_linenumber, l_quantity,
       l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus,
       (l_shipdate + INTERVAL '900 days')::date,
       (l_commitdate + INTERVAL '900 days')::date,
       (l_receiptdate + INTERVAL '900 days')::date,
       l_shipinstruct, l_shipmode, l_comment
FROM lineitem
WHERE l_orderkey IN (SELECT o_orderkey - 10000000000 FROM orders WHERE o_orderkey > 10000000000);

-- RF2 deletes (oldest year)
DELETE FROM lineitem WHERE l_orderkey IN (SELECT o_orderkey FROM orders WHERE o_orderkey < 10000000000 AND o_orderdate < DATE '1992-12-31');
DELETE FROM orders   WHERE o_orderkey < 10000000000 AND o_orderdate < DATE '1992-12-31';
ANALYZE orders; ANALYZE lineitem;
SELECT 'orders' t, count(*), min(o_orderdate), max(o_orderdate) FROM orders
UNION ALL
SELECT 'lineitem', count(*), min(l_shipdate), max(l_shipdate) FROM lineitem;
SQL
echo "TPCH_SETUP_DRIFT_DONE $(date)"
