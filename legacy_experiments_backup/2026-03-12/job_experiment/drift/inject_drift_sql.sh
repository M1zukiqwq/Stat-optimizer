#!/bin/bash
# 使用 SQL 文件注入多轮漂移

export PATH="/home/tianqc/.jenv/versions/17/bin:$PATH"
PRESTO_CLI="/home/tianqc/presto-server-0.296/presto-cli"
SERVER="localhost:8080"
CATALOG="iceberg"
SCHEMA="imdb"

ROUNDS=15
SQL_TEMPLATE="/home/tianqc/presto-optimizer/presto-cdf-simulation/job_experiment/drift/drift_round.sql"

echo "======================================================================"
echo "JOB Benchmark Drift Injection"
echo "======================================================================"
echo "Rounds: $ROUNDS"
echo "Started at: $(date)"
echo ""

# 记录初始行数
echo "初始表大小:"
for table in title cast_info movie_info movie_companies name movie_keyword; do
    count=$($PRESTO_CLI --server $SERVER --catalog $CATALOG --schema $SCHEMA --execute "SELECT COUNT(*) FROM $table" 2>/dev/null | tr -d '"')
    echo "  $table: $count rows"
done
echo ""

# 执行多轮漂移
for round in $(seq 1 $ROUNDS); do
    echo "=== Drift Round $round/$ROUNDS ==="
    
    # 生成本轮 SQL
    round_sql=$(sed "s/{round_num}/$round/g" $SQL_TEMPLATE)
    
    # 执行漂移 SQL
    echo "$round_sql" | $PRESTO_CLI --server $SERVER --catalog $CATALOG --schema $SCHEMA 2>&1 | grep -E "(INSERT|rows)" || true
    
    echo "  ✓ Round $round complete"
    echo ""
done

echo "======================================================================"
echo "漂移注入完成"
echo "Finished at: $(date)"
echo ""
echo "最终表大小:"
for table in title cast_info movie_info movie_companies name movie_keyword; do
    count=$($PRESTO_CLI --server $SERVER --catalog $CATALOG --schema $SCHEMA --execute "SELECT COUNT(*) FROM $table" 2>/dev/null | tr -d '"')
    echo "  $table: $count rows"
done
echo "======================================================================"
