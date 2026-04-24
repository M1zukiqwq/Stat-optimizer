#!/bin/bash
# 执行 15 轮漂移注入

export PATH="/home/tianqc/.jenv/versions/17/bin:$PATH"
PRESTO_CLI="/home/tianqc/presto-server-0.296/presto-cli"
SERVER="localhost:8080"
SQL_FILE="/home/tianqc/presto-optimizer/presto-cdf-simulation/job_experiment/drift/drift_proportional.sql"

echo "======================================================================"
echo "JOB Benchmark Drift Injection - 15 Rounds"
echo "======================================================================"
echo "Started at: $(date)"
echo ""

# 记录初始行数
echo "初始表大小:"
for table in title cast_info movie_info movie_companies name movie_keyword; do
    count=$($PRESTO_CLI --server $SERVER --catalog iceberg --schema imdb --execute "SELECT COUNT(*) FROM $table" 2>/dev/null | tr -d '"' | grep -v SET | head -1)
    echo "  $table: $count rows"
done
echo ""

# 执行 15 轮漂移
for round in $(seq 1 15); do
    echo "=== Drift Round $round/15 ==="
    $PRESTO_CLI --server $SERVER --catalog iceberg --schema imdb --file $SQL_FILE 2>&1 | grep -E "(INSERT|DELETE|rows)" | head -20
    echo "  ✓ Round $round complete"
    echo ""
done

echo "======================================================================"
echo "漂移注入完成"
echo "Finished at: $(date)"
echo ""
echo "最终表大小:"
for table in title cast_info movie_info movie_companies name movie_keyword; do
    count=$($PRESTO_CLI --server $SERVER --catalog iceberg --schema imdb --execute "SELECT COUNT(*) FROM $table" 2>/dev/null | tr -d '"' | grep -v SET | head -1)
    echo "  $table: $count rows"
done
echo "======================================================================"
