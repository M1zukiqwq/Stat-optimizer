#!/bin/bash
#
# PostgreSQL JOB Benchmark: Stale Prior vs OASIS vs Full ANALYZE 对比实验
#
# 实验流程:
# 1. Stale Prior: 使用已注入漂移的表（统计过期）运行 JOB 查询
# 2. OASIS: 设置 analyze_only_histogram = on，运行 ANALYZE
# 3. Full ANALYZE: 设置 analyze_only_histogram = off，重新运行 ANALYZE
#

set -e

# 配置
PG_HOST="${PG_HOST:-localhost}"
PG_PORT="${PG_PORT:-5433}"
PG_USER="${PG_USER:-postgres}"
PG_DB="${PG_DB:-imdb}"
PG_PASSWORD="${PG_PASSWORD:-}"

QUERY_DIR="${QUERY_DIR:-./queries/job}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/histogram_comparison}"
DRIFT_ROUNDS="${DRIFT_ROUNDS:-15}"
DRIFT_RATIO="${DRIFT_RATIO:-0.02}"
TIMEOUT="${TIMEOUT:-30}"

# JOB 核心表（用于 ANALYZE）
JOB_TABLES=(
    title cast_info movie_info movie_keyword name
    char_name person_info movie_companies movie_info_idx
    aka_name aka_title complete_cast movie_link
)

echo "=================================="
echo "PostgreSQL Histogram Benchmark Experiment"
echo "=================================="
echo "Database: $PG_USER@$PG_HOST:$PG_PORT/$PG_DB"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# 检查漂移是否已注入
echo "Checking drift status..."
STALE_COUNT=$(psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -t -c "
    SELECT COUNT(*) FROM pg_stat_user_tables 
    WHERE relname = 'title' 
    AND (last_analyze IS NULL OR last_analyze < NOW() - INTERVAL '1 minute')
" | xargs)

if [ "$STALE_COUNT" -eq "0" ]; then
    echo "  ⚠ Statistics appear fresh. Injecting drift ($DRIFT_ROUNDS rounds)..."
    python3 drift/inject_drift_pg.py --dbname "$PG_DB" --user "$PG_USER" --host "$PG_HOST" --port "$PG_PORT" --rounds "$DRIFT_ROUNDS" --drift-ratio "$DRIFT_RATIO"
    echo "  ✓ Drift injected"
else
    echo "  ✓ Statistics are stale (drift detected)"
fi

_run_queries() {
    local strat="$1"
    local run_name="$2"
    echo "Running queries for $run_name..."
    python3 experiment/run_simple_pg_experiment.py \
        --host "$PG_HOST" \
        --port "$PG_PORT" \
        --dbname "$PG_DB" \
        --user "$PG_USER" \
        --password "$PG_PASSWORD" \
        --query-dir "$QUERY_DIR" \
        --strategy "$strat" \
        --output-dir "$OUTPUT_DIR" \
        --timeout "$TIMEOUT" \
        --skip-queries 33c 33d || true
        
    local total_time=$(python3 -c "
import json
try:
    with open('$OUTPUT_DIR/${strat}_results.json') as f:
        data = json.load(f)
        total = sum(r['execution_time_ms'] for r in data['results'] if r['status'] == 'success')
        print(f'{total/1000:.1f}')
except:
    print('N/A')
" 2>/dev/null || echo "N/A")

    local q_error=$(python3 -c "
import json, math
try:
    with open('$OUTPUT_DIR/${strat}_results.json') as f:
        data = json.load(f)
        qerrors = []
        for r in data['results']:
            if r['status'] == 'success' and r.get('qerror'):
                qerrors.append(r['qerror']['geometric_mean'])
        if qerrors:
            geom_mean = math.exp(sum(math.log(q) for q in qerrors) / len(qerrors))
            print(f'{geom_mean:.2f}')
        else:
            print('N/A')
except:
    print('N/A')
" 2>/dev/null || echo "N/A")

    echo "  -> Time: ${total_time}s, Q-Error: $q_error"
    eval "${strat}_TIME=\"$total_time\""
    eval "${strat}_QERROR=\"$q_error\""
}

# ============================================
# 1. Stale Prior 测试
# ============================================
echo ""
echo "=================================="
echo "Step 1: Stale Prior Test"
echo "=================================="
_run_queries "stale_prior" "Stale Statistics"

# ============================================
# 2. OASIS 测试
# ============================================
echo ""
echo "=================================="
echo "Step 2: Histogram Update Only"
echo "=================================="
echo "Setting analyze_only_histogram = on and running ANALYZE..."
for table in "${JOB_TABLES[@]}"; do
    psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -c "SET analyze_only_histogram = on; ANALYZE $table;" > /dev/null 2>&1 || true
done
_run_queries "histogram_only" "OASIS Updates"

# ============================================
# 3. Full ANALYZE 测试
# ============================================
echo ""
echo "=================================="
echo "Step 3: Full ANALYZE"
echo "=================================="
echo "Setting analyze_only_histogram = off and running ANALYZE..."
for table in "${JOB_TABLES[@]}"; do
    psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -c "SET analyze_only_histogram = off; ANALYZE $table;" > /dev/null 2>&1 || true
done
_run_queries "full_analyze" "Full ANALYZE Updates"

# ============================================
# 4. 结果汇总
# ============================================
echo ""
echo "=================================="
echo "Results Summary"
echo "=================================="
echo ""
echo "+----------------+---------------+----------+"
echo "| Strategy       | Total Time    | Q-Error  |"
echo "+----------------+---------------+----------+"
printf "| %-14s | %11ss | %8s |\n" "Stale Prior" "$stale_prior_TIME" "$stale_prior_QERROR"
printf "| %-14s | %11ss | %8s |\n" "OASIS" "$histogram_only_TIME" "$histogram_only_QERROR"
printf "| %-14s | %11ss | %8s |\n" "Full ANALYZE" "$full_analyze_TIME" "$full_analyze_QERROR"
echo "+----------------+---------------+----------+"

echo ""
echo "Detailed results saved to: $OUTPUT_DIR/"
echo "Done!"
