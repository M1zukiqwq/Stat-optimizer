#!/bin/bash
#
# PostgreSQL JOB Benchmark: Stale Prior vs Full ANALYZE 对比实验
#
# 实验流程:
# 1. Stale Prior: 使用已注入漂移的表（统计过期）运行 JOB 查询
# 2. Full ANALYZE: 重新 ANALYZE 所有表，然后运行 JOB 查询
#

set -e

# 配置
PG_HOST="${PG_HOST:-localhost}"
PG_PORT="${PG_PORT:-5432}"
PG_USER="${PG_USER:-postgres}"
PG_DB="${PG_DB:-imdb}"
PG_PASSWORD="${PG_PASSWORD:-}"

QUERY_DIR="${QUERY_DIR:-./queries/job}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/simple_comparison}"
DRIFT_ROUNDS="${DRIFT_ROUNDS:-15}"
DRIFT_RATIO="${DRIFT_RATIO:-0.02}"
TIMEOUT="${TIMEOUT:-300}"

# JOB 核心表（用于 ANALYZE）
JOB_TABLES=(
    title cast_info movie_info movie_keyword name
    char_name person_info movie_companies movie_info_idx
    aka_name aka_title complete_cast movie_link
)

echo "=================================="
echo "PostgreSQL JOB Benchmark 对比实验"
echo "=================================="
echo "Database: $PG_USER@$PG_HOST:$PG_PORT/$PG_DB"
echo "Output: $OUTPUT_DIR"
echo ""

# 检查 psql 和 Python
if ! command -v psql &> /dev/null; then
    echo "Error: psql not found"
    exit 1
fi

if ! python3 -c "import psycopg2" 2>/dev/null; then
    echo "Installing psycopg2..."
    pip3 install psycopg2-binary
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查数据库连接
echo "Checking database connection..."
if ! psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -c "SELECT 1" > /dev/null 2>&1; then
    echo "Error: Cannot connect to PostgreSQL"
    echo "Please check:"
    echo "  1. PostgreSQL is running"
    echo "  2. Database '$PG_DB' exists"
    echo "  3. User '$PG_USER' has access"
    exit 1
fi

# 检查表是否存在
echo "Checking tables..."
for table in title cast_info movie_info; do
    if ! psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -c "SELECT COUNT(*) FROM $table" > /dev/null 2>&1; then
        echo "Error: Table '$table' not found"
        echo "Please run setup first:"
        echo "  cd setup && ./1_download_imdb.sh && python3 3_load_data_pg.py"
        exit 1
    fi
done
echo "  ✓ Tables exist"

# 检查漂移是否已注入
echo ""
echo "Checking drift status..."
STALE_COUNT=$(psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -t -c "
    SELECT COUNT(*) FROM pg_stat_user_tables 
    WHERE relname = 'title' 
    AND (last_analyze IS NULL OR last_analyze < NOW() - INTERVAL '1 hour')
" | xargs)

if [ "$STALE_COUNT" -eq "0" ]; then
    echo "  ⚠ Statistics appear fresh. Need to inject drift first."
    echo ""
    echo "Do you want to inject drift now? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Injecting drift ($DRIFT_ROUNDS rounds, $DRIFT_RATIO ratio)..."
        
        # 简单的漂移注入：直接对主要表进行 INSERT/DELETE
        for round in $(seq 1 $DRIFT_ROUNDS); do
            echo "  Round $round/$DRIFT_ROUNDS..."
            
            # title 表：插入早期年份的数据
            psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -c "
                INSERT INTO title (id, title, imdb_index, kind_id, production_year, imdb_id, phonetic_code, episode_of_id, season_nr, episode_nr, series_years, md5sum)
                SELECT 
                    (SELECT COALESCE(MAX(id), 0) FROM title) + generate_series(1, 5000),
                    'Drift Movie ' || generate_series(1, 5000),
                    NULL,
                    1,
                    1920 + (random() * 30)::int,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL
                FROM generate_series(1, 1)
            " > /dev/null 2>&1 || true
            
            # cast_info 表：插入数据
            psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -c "
                INSERT INTO cast_info (id, person_id, movie_id, person_role_id, note, nr_order, role_id)
                SELECT 
                    (SELECT COALESCE(MAX(id), 0) FROM cast_info) + generate_series(1, 10000),
                    (SELECT person_id FROM cast_info ORDER BY random() LIMIT 1),
                    (SELECT movie_id FROM cast_info ORDER BY random() LIMIT 1),
                    NULL,
                    NULL,
                    NULL,
                    1
                FROM generate_series(1, 10000)
            " > /dev/null 2>&1 || true
        done
        
        echo "  ✓ Drift injected"
    else
        echo "Skipping drift injection. Using current state."
    fi
else
    echo "  ✓ Statistics are stale (drift detected)"
fi

# 显示当前统计状态
echo ""
echo "Current table statistics:"
psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -c "
    SELECT 
        relname as table_name,
        n_live_tup as row_count,
        COALESCE(last_analyze::text, last_autoanalyze::text, 'Never') as last_analyze
    FROM pg_stat_user_tables
    WHERE relname IN ('title', 'cast_info', 'movie_info', 'movie_keyword', 'name')
    ORDER BY n_live_tup DESC;
"

# ============================================
# 1. Stale Prior 测试
# ============================================
echo ""
echo "=================================="
echo "Step 1: Stale Prior Test"
echo "=================================="
echo "Running JOB queries with stale statistics..."

python3 experiment/run_simple_pg_experiment.py \
    --host "$PG_HOST" \
    --port "$PG_PORT" \
    --dbname "$PG_DB" \
    --user "$PG_USER" \
    --password "$PG_PASSWORD" \
    --query-dir "$QUERY_DIR" \
    --strategy stale_prior \
    --output-dir "$OUTPUT_DIR" \
    --timeout "$TIMEOUT" \
    --skip-queries 33c 33d || true

STALE_TIME=$(python3 -c "
import json
with open('$OUTPUT_DIR/stale_prior_results.json') as f:
    data = json.load(f)
    total = sum(r['execution_time_ms'] for r in data['results'] if r['status'] == 'success')
    print(f'{total/1000:.1f}')
" 2>/dev/null || echo "N/A")

STALE_QERROR=$(python3 -c "
import json, math
with open('$OUTPUT_DIR/stale_prior_results.json') as f:
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
" 2>/dev/null || echo "N/A")

echo ""
echo "Stale Prior Results:"
echo "  Total time: ${STALE_TIME}s"
echo "  Q-Error: ${STALE_QERROR}"

# ============================================
# 2. Full ANALYZE 测试
# ============================================
echo ""
echo "=================================="
echo "Step 2: Full ANALYZE Test"
echo "=================================="
echo "Running ANALYZE on all tables..."

for table in "${JOB_TABLES[@]}"; do
    echo "  ANALYZE $table..."
    psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -c "ANALYZE $table;" > /dev/null 2>&1 || true
done

echo "  ✓ ANALYZE complete"

echo ""
echo "Running JOB queries with fresh statistics..."

python3 experiment/run_simple_pg_experiment.py \
    --host "$PG_HOST" \
    --port "$PG_PORT" \
    --dbname "$PG_DB" \
    --user "$PG_USER" \
    --password "$PG_PASSWORD" \
    --query-dir "$QUERY_DIR" \
    --strategy full_analyze \
    --output-dir "$OUTPUT_DIR" \
    --timeout "$TIMEOUT" \
    --skip-queries 33c 33d || true

FRESH_TIME=$(python3 -c "
import json
with open('$OUTPUT_DIR/full_analyze_results.json') as f:
    data = json.load(f)
    total = sum(r['execution_time_ms'] for r in data['results'] if r['status'] == 'success')
    print(f'{total/1000:.1f}')
" 2>/dev/null || echo "N/A")

FRESH_QERROR=$(python3 -c "
import json, math
with open('$OUTPUT_DIR/full_analyze_results.json') as f:
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
" 2>/dev/null || echo "N/A")

echo ""
echo "Full ANALYZE Results:"
echo "  Total time: ${FRESH_TIME}s"
echo "  Q-Error: ${FRESH_QERROR}"

# ============================================
# 3. 结果汇总
# ============================================
echo ""
echo "=================================="
echo "Results Summary"
echo "=================================="
echo ""
echo "+----------------+---------------+----------+"
echo "| Strategy       | Total Time    | Q-Error  |"
echo "+----------------+---------------+----------+"
printf "| %-14s | %11ss | %8s |\n" "Stale Prior" "$STALE_TIME" "$STALE_QERROR"
printf "| %-14s | %11ss | %8s |\n" "Full ANALYZE" "$FRESH_TIME" "$FRESH_QERROR"
echo "+----------------+---------------+----------+"

# 计算提升
if [ "$STALE_TIME" != "N/A" ] && [ "$FRESH_TIME" != "N/A" ]; then
    TIME_IMPROVEMENT=$(python3 -c "print(f'{((float('$STALE_TIME') - float('$FRESH_TIME')) / float('$STALE_TIME') * 100):.1f}%')")
    echo ""
    echo "Improvement: $TIME_IMPROVEMENT faster with Full ANALYZE"
fi

if [ "$STALE_QERROR" != "N/A" ] && [ "$FRESH_QERROR" != "N/A" ]; then
    QERROR_IMPROVEMENT=$(python3 -c "print(f'{((float('$STALE_QERROR') - float('$FRESH_QERROR')) / float('$STALE_QERROR') * 100):.1f}%')")
    echo "Q-Error reduction: $QERROR_IMPROVEMENT"
fi

echo ""
echo "Detailed results saved to: $OUTPUT_DIR/"
echo "  - stale_prior_results.json"
echo "  - full_analyze_results.json"
echo ""
echo "Done!"
