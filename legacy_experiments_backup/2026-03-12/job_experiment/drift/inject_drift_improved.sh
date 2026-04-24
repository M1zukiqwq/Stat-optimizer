#!/bin/bash
# 改进的数据漂移注入脚本（Shell 版本）
# 修复了原版的 ID 冲突和外键一致性问题

set -e  # 遇到错误立即退出

# 配置
PRESTO_CLI="${PRESTO_CLI:-/home/tianqc/presto-server-0.296/presto-cli}"
HOST="localhost:8080"
CATALOG="iceberg"
SCHEMA="imdb"
ROUNDS=15
DRIFT_RATIO=0.02

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# 执行 SQL 并检查结果
execute_sql() {
    local sql="$1"
    local description="$2"

    log_info "$description"

    result=$($PRESTO_CLI --server $HOST --catalog $CATALOG --schema $SCHEMA \
        --execute "$sql" 2>&1)

    if [ $? -ne 0 ]; then
        log_error "执行失败: $description"
        log_error "错误信息: $result"
        return 1
    fi

    echo "$result"
    return 0
}

# 获取表行数
get_table_count() {
    local table="$1"
    local count=$(execute_sql "SELECT COUNT(*) FROM $table" "查询 $table 行数" | \
        grep -v "SET" | tr -d '"' | grep -E '^[0-9]+$' | head -1)
    echo "$count"
}

# 注入一轮漂移
inject_drift_round() {
    local round=$1

    echo ""
    echo "=========================================="
    echo "开始第 $round 轮数据漂移"
    echo "=========================================="

    # 记录漂移前的数据量
    log_info "漂移前数据量:"
    for table in title cast_info movie_info movie_companies name movie_keyword; do
        count=$(get_table_count "$table")
        echo "  $table: $count 行"
    done

    # INSERT 操作（使用动态 ID）
    echo ""
    log_info "📥 插入新数据（2%）..."

    # title 表
    execute_sql "
    INSERT INTO $CATALOG.$SCHEMA.title
    SELECT
        (SELECT COALESCE(MAX(id), 0) FROM $CATALOG.$SCHEMA.title) + row_number() OVER () as id,
        title || '_drift_r${round}',
        imdb_index,
        kind_id,
        production_year,
        imdb_id,
        phonetic_code,
        episode_of_id,
        season_nr,
        episode_nr,
        series_years,
        md5sum
    FROM $CATALOG.$SCHEMA.title
    WHERE id % 50 = ${round}
    LIMIT (SELECT CAST(COUNT(*) * $DRIFT_RATIO AS BIGINT) FROM $CATALOG.$SCHEMA.title)
    " "插入 title" || log_info "✅ title 插入完成"

    # name 表
    execute_sql "
    INSERT INTO $CATALOG.$SCHEMA.name
    SELECT
        (SELECT COALESCE(MAX(id), 0) FROM $CATALOG.$SCHEMA.name) + row_number() OVER () as id,
        name || '_drift_r${round}',
        imdb_index,
        imdb_id,
        gender,
        name_pcode_cf,
        name_pcode_nf,
        surname_pcode,
        md5sum
    FROM $CATALOG.$SCHEMA.name
    WHERE id % 50 = ${round}
    LIMIT (SELECT CAST(COUNT(*) * $DRIFT_RATIO AS BIGINT) FROM $CATALOG.$SCHEMA.name)
    " "插入 name" || log_info "✅ name 插入完成"

    # cast_info 表（保持外键一致性）
    execute_sql "
    INSERT INTO $CATALOG.$SCHEMA.cast_info
    SELECT
        (SELECT COALESCE(MAX(id), 0) FROM $CATALOG.$SCHEMA.cast_info) + row_number() OVER () as id,
        person_id,
        movie_id,
        person_role_id,
        note,
        nr_order,
        role_id
    FROM $CATALOG.$SCHEMA.cast_info
    WHERE id % 50 = ${round}
    LIMIT (SELECT CAST(COUNT(*) * $DRIFT_RATIO AS BIGINT) FROM $CATALOG.$SCHEMA.cast_info)
    " "插入 cast_info" || log_info "✅ cast_info 插入完成"

    # movie_info 表
    execute_sql "
    INSERT INTO $CATALOG.$SCHEMA.movie_info
    SELECT
        (SELECT COALESCE(MAX(id), 0) FROM $CATALOG.$SCHEMA.movie_info) + row_number() OVER () as id,
        movie_id,
        info_type_id,
        info || '_drift_r${round}',
        note
    FROM $CATALOG.$SCHEMA.movie_info
    WHERE id % 50 = ${round}
    LIMIT (SELECT CAST(COUNT(*) * $DRIFT_RATIO AS BIGINT) FROM $CATALOG.$SCHEMA.movie_info)
    " "插入 movie_info" || log_info "✅ movie_info 插入完成"

    # movie_companies 表
    execute_sql "
    INSERT INTO $CATALOG.$SCHEMA.movie_companies
    SELECT
        (SELECT COALESCE(MAX(id), 0) FROM $CATALOG.$SCHEMA.movie_companies) + row_number() OVER () as id,
        movie_id,
        company_id,
        company_type_id,
        note
    FROM $CATALOG.$SCHEMA.movie_companies
    WHERE id % 50 = ${round}
    LIMIT (SELECT CAST(COUNT(*) * $DRIFT_RATIO AS BIGINT) FROM $CATALOG.$SCHEMA.movie_companies)
    " "插入 movie_companies" || log_info "✅ movie_companies 插入完成"

    # movie_keyword 表
    execute_sql "
    INSERT INTO $CATALOG.$SCHEMA.movie_keyword
    SELECT
        (SELECT COALESCE(MAX(id), 0) FROM $CATALOG.$SCHEMA.movie_keyword) + row_number() OVER () as id,
        movie_id,
        keyword_id
    FROM $CATALOG.$SCHEMA.movie_keyword
    WHERE id % 50 = ${round}
    LIMIT (SELECT CAST(COUNT(*) * $DRIFT_RATIO AS BIGINT) FROM $CATALOG.$SCHEMA.movie_keyword)
    " "插入 movie_keyword" || log_info "✅ movie_keyword 插入完成"

    # DELETE 操作（1%）
    echo ""
    log_info "🗑️  删除旧数据（1%）..."

    for table in title cast_info movie_info movie_companies name movie_keyword; do
        execute_sql "
        DELETE FROM $CATALOG.$SCHEMA.$table
        WHERE id % 100 = ${round}
        AND id < (SELECT MAX(id) * 0.5 FROM $CATALOG.$SCHEMA.$table)
        " "删除 $table" || log_info "✅ $table 删除完成"
    done

    # 验证漂移后的数据量
    echo ""
    log_info "📊 漂移后数据量:"
    for table in title cast_info movie_info movie_companies name movie_keyword; do
        count=$(get_table_count "$table")
        echo "  $table: $count 行"
    done

    echo ""
    log_info "✅ 第 $round 轮漂移完成"
}

# 主函数
main() {
    echo "=========================================="
    echo "JOB Benchmark 数据漂移注入（改进版）"
    echo "=========================================="
    echo "配置:"
    echo "  Presto: $HOST"
    echo "  Catalog: $CATALOG"
    echo "  Schema: $SCHEMA"
    echo "  轮数: $ROUNDS"
    echo "  漂移比例: $(echo "$DRIFT_RATIO * 100" | bc)%"
    echo "=========================================="

    # 检查 Presto CLI
    if [ ! -f "$PRESTO_CLI" ]; then
        log_error "Presto CLI 不存在: $PRESTO_CLI"
        exit 1
    fi

    # 执行漂移
    start_time=$(date +%s)

    for round in $(seq 1 $ROUNDS); do
        inject_drift_round $round

        if [ $? -ne 0 ]; then
            log_error "第 $round 轮漂移失败，停止实验"
            exit 1
        fi
    done

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo ""
    echo "=========================================="
    log_info "实验完成！"
    echo "  总耗时: $((duration / 60)) 分钟 $((duration % 60)) 秒"
    echo "  成功轮数: $ROUNDS/$ROUNDS"
    echo "=========================================="
}

# 运行主函数
main
