#!/bin/bash
#
# 加载 IMDB CSV 数据到 PostgreSQL
# 使用 \COPY 命令，比 INSERT 快 10-50 倍
#

set -e

# 配置
PG_HOST="${PG_HOST:-localhost}"
PG_PORT="${PG_PORT:-5432}"
PG_USER="${PG_USER:-postgres}"
PG_DB="${PG_DB:-imdb}"
DATA_DIR="${DATA_DIR:-./imdb_data/imdb}"

echo "Loading IMDB data into PostgreSQL..."
echo "Database: $PG_USER@$PG_HOST:$PG_PORT/$PG_DB"
echo "Data directory: $DATA_DIR"
echo ""

# 检查数据目录
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Please run ./1_download_imdb.sh first"
    exit 1
fi

# 检查 psql
if ! command -v psql &> /dev/null; then
    echo "Error: psql not found"
    exit 1
fi

# 创建数据库（如果不存在）
echo "Creating database if not exists..."
psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -c "CREATE DATABASE $PG_DB;" 2>/dev/null || true

# 创建表
echo "Creating tables..."
psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -f create_tables_pg.sql

# 加载数据的函数
load_table() {
    local table=$1
    local file=$2
    
    if [ ! -f "$DATA_DIR/$file" ]; then
        echo "  Warning: $file not found, skipping $table"
        return
    fi
    
    echo "  Loading $table from $file..."
    
    # 使用 \COPY 加载数据
    psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" <<EOF
\COPY $table FROM '$DATA_DIR/$file' WITH (FORMAT csv, HEADER false, NULL '', ESCAPE '\');
EOF
}

echo "Loading data..."
echo ""

# 维度表（小表，先加载）
load_table "comp_cast_type" "comp_cast_type.csv"
load_table "company_type" "company_type.csv"
load_table "info_type" "info_type.csv"
load_table "kind_type" "kind_type.csv"
load_table "link_type" "link_type.csv"
load_table "role_type" "role_type.csv"

# 中等表
load_table "keyword" "keyword.csv"
load_table "company_name" "company_name.csv"

# 大表（主要事实表）
load_table "name" "name.csv"
load_table "char_name" "char_name.csv"
load_table "title" "title.csv"
load_table "aka_name" "aka_name.csv"
load_table "aka_title" "aka_title.csv"
load_table "complete_cast" "complete_cast.csv"
load_table "movie_link" "movie_link.csv"
load_table "movie_companies" "movie_companies.csv"
load_table "movie_keyword" "movie_keyword.csv"
load_table "movie_info_idx" "movie_info_idx.csv"
load_table "person_info" "person_info.csv"
load_table "movie_info" "movie_info.csv"
load_table "cast_info" "cast_info.csv"

echo ""
echo "Running ANALYZE..."
psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" <<'EOF'
ANALYZE title;
ANALYZE cast_info;
ANALYZE movie_info;
ANALYZE movie_keyword;
ANALYZE name;
ANALYZE char_name;
ANALYZE person_info;
ANALYZE movie_companies;
ANALYZE movie_info_idx;
ANALYZE aka_name;
ANALYZE aka_title;
ANALYZE complete_cast;
ANALYZE movie_link;
ANALYZE company_name;
ANALYZE keyword;
ANALYZE comp_cast_type;
ANALYZE company_type;
ANALYZE info_type;
ANALYZE kind_type;
ANALYZE link_type;
ANALYZE role_type;
EOF

echo ""
echo "Table row counts:"
psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" <<'EOF'
SELECT 
    relname as table_name,
    n_live_tup as row_count
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_live_tup DESC;
EOF

echo ""
echo "✓ Data loading complete!"
