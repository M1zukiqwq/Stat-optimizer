#!/bin/bash
# 升级 Iceberg 表到 v2 格式以支持行级别 DELETE/UPDATE

PRESTO_CLI="${PRESTO_CLI:-/home/tianqc/presto-server-0.296/presto-cli}"
HOST="localhost:8080"
CATALOG="iceberg"
SCHEMA="imdb"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 所有事实表
TABLES=(
    "cast_info"
    "movie_info"
    "movie_keyword"
    "name"
    "char_name"
    "person_info"
    "movie_companies"
    "title"
    "movie_info_idx"
    "aka_name"
    "aka_title"
    "complete_cast"
    "movie_link"
)

echo "=========================================="
echo "升级 Iceberg 表到 v2 格式"
echo "=========================================="
echo "Catalog: $CATALOG"
echo "Schema: $SCHEMA"
echo "Tables: ${#TABLES[@]}"
echo "=========================================="
echo ""

success_count=0
fail_count=0

for table in "${TABLES[@]}"; do
    echo -e "${GREEN}[INFO]${NC} 升级 $table 到 format-version=2..."

    result=$($PRESTO_CLI --server $HOST --catalog $CATALOG --schema $SCHEMA \
        --execute "ALTER TABLE $table SET PROPERTIES format_version = 2" 2>&1)

    exit_code=$?

    if [ $exit_code -eq 0 ] && ! echo "$result" | grep -qi "error\|failed\|exception"; then
        echo -e "${GREEN}  ✓ $table 升级成功${NC}"
        success_count=$((success_count + 1))
    else
        echo -e "${RED}  ✗ $table 升级失败${NC}"
        if [ -n "$result" ]; then
            echo "     错误: $result" | head -3
        fi
        fail_count=$((fail_count + 1))
    fi
    echo ""
done

echo "=========================================="
echo "升级完成"
echo "=========================================="
echo "  成功: $success_count/${#TABLES[@]}"
echo "  失败: $fail_count"
echo "=========================================="

if [ $fail_count -gt 0 ]; then
    exit 1
fi
