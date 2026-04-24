#!/bin/bash
# 触发新 snapshot：对每个表插入并立即删除一条记录
#
# 用途：在重新 ANALYZE 后创建新的 snapshot，确保 Full ANALYZE 测试
#       使用的是新鲜的统计文件，而不是与 Stale Prior 测试相同的统计

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRESTO_CLI="${PRESTO_CLI:-presto}"

# 默认处理核心 6 张表
TABLES="${1:-title cast_info movie_info movie_companies name movie_keyword}"

echo "========================================"
echo "触发新 Snapshot"
echo "========================================"
echo ""
echo "说明：对每个表执行 INSERT + DELETE 操作"
echo "      这会创建新的 snapshot，但不改变数据内容"
echo ""

for table in $TABLES; do
    echo "处理表: $table"

    # 插入一条记录（复制表中的第一条）
    echo "  [1/2] 插入 1 条记录..."
    $PRESTO_CLI --server localhost:8080 --catalog iceberg --schema imdb \
        --execute "INSERT INTO $table SELECT * FROM $table LIMIT 1;" 2>&1 | tail -1

    if [ $? -eq 0 ]; then
        echo "        ✓ 插入成功"
    else
        echo "        ✗ 插入失败"
        exit 1
    fi

    # 立即删除刚插入的记录（保持数据不变）
    echo "  [2/2] 删除刚插入的记录..."
    $PRESTO_CLI --server localhost:8080 --catalog iceberg --schema imdb \
        --execute "DELETE FROM $table WHERE id IN (SELECT id FROM $table ORDER BY id DESC LIMIT 1);" 2>&1 | tail -1

    if [ $? -eq 0 ]; then
        echo "        ✓ 删除成功"
    else
        echo "        ✗ 删除失败"
        exit 1
    fi

    echo ""
done

echo "========================================"
echo "✓ 新 Snapshot 已创建"
echo "========================================"
echo ""
echo "验证 snapshot 历史："
echo "  $PRESTO_CLI --server localhost:8080 --catalog iceberg --schema imdb \\"
echo "    --execute \"SELECT snapshot_id, committed_at, operation FROM \\\"iceberg\\\".\\\"imdb\\\".\\\"title\\\$snapshots\\\" ORDER BY committed_at DESC LIMIT 5;\""
echo ""
