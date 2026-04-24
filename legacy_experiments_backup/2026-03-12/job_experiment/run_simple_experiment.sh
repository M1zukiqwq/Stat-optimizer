#!/bin/bash
# JOB Benchmark 简化实验脚本：Stale Prior vs Full ANALYZE
#
# 此脚本只执行核心对比实验，不包括 OASIS 模型服务
# 适用于快速验证统计更新的效果

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PRESTO_CLI="${PRESTO_CLI:-presto}"
PRESTO_HOST="${PRESTO_HOST:-localhost:8080}"
CATALOG="${CATALOG:-iceberg}"
SCHEMA="${SCHEMA:-imdb}"
USER="${USER:-tianqc}"

# 实验参数
DRIFT_ROUNDS="${DRIFT_ROUNDS:-15}"
DRIFT_RATIO="${DRIFT_RATIO:-0.02}"
QUERY_DIR="${QUERY_DIR:-$SCRIPT_DIR/queries/job}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"

# 表列表（核心 6 张表）
TABLES="${TABLES:-title cast_info movie_info movie_companies name movie_keyword}"

echo "========================================"
echo "JOB Benchmark 简化实验"
echo "Stale Prior vs Full ANALYZE"
echo "========================================"
echo ""
echo "配置："
echo "  Presto: $PRESTO_HOST"
echo "  Catalog: $CATALOG"
echo "  Schema: $SCHEMA"
echo "  User: $USER"
echo "  漂移轮次: $DRIFT_ROUNDS"
echo "  漂移比例: $DRIFT_RATIO"
echo "  查询目录: $QUERY_DIR"
echo "  结果目录: $RESULTS_DIR"
echo "  表列表: $TABLES"
echo ""
read -p "按 Enter 继续，或 Ctrl+C 取消..."
echo ""

# ============================================================
# Step 4: 初始 ANALYZE
# ============================================================
echo "========================================"
echo "Step 4: 初始 ANALYZE"
echo "========================================"
echo ""

for table in $TABLES; do
    echo "ANALYZE $table..."
    $PRESTO_CLI --server $PRESTO_HOST --catalog $CATALOG --schema $SCHEMA \
        --execute "SET SESSION iceberg.statistics_kll_sketch_k_parameter = 1024; ANALYZE $table;" 2>&1 | tail -1
done

echo ""
echo "✓ 初始 ANALYZE 完成"
echo ""

# ============================================================
# Step 5: 数据漂移
# ============================================================
echo "========================================"
echo "Step 5: 数据漂移"
echo "========================================"
echo ""

cd "$SCRIPT_DIR/drift"

python3 inject_drift.py \
    --host localhost \
    --port 8080 \
    --catalog $CATALOG \
    --schema $SCHEMA \
    --rounds $DRIFT_ROUNDS \
    --drift-ratio $DRIFT_RATIO \
    --no-update

echo ""
echo "✓ 数据漂移完成"
echo ""

# ============================================================
# Step 6: Stale Prior 测试
# ============================================================
echo "========================================"
echo "Step 6: Stale Prior 测试（过期统计）"
echo "========================================"
echo ""

cd "$SCRIPT_DIR/experiment"
mkdir -p "$RESULTS_DIR/stale_prior"

python3 run_experiment.py \
    --presto-host $PRESTO_HOST \
    --catalog $CATALOG \
    --schema $SCHEMA \
    --user $USER \
    --query-dir "$QUERY_DIR" \
    --strategy stale_prior \
    --output-dir "$RESULTS_DIR/stale_prior" \
    --no-correction

echo ""
echo "✓ Stale Prior 测试完成"
echo ""

# ============================================================
# Step 7: 重新 ANALYZE
# ============================================================
echo "========================================"
echo "Step 7: 重新 ANALYZE（生成新鲜统计）"
echo "========================================"
echo ""

for table in $TABLES; do
    echo "ANALYZE $table..."
    $PRESTO_CLI --server $PRESTO_HOST --catalog $CATALOG --schema $SCHEMA \
        --execute "SET SESSION iceberg.statistics_kll_sketch_k_parameter = 1024; ANALYZE $table;" 2>&1 | tail -1
done

echo ""
echo "✓ 重新 ANALYZE 完成"
echo ""

# ============================================================
# Step 7.5: 触发新 Snapshot（关键！）
# ============================================================
echo "========================================"
echo "Step 7.5: 触发新 Snapshot（关键步骤）"
echo "========================================"
echo ""
echo "说明：这一步确保 Full ANALYZE 测试使用新的 snapshot"
echo "      从而读取新鲜的统计文件，而不是与 Stale Prior 相同的统计"
echo ""

cd "$SCRIPT_DIR/drift"

if [ -f trigger_new_snapshot.sh ]; then
    ./trigger_new_snapshot.sh "$TABLES"
else
    echo "⚠️  trigger_new_snapshot.sh 未找到，手动触发 snapshot..."
    for table in $TABLES; do
        echo "  处理 $table..."
        $PRESTO_CLI --server $PRESTO_HOST --catalog $CATALOG --schema $SCHEMA \
            --execute "INSERT INTO $table SELECT * FROM $table LIMIT 1;" 2>&1 | tail -1
        $PRESTO_CLI --server $PRESTO_HOST --catalog $CATALOG --schema $SCHEMA \
            --execute "DELETE FROM $table WHERE id IN (SELECT id FROM $table ORDER BY id DESC LIMIT 1);" 2>&1 | tail -1
    done
fi

echo ""
echo "✓ 新 Snapshot 已创建"
echo ""

# ============================================================
# Step 8: Full ANALYZE 测试
# ============================================================
echo "========================================"
echo "Step 8: Full ANALYZE 测试（新鲜统计）"
echo "========================================"
echo ""

cd "$SCRIPT_DIR/experiment"
mkdir -p "$RESULTS_DIR/full_analyze"

python3 run_experiment.py \
    --presto-host $PRESTO_HOST \
    --catalog $CATALOG \
    --schema $SCHEMA \
    --user $USER \
    --query-dir "$QUERY_DIR" \
    --strategy full_analyze \
    --output-dir "$RESULTS_DIR/full_analyze" \
    --no-correction

echo ""
echo "✓ Full ANALYZE 测试完成"
echo ""

# ============================================================
# Step 9: 结果对比
# ============================================================
echo "========================================"
echo "Step 9: 结果对比"
echo "========================================"
echo ""

python3 << EOF
import json

# 加载结果
with open('$RESULTS_DIR/stale_prior/stale_prior_results.json') as f:
    stale = json.load(f)
with open('$RESULTS_DIR/full_analyze/full_analyze_results.json') as f:
    fresh = json.load(f)

# 计算总时间
stale_time = sum(r['execution_time_ms'] for r in stale['results'] if r['status'] == 'success')
fresh_time = sum(r['execution_time_ms'] for r in fresh['results'] if r['status'] == 'success')

# 计算 Q-error
stale_qerrors = [r['qerror']['mean'] for r in stale['results'] if r.get('qerror')]
fresh_qerrors = [r['qerror']['mean'] for r in fresh['results'] if r.get('qerror')]

print("=" * 70)
print("JOB Benchmark 实验结果对比")
print("=" * 70)
print(f"\nStale Prior (过期统计):")
print(f"  总时间: {stale_time/1000:.1f}s")
print(f"  成功查询: {sum(1 for r in stale['results'] if r['status'] == 'success')}/{len(stale['results'])}")
if stale_qerrors:
    print(f"  平均 Q-error: {sum(stale_qerrors)/len(stale_qerrors):.2f}")
    print(f"  最大 Q-error: {max(stale_qerrors):.2f}")

print(f"\nFull ANALYZE (新鲜统计):")
print(f"  总时间: {fresh_time/1000:.1f}s")
print(f"  成功查询: {sum(1 for r in fresh['results'] if r['status'] == 'success')}/{len(fresh['results'])}")
if fresh_qerrors:
    print(f"  平均 Q-error: {sum(fresh_qerrors)/len(fresh_qerrors):.2f}")
    print(f"  最大 Q-error: {max(fresh_qerrors):.2f}")

time_improvement = (stale_time - fresh_time) / stale_time * 100
print(f"\n性能提升: {time_improvement:+.1f}%")

if stale_qerrors and fresh_qerrors:
    qerror_improvement = (sum(stale_qerrors)/len(stale_qerrors) - sum(fresh_qerrors)/len(fresh_qerrors)) / (sum(stale_qerrors)/len(stale_qerrors)) * 100
    print(f"Q-error 改善: {qerror_improvement:+.1f}%")

print("=" * 70)
EOF

echo ""
echo "========================================"
echo "✓ 实验完成！"
echo "========================================"
echo ""
echo "结果文件："
echo "  - Stale Prior: $RESULTS_DIR/stale_prior/stale_prior_results.json"
echo "  - Full ANALYZE: $RESULTS_DIR/full_analyze/full_analyze_results.json"
echo ""
echo "诊断 snapshot 状态："
echo "  cd $SCRIPT_DIR/experiment"
echo "  python3 diagnose_statistics.py --presto-host $PRESTO_HOST --tables $TABLES"
echo ""
