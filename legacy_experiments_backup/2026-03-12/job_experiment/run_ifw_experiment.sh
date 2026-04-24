#!/bin/bash
# IMDB Filter Workload (IFW) 实验脚本
#
# 验证直方图修正效果的端到端实验
# 与 JOB Benchmark 互补：JOB 测试 JOIN，IFW 测试 Filter

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PRESTO_CLI="${PRESTO_CLI:-presto}"
PRESTO_HOST="${PRESTO_HOST:-localhost:8080}"
CATALOG="${CATALOG:-iceberg}"
SCHEMA="${SCHEMA:-imdb}"
USER="${USER:-tianqc}"

DRIFT_ROUNDS="${DRIFT_ROUNDS:-15}"
DRIFT_RATIO="${DRIFT_RATIO:-0.02}"
QUERY_DIR="$SCRIPT_DIR/queries/ifw"
RESULTS_DIR="$SCRIPT_DIR/results/ifw"
TABLES="title cast_info movie_info movie_companies name movie_keyword"

echo "========================================"
echo "IMDB Filter Workload (IFW) 实验"
echo "验证直方图修正效果"
echo "========================================"
echo ""
echo "配置："
echo "  Presto: $PRESTO_HOST"
echo "  查询目录: $QUERY_DIR"
echo "  结果目录: $RESULTS_DIR"
echo "  漂移轮次: $DRIFT_ROUNDS"
echo "  漂移比例: $DRIFT_RATIO"
echo ""

# 检查查询文件
QUERY_COUNT=$(ls "$QUERY_DIR"/*.sql 2>/dev/null | wc -l | tr -d ' ')
if [ "$QUERY_COUNT" -eq 0 ]; then
    echo "✗ 没有找到查询文件，先生成："
    echo "  python3 scripts/generate_ifw_queries.py --output queries/ifw/"
    exit 1
fi
echo "查询文件: $QUERY_COUNT 个"
echo ""
read -p "按 Enter 继续，或 Ctrl+C 取消..."
echo ""

# ============================================================
# Phase 1: Baseline（无漂移，新鲜统计）
# ============================================================
echo "========================================"
echo "Phase 1: Baseline（无漂移，新鲜统计）"
echo "========================================"
echo ""

echo "[1.1] ANALYZE..."
for table in $TABLES; do
    echo "  ANALYZE $table..."
    $PRESTO_CLI --server $PRESTO_HOST --catalog $CATALOG --schema $SCHEMA \
        --execute "SET SESSION iceberg.statistics_kll_sketch_k_parameter = 1024; ANALYZE $table;" 2>&1 | tail -1
done

echo ""
echo "[1.2] 运行 IFW 查询..."
cd "$SCRIPT_DIR/experiment"
mkdir -p "$RESULTS_DIR/baseline"

python3 run_experiment.py \
    --presto-host $PRESTO_HOST \
    --catalog $CATALOG \
    --schema $SCHEMA \
    --user $USER \
    --query-dir "$QUERY_DIR" \
    --strategy baseline \
    --output-dir "$RESULTS_DIR/baseline" \
    --no-correction

echo ""
echo "✓ Phase 1 完成"
echo ""

# ============================================================
# Phase 2: 数据漂移
# ============================================================
echo "========================================"
echo "Phase 2: 数据漂移（分布反转）"
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
echo "✓ Phase 2 完成"
echo ""

# ============================================================
# Phase 3: Stale Prior（过期直方图）
# ============================================================
echo "========================================"
echo "Phase 3: Stale Prior（过期直方图）"
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
echo "✓ Phase 3 完成"
echo ""

# ============================================================
# Phase 4: Full ANALYZE（新鲜直方图）
# ============================================================
echo "========================================"
echo "Phase 4: Full ANALYZE（新鲜直方图）"
echo "========================================"
echo ""

echo "[4.1] 重新 ANALYZE..."
for table in $TABLES; do
    echo "  ANALYZE $table..."
    $PRESTO_CLI --server $PRESTO_HOST --catalog $CATALOG --schema $SCHEMA \
        --execute "SET SESSION iceberg.statistics_kll_sketch_k_parameter = 1024; ANALYZE $table;" 2>&1 | tail -1
done

echo ""
echo "[4.2] 触发新 Snapshot..."
cd "$SCRIPT_DIR/drift"
if [ -f trigger_new_snapshot.sh ]; then
    ./trigger_new_snapshot.sh "$TABLES"
else
    for table in $TABLES; do
        $PRESTO_CLI --server $PRESTO_HOST --catalog $CATALOG --schema $SCHEMA \
            --execute "INSERT INTO $table SELECT * FROM $table LIMIT 1;" 2>&1 | tail -1
        $PRESTO_CLI --server $PRESTO_HOST --catalog $CATALOG --schema $SCHEMA \
            --execute "DELETE FROM $table WHERE id IN (SELECT id FROM $table ORDER BY id DESC LIMIT 1);" 2>&1 | tail -1
    done
fi

echo ""
echo "[4.3] 运行 IFW 查询..."
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
echo "✓ Phase 4 完成"
echo ""

# ============================================================
# Phase 5: 结果对比
# ============================================================
echo "========================================"
echo "Phase 5: 结果对比"
echo "========================================"
echo ""

python3 << 'EOF'
import json
import os

results_dir = os.environ.get('RESULTS_DIR', 'results/ifw')

strategies = {}
for strategy in ['baseline', 'stale_prior', 'full_analyze']:
    result_file = f"{results_dir}/{strategy}/{strategy}_results.json"
    if os.path.exists(result_file):
        with open(result_file) as f:
            strategies[strategy] = json.load(f)

if len(strategies) < 2:
    print("⚠️  需要至少 2 个策略的结果才能对比")
    exit(0)

print("=" * 80)
print("IMDB Filter Workload (IFW) 实验结果")
print("=" * 80)

# 每个策略的汇总
for name, data in strategies.items():
    results = data['results']
    success = [r for r in results if r['status'] == 'success']
    total_time = sum(r['execution_time_ms'] for r in success)
    qerror_results = [r for r in success if r.get('qerror')]

    print(f"\n{name}:")
    print(f"  成功: {len(success)}/{len(results)}")
    print(f"  总时间: {total_time/1000:.1f}s")

    if qerror_results:
        mean_qerrors = [r['qerror']['mean'] for r in qerror_results]
        max_qerrors = [r['qerror']['max'] for r in qerror_results]
        print(f"  平均 Q-error: {sum(mean_qerrors)/len(mean_qerrors):.2f}")
        print(f"  最大 Q-error (avg): {sum(max_qerrors)/len(max_qerrors):.2f}")
        print(f"  最大 Q-error (max): {max(max_qerrors):.2f}")

# 逐查询对比
if 'stale_prior' in strategies and 'full_analyze' in strategies:
    stale = {r['query_id']: r for r in strategies['stale_prior']['results']}
    fresh = {r['query_id']: r for r in strategies['full_analyze']['results']}

    print(f"\n{'='*80}")
    print(f"逐查询 Q-error 对比 (Stale Prior vs Full ANALYZE)")
    print(f"{'='*80}")
    print(f"{'查询':<35} {'Stale Q-err':>12} {'Fresh Q-err':>12} {'改善':>8}")
    print(f"{'-'*35} {'-'*12} {'-'*12} {'-'*8}")

    for qid in sorted(stale.keys()):
        if qid in fresh:
            s = stale[qid]
            f = fresh[qid]
            if s.get('qerror') and f.get('qerror'):
                sq = s['qerror']['mean']
                fq = f['qerror']['mean']
                improvement = (sq - fq) / sq * 100 if sq > 0 else 0
                marker = " ***" if sq > 5 and improvement > 50 else ""
                print(f"{qid:<35} {sq:>12.2f} {fq:>12.2f} {improvement:>7.1f}%{marker}")

    print(f"\n*** = 直方图修正效果显著的查询")

print(f"\n{'='*80}")
print("实验完成")
print(f"{'='*80}")
EOF

echo ""
echo "========================================"
echo "IFW 实验完成"
echo "========================================"
echo "结果目录: $RESULTS_DIR/"
echo ""
