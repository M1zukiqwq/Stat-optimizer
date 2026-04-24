#!/bin/bash
#
# 完整的 Q-error 对比实验流程
#
# 步骤：
# 1. 注入漂移（inject drift）
# 2. 提取漂移后的 Q-error（before ANALYZE）
# 3. 运行 ANALYZE 更新统计信息
# 4. 提取 ANALYZE 后的 Q-error（after ANALYZE）
# 5. 对比并生成报告
#

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数
# ============================================================================

PRESTO_HOST="${PRESTO_HOST:-localhost:8080}"
CATALOG="${CATALOG:-iceberg}"
SCHEMA="${SCHEMA:-imdb}"
USER="${USER:-tianqc}"

# 实验目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
QUERY_DIR="$PROJECT_ROOT/queries/job"
OUTPUT_DIR="$PROJECT_ROOT/results/qerror_experiment_$(date +%Y%m%d_%H%M%S)"

# 漂移参数
DRIFT_ROUNDS="${DRIFT_ROUNDS:-15}"
DRIFT_RATIO="${DRIFT_RATIO:-0.02}"

# 是否使用直方图
USE_HISTOGRAMS="${USE_HISTOGRAMS:-true}"

echo "============================================================================"
echo "Q-error Comparison Experiment"
echo "============================================================================"
echo "Presto: $PRESTO_HOST"
echo "Catalog: $CATALOG"
echo "Schema: $SCHEMA"
echo "Query dir: $QUERY_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Drift: $DRIFT_ROUNDS rounds × $DRIFT_RATIO ratio"
echo "Use histograms: $USE_HISTOGRAMS"
echo "============================================================================"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Step 1: 注入漂移
# ============================================================================

echo "Step 1: Injecting drift..."
echo "----------------------------------------"

python3 "$PROJECT_ROOT/drift/inject_drift.py" \
    --host "${PRESTO_HOST%:*}" \
    --port "${PRESTO_HOST#*:}" \
    --catalog "$CATALOG" \
    --schema "$SCHEMA" \
    --user "$USER" \
    --rounds "$DRIFT_ROUNDS" \
    --drift-ratio "$DRIFT_RATIO" \
    --no-delete \
    --no-update \
    2>&1 | tee "$OUTPUT_DIR/01_inject_drift.log"

echo ""
echo "✓ Drift injection complete"
echo ""

# ============================================================================
# Step 2: 提取漂移后的 Q-error（Before ANALYZE）
# ============================================================================

echo "Step 2: Extracting Q-error BEFORE ANALYZE..."
echo "----------------------------------------"

BEFORE_OUTPUT="$OUTPUT_DIR/before_analyze"
mkdir -p "$BEFORE_OUTPUT"

HISTOGRAM_FLAG=""
if [ "$USE_HISTOGRAMS" = "true" ]; then
    HISTOGRAM_FLAG="--use-histograms"
else
    HISTOGRAM_FLAG="--no-histograms"
fi

python3 "$SCRIPT_DIR/extract_qerror.py" \
    --presto-host "$PRESTO_HOST" \
    --catalog "$CATALOG" \
    --schema "$SCHEMA" \
    --user "$USER" \
    --query-dir "$QUERY_DIR" \
    --output-dir "$BEFORE_OUTPUT" \
    $HISTOGRAM_FLAG \
    2>&1 | tee "$OUTPUT_DIR/02_qerror_before.log"

echo ""
echo "✓ Q-error extraction (before ANALYZE) complete"
echo ""

# ============================================================================
# Step 3: 运行 ANALYZE 更新统计信息
# ============================================================================

echo "Step 3: Running ANALYZE to update statistics..."
echo "----------------------------------------"

# 使用现有的 analyze_tables.sh 脚本
if [ -f "$PROJECT_ROOT/drift/analyze_tables.sh" ]; then
    bash "$PROJECT_ROOT/drift/analyze_tables.sh" \
        2>&1 | tee "$OUTPUT_DIR/03_analyze.log"
else
    echo "⚠️  analyze_tables.sh not found, running manual ANALYZE..."

    # 手动 ANALYZE 主要表
    TABLES=(
        "cast_info"
        "movie_info"
        "movie_keyword"
        "name"
        "title"
        "movie_companies"
        "movie_info_idx"
    )

    for table in "${TABLES[@]}"; do
        echo "  Analyzing $table..."
        presto-cli --server "$PRESTO_HOST" \
            --catalog "$CATALOG" \
            --schema "$SCHEMA" \
            --execute "ANALYZE $table" \
            2>&1 | tee -a "$OUTPUT_DIR/03_analyze.log"
    done
fi

echo ""
echo "✓ ANALYZE complete"
echo ""

# ============================================================================
# Step 4: 提取 ANALYZE 后的 Q-error（After ANALYZE）
# ============================================================================

echo "Step 4: Extracting Q-error AFTER ANALYZE..."
echo "----------------------------------------"

AFTER_OUTPUT="$OUTPUT_DIR/after_analyze"
mkdir -p "$AFTER_OUTPUT"

python3 "$SCRIPT_DIR/extract_qerror.py" \
    --presto-host "$PRESTO_HOST" \
    --catalog "$CATALOG" \
    --schema "$SCHEMA" \
    --user "$USER" \
    --query-dir "$QUERY_DIR" \
    --output-dir "$AFTER_OUTPUT" \
    $HISTOGRAM_FLAG \
    2>&1 | tee "$OUTPUT_DIR/04_qerror_after.log"

echo ""
echo "✓ Q-error extraction (after ANALYZE) complete"
echo ""

# ============================================================================
# Step 5: 对比并生成报告
# ============================================================================

echo "Step 5: Comparing Q-error and generating report..."
echo "----------------------------------------"

COMPARISON_OUTPUT="$OUTPUT_DIR/comparison"
mkdir -p "$COMPARISON_OUTPUT"

python3 "$SCRIPT_DIR/compare_qerror.py" \
    --before "$BEFORE_OUTPUT/qerror_results.json" \
    --after "$AFTER_OUTPUT/qerror_results.json" \
    --output-dir "$COMPARISON_OUTPUT" \
    2>&1 | tee "$OUTPUT_DIR/05_comparison.log"

echo ""
echo "✓ Comparison complete"
echo ""

# ============================================================================
# 完成
# ============================================================================

echo "============================================================================"
echo "Experiment Complete!"
echo "============================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key outputs:"
echo "  - Before ANALYZE Q-error: $BEFORE_OUTPUT/qerror_results.json"
echo "  - After ANALYZE Q-error:  $AFTER_OUTPUT/qerror_results.json"
echo "  - Comparison summary:     $COMPARISON_OUTPUT/qerror_comparison_summary.csv"
echo "  - Comparison detail:      $COMPARISON_OUTPUT/qerror_comparison_detail.csv"
echo "  - CDF data:               $COMPARISON_OUTPUT/qerror_cdf_data.csv"
echo ""
echo "LaTeX tables:"
echo "  - $COMPARISON_OUTPUT/qerror_comparison_summary.tex"
echo ""
echo "============================================================================"
