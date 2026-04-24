#!/bin/bash
# JOB Benchmark 端到端实验 - 主控脚本
#
# 使用方法：
#   ./run_full_experiment.sh [--skip-setup] [--skip-drift] [--drift-rounds 15]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 默认参数
SKIP_SETUP=false
SKIP_DRIFT=false
DRIFT_ROUNDS=15
PRESTO_HOST="localhost:8080"
CATALOG="iceberg"
SCHEMA="imdb"
FEEDBACK_DIR="/tmp/ml-feedback"
MODEL_PORT=8080

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --skip-drift)
            SKIP_DRIFT=true
            shift
            ;;
        --drift-rounds)
            DRIFT_ROUNDS="$2"
            shift 2
            ;;
        --presto-host)
            PRESTO_HOST="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "JOB Benchmark 端到端实验"
echo "========================================================================"
echo "Presto: $PRESTO_HOST"
echo "Catalog: $CATALOG"
echo "Schema: $SCHEMA"
echo "Drift rounds: $DRIFT_ROUNDS (equivalent to q=$DRIFT_ROUNDS in ablation study)"
echo "Feedback dir: $FEEDBACK_DIR"
echo "========================================================================"

# ============================================================================
# Phase 1: 环境准备
# ============================================================================

if [ "$SKIP_SETUP" = false ]; then
    echo ""
    echo "=== Phase 1: 数据准备 ==="

    # 1.1 下载 IMDB 数据集
    echo "[1/6] Downloading IMDB dataset..."
    cd "$SCRIPT_DIR/setup"
    ./1_download_imdb.sh

    # 1.2 下载 JOB 查询
    echo "[2/6] Downloading JOB queries..."
    cd "$SCRIPT_DIR/queries"
    ./download_job_queries.sh

    # 1.3 创建 Iceberg 表
    echo "[3/6] Creating Iceberg tables..."
    cd "$SCRIPT_DIR/setup"
    presto-cli --server "$PRESTO_HOST" --catalog "$CATALOG" --schema "$SCHEMA" \
        --file 2_create_tables.sql

    # 1.4 加载数据
    echo "[4/6] Loading data into Iceberg..."
    python3 3_load_data.py \
        --data-dir ./imdb_data/imdb \
        --presto-host "$PRESTO_HOST" \
        --catalog "$CATALOG" \
        --schema "$SCHEMA"

    # 1.5 生成初始统计
    echo "[5/6] Running initial ANALYZE..."
    presto-cli --server "$PRESTO_HOST" --catalog "$CATALOG" --schema "$SCHEMA" \
        --file 4_initial_analyze.sql

    # 1.6 清空 feedback 目录
    echo "[6/6] Cleaning feedback directory..."
    rm -rf "$FEEDBACK_DIR"
    mkdir -p "$FEEDBACK_DIR"

    echo "✓ Phase 1 complete"
else
    echo "=== Phase 1: SKIPPED (--skip-setup) ==="
fi

# ============================================================================
# Phase 2: 启动模型服务
# ============================================================================

echo ""
echo "=== Phase 2: 启动模型服务 ==="

# 检查模型服务是否已运行
if curl -s "http://localhost:$MODEL_PORT/health" > /dev/null 2>&1; then
    echo "Model service already running at http://localhost:$MODEL_PORT"
else
    echo "Starting model service..."
    cd "$PROJECT_ROOT/cdf_kll_ml_pipeline"

    # 检查依赖
    if ! python3 -c "import fastapi" 2>/dev/null; then
        echo "Installing dependencies..."
        pip install -r requirements-service.txt
    fi

    # 后台启动模型服务
    nohup python3 model_service.py > "$SCRIPT_DIR/results/model_service.log" 2>&1 &
    MODEL_PID=$!
    echo $MODEL_PID > "$SCRIPT_DIR/results/model_service.pid"

    # 等待服务启动
    echo "Waiting for model service to start..."
    for i in {1..30}; do
        if curl -s "http://localhost:$MODEL_PORT/health" > /dev/null 2>&1; then
            echo "✓ Model service started (PID: $MODEL_PID)"
            break
        fi
        sleep 1
    done

    if ! curl -s "http://localhost:$MODEL_PORT/health" > /dev/null 2>&1; then
        echo "✗ Failed to start model service"
        exit 1
    fi
fi

# ============================================================================
# Phase 3: Baseline 测试（无漂移）
# ============================================================================

echo ""
echo "=== Phase 3: Baseline 测试（无漂移，Fresh Statistics） ==="

cd "$SCRIPT_DIR/experiment"

python3 run_experiment.py \
    --presto-host "$PRESTO_HOST" \
    --catalog "$CATALOG" \
    --schema "$SCHEMA" \
    --query-dir ../queries/job \
    --strategy baseline \
    --output-dir ../results/baseline \
    --no-correction

echo "✓ Phase 3 complete"

# ============================================================================
# Phase 4: 注入漂移
# ============================================================================

if [ "$SKIP_DRIFT" = false ]; then
    echo ""
    echo "=== Phase 4: 注入漂移 (q=$DRIFT_ROUNDS) ==="

    cd "$SCRIPT_DIR/drift"

    python3 inject_drift.py \
        --host "${PRESTO_HOST%:*}" \
        --port "${PRESTO_HOST#*:}" \
        --catalog "$CATALOG" \
        --schema "$SCHEMA" \
        --rounds "$DRIFT_ROUNDS" \
        --drift-ratio 0.02 \
        --output ../results/drift_log.json

    echo "✓ Phase 4 complete"
else
    echo "=== Phase 4: SKIPPED (--skip-drift) ==="
fi

# ============================================================================
# Phase 5: Stale Prior 测试（有漂移，不修正）
# ============================================================================

echo ""
echo "=== Phase 5: Stale Prior 测试（有漂移，不修正） ==="

cd "$SCRIPT_DIR/experiment"

python3 run_experiment.py \
    --presto-host "$PRESTO_HOST" \
    --catalog "$CATALOG" \
    --schema "$SCHEMA" \
    --query-dir ../queries/job \
    --strategy stale_prior \
    --output-dir ../results/stale_prior \
    --no-correction

echo "✓ Phase 5 complete"

# ============================================================================
# Phase 6: OASIS 测试（有漂移，模型修正）
# ============================================================================

echo ""
echo "=== Phase 6: OASIS 测试（有漂移，模型修正） ==="

cd "$SCRIPT_DIR/experiment"

# 6.1 Warmup：收集初始 observations
echo "[6.1] Running warmup queries to collect observations..."
python3 run_experiment.py \
    --presto-host "$PRESTO_HOST" \
    --catalog "$CATALOG" \
    --schema "$SCHEMA" \
    --query-dir ../queries/job \
    --strategy warmup \
    --output-dir ../results/warmup \
    --warmup-only \
    --warmup-count 20

# 6.2 OASIS 测试
echo "[6.2] Running OASIS correction..."
python3 run_experiment.py \
    --presto-host "$PRESTO_HOST" \
    --catalog "$CATALOG" \
    --schema "$SCHEMA" \
    --query-dir ../queries/job \
    --strategy oasis \
    --output-dir ../results/oasis \
    --enable-correction

echo "✓ Phase 6 complete"

# ============================================================================
# Phase 7: Full ANALYZE 测试（理想上界）
# ============================================================================

echo ""
echo "=== Phase 7: Full ANALYZE 测试（理想上界） ==="

# 7.1 重新 ANALYZE
echo "[7.1] Running full ANALYZE..."
cd "$SCRIPT_DIR/setup"
presto-cli --server "$PRESTO_HOST" --catalog "$CATALOG" --schema "$SCHEMA" \
    --file 4_initial_analyze.sql

# 7.2 触发新 Snapshot（关键！）
echo "[7.2] Triggering new snapshot..."
echo "      This ensures Full ANALYZE test uses fresh statistics"
echo "      instead of the same statistics as Stale Prior test"
cd "$SCRIPT_DIR/drift"
if [ -f trigger_new_snapshot.sh ]; then
    ./trigger_new_snapshot.sh "title cast_info movie_info movie_companies name movie_keyword"
else
    echo "      Warning: trigger_new_snapshot.sh not found, skipping..."
    echo "      This may cause Q-error to be identical between Stale Prior and Full ANALYZE"
fi

# 7.3 测试
echo "[7.3] Running queries with fresh statistics..."
cd "$SCRIPT_DIR/experiment"
python3 run_experiment.py \
    --presto-host "$PRESTO_HOST" \
    --catalog "$CATALOG" \
    --schema "$SCHEMA" \
    --query-dir ../queries/job \
    --strategy full_analyze \
    --output-dir ../results/full_analyze \
    --no-correction

echo "✓ Phase 7 complete"

# ============================================================================
# Phase 8: 结果分析
# ============================================================================

echo ""
echo "=== Phase 8: 结果分析 ==="

cd "$SCRIPT_DIR/experiment"

python3 analyze_results.py \
    --baseline ../results/baseline \
    --stale-prior ../results/stale_prior \
    --oasis ../results/oasis \
    --full-analyze ../results/full_analyze \
    --output ../results/summary

echo "✓ Phase 8 complete"

# ============================================================================
# 清理
# ============================================================================

echo ""
echo "=== 清理 ==="

# 停止模型服务
if [ -f "$SCRIPT_DIR/results/model_service.pid" ]; then
    MODEL_PID=$(cat "$SCRIPT_DIR/results/model_service.pid")
    if kill -0 "$MODEL_PID" 2>/dev/null; then
        echo "Stopping model service (PID: $MODEL_PID)..."
        kill "$MODEL_PID"
        rm "$SCRIPT_DIR/results/model_service.pid"
    fi
fi

# ============================================================================
# 完成
# ============================================================================

echo ""
echo "========================================================================"
echo "实验完成！"
echo "========================================================================"
echo "结果目录: $SCRIPT_DIR/results/"
echo ""
echo "查看结果："
echo "  - Summary: $SCRIPT_DIR/results/summary/table4.csv"
echo "  - Plots: $SCRIPT_DIR/results/summary/*.png"
echo "  - Logs: $SCRIPT_DIR/results/*/*.log"
echo "========================================================================"
