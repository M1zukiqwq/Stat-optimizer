#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MYSQL_HOST="${MYSQL_HOST:-localhost}"
MYSQL_PORT="${MYSQL_PORT:-3306}"
MYSQL_USER="${MYSQL_USER:-root}"
MYSQL_DB="${MYSQL_DB:-tpcds}"
MYSQL_PASSWORD="${MYSQL_PASSWORD:-tianqichu123}"
export MYSQL_PWD="${MYSQL_PWD:-$MYSQL_PASSWORD}"
MYSQL_PASSWORD="$MYSQL_PWD"
TPCDS_SF="${TPCDS_SF:-5}"

if [[ -n "${QUERY_DIR:-}" ]]; then
    QUERY_DIR="$QUERY_DIR"
elif [[ -d "$SCRIPT_DIR/generated_queries_mysql_sf${TPCDS_SF}" ]]; then
    QUERY_DIR="$SCRIPT_DIR/generated_queries_mysql_sf${TPCDS_SF}"
elif [[ -d "$SCRIPT_DIR/generated_queries_mysql" ]]; then
    QUERY_DIR="$SCRIPT_DIR/generated_queries_mysql"
else
    QUERY_DIR="$SCRIPT_DIR/generated_queries_mysql_sf${TPCDS_SF}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/tpcds_histogram_comparison_mysql_sf${TPCDS_SF}}"
DRIFT_ROUNDS="${DRIFT_ROUNDS:-10}"
DRIFT_RATIO="${DRIFT_RATIO:-0.05}"
TIMEOUT="${TIMEOUT:-30}"
DRIFT_STATE_TABLE="${DRIFT_STATE_TABLE:-tpcds_experiment_state_mysql}"
DRIFT_STATE_KEY="${DRIFT_STATE_KEY:-tpcds_histogram_scd2_mysql_sf${TPCDS_SF}}"
HISTOGRAM_BUCKETS="${HISTOGRAM_BUCKETS:-256}"
SLOWDOWN_THRESHOLD="${SLOWDOWN_THRESHOLD:-0.20}"
SUMMARY_SCRIPT="$SCRIPT_DIR/tools/summarize_strategy_regressions.py"
RUNNER="${RUNNER_PATH:-$SCRIPT_DIR/experiment/run_simple_mysql_experiment.py}"

mysql_scalar() {
    local sql="$1"
    mysql --host="$MYSQL_HOST" --port="$MYSQL_PORT" --user="$MYSQL_USER" --database="$MYSQL_DB" --batch --skip-column-names -e "$sql"
}

ensure_state_table() {
    mysql_scalar "
        CREATE TABLE IF NOT EXISTS ${DRIFT_STATE_TABLE} (
            experiment_name varchar(128) PRIMARY KEY,
            drift_injected_at datetime NOT NULL,
            drift_rounds int NOT NULL,
            drift_ratio double NOT NULL
        );
    " > /dev/null
}

recorded_drift_count() {
    mysql_scalar "SELECT COUNT(*) FROM ${DRIFT_STATE_TABLE} WHERE experiment_name = '${DRIFT_STATE_KEY}';"
}

mark_drift_injected() {
    mysql_scalar "
        INSERT INTO ${DRIFT_STATE_TABLE} (experiment_name, drift_injected_at, drift_rounds, drift_ratio)
        VALUES ('${DRIFT_STATE_KEY}', NOW(), ${DRIFT_ROUNDS}, ${DRIFT_RATIO})
        ON DUPLICATE KEY UPDATE
            drift_injected_at = VALUES(drift_injected_at),
            drift_rounds = VALUES(drift_rounds),
            drift_ratio = VALUES(drift_ratio);
    " > /dev/null
}

update_histograms_only() {
    mysql --host="$MYSQL_HOST" --port="$MYSQL_PORT" --user="$MYSQL_USER" --database="$MYSQL_DB" \
        -e "ANALYZE TABLE item UPDATE HISTOGRAM ON i_item_sk WITH ${HISTOGRAM_BUCKETS} BUCKETS;" >/dev/null 2>&1 || true
    mysql --host="$MYSQL_HOST" --port="$MYSQL_PORT" --user="$MYSQL_USER" --database="$MYSQL_DB" \
        -e "ANALYZE TABLE customer UPDATE HISTOGRAM ON c_customer_sk WITH ${HISTOGRAM_BUCKETS} BUCKETS;" >/dev/null 2>&1 || true
    mysql --host="$MYSQL_HOST" --port="$MYSQL_PORT" --user="$MYSQL_USER" --database="$MYSQL_DB" \
        -e "ANALYZE TABLE store_sales UPDATE HISTOGRAM ON ss_item_sk, ss_customer_sk WITH ${HISTOGRAM_BUCKETS} BUCKETS;" >/dev/null 2>&1 || true
}

run_full_analyze() {
    mysql --host="$MYSQL_HOST" --port="$MYSQL_PORT" --user="$MYSQL_USER" --database="$MYSQL_DB" \
        -e "ANALYZE TABLE item, customer, store_sales;" >/dev/null 2>&1 || true
    update_histograms_only
}

run_queries() {
    local strategy="$1"
    local run_name="$2"

    echo "Running queries for $run_name..."
    python3 "$RUNNER" \
        --host "$MYSQL_HOST" \
        --port "$MYSQL_PORT" \
        --dbname "$MYSQL_DB" \
        --user "$MYSQL_USER" \
        --password "$MYSQL_PASSWORD" \
        --query-dir "$QUERY_DIR" \
        --strategy "$strategy" \
        --output-dir "$OUTPUT_DIR" \
        --timeout "$TIMEOUT" || true

    local total_time
    total_time="$(python3 -c "
import json
try:
    with open('$OUTPUT_DIR/${strategy}_results.json') as f:
        data = json.load(f)
    total = sum(r['execution_time_ms'] for r in data['results'] if r['status'] == 'success')
    print(f'{total/1000:.1f}')
except Exception:
    print('N/A')
" 2>/dev/null || echo "N/A")"

    local q_error
    q_error="$(python3 -c "
import json, math
try:
    with open('$OUTPUT_DIR/${strategy}_results.json') as f:
        data = json.load(f)
    qerrors = [r['qerror']['geometric_mean'] for r in data['results'] if r['status'] == 'success' and r.get('qerror')]
    if qerrors:
        geom_mean = math.exp(sum(math.log(q) for q in qerrors) / len(qerrors))
        print(f'{geom_mean:.2f}')
    else:
        print('N/A')
except Exception:
    print('N/A')
" 2>/dev/null || echo "N/A")"

    echo "  -> Time: ${total_time}s, Q-Error: $q_error"
    printf -v "${strategy}_TIME" '%s' "$total_time"
    printf -v "${strategy}_QERROR" '%s' "$q_error"
}

run_regression_summary() {
    local compare_label="$1"
    python3 "$SUMMARY_SCRIPT" \
        --stale-results "$OUTPUT_DIR/stale_prior_results.json" \
        --compare-results "$OUTPUT_DIR/${compare_label}_results.json" \
        --label "$compare_label" \
        --threshold "$SLOWDOWN_THRESHOLD" \
        --output-dir "$OUTPUT_DIR"
}

echo "=================================="
echo "MySQL TPC-DS SCD Type 2 Histogram Benchmark"
echo "=================================="
echo "Database: $MYSQL_USER@$MYSQL_HOST:$MYSQL_PORT/$MYSQL_DB"
echo "Scale:    SF=$TPCDS_SF"
echo "Queries:  $QUERY_DIR"
echo "Output:   $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

if [[ ! -d "$QUERY_DIR" ]]; then
    echo "Query directory not found: $QUERY_DIR"
    echo "Set QUERY_DIR to a directory containing generated TPC-DS .sql files."
    exit 1
fi

ensure_state_table

echo "Checking drift status..."
DRIFT_RECORDED_COUNT="$(recorded_drift_count 2>/dev/null || echo 0)"
DRIFT_RECORDED_COUNT="$(echo "$DRIFT_RECORDED_COUNT" | xargs)"

if [[ "$DRIFT_RECORDED_COUNT" -eq 0 ]]; then
    echo "  ⚠ No experiment drift marker found. Injecting SCD2 drift (${DRIFT_ROUNDS} rounds)..."
    python3 "$SCRIPT_DIR/drift/inject_scd2_drift_mysql.py" \
        --dbname "$MYSQL_DB" \
        --user "$MYSQL_USER" \
        --password "$MYSQL_PASSWORD" \
        --host "$MYSQL_HOST" \
        --port "$MYSQL_PORT" \
        --rounds "$DRIFT_ROUNDS" \
        --drift-ratio "$DRIFT_RATIO"
    mark_drift_injected
    echo "  ✓ SCD2 drift injected"
else
    echo "  ✓ Drift is already present (state_rows=${DRIFT_RECORDED_COUNT})"
fi

echo ""
echo "=================================="
echo "Step 1: Stale Prior Test"
echo "=================================="
run_queries "stale_prior" "Stale Statistics"

echo ""
echo "=================================="
echo "Step 2: Histogram Update Only"
echo "=================================="
echo "Updating MySQL histograms only..."
update_histograms_only
run_queries "histogram_only" "Histogram Updates"

echo ""
echo "=================================="
echo "Step 3: Full ANALYZE"
echo "=================================="
echo "Running ANALYZE TABLE plus histogram refresh..."
run_full_analyze
run_queries "full_analyze" "Full ANALYZE Updates"

echo ""
echo "=================================="
echo "Step 4: Regression Summary vs Stale Prior"
echo "=================================="
run_regression_summary "histogram_only"
run_regression_summary "full_analyze"

echo ""
echo "=================================="
echo "Results Summary"
echo "=================================="
echo ""
echo "+----------------+---------------+----------+"
echo "| Strategy       | Total Time    | Q-Error  |"
echo "+----------------+---------------+----------+"
printf "| %-14s | %11ss | %8s |\n" "Stale Prior" "$stale_prior_TIME" "$stale_prior_QERROR"
printf "| %-14s | %11ss | %8s |\n" "Histogram Only" "$histogram_only_TIME" "$histogram_only_QERROR"
printf "| %-14s | %11ss | %8s |\n" "Full ANALYZE" "$full_analyze_TIME" "$full_analyze_QERROR"
echo "+----------------+---------------+----------+"

echo ""
echo "Detailed results saved to: $OUTPUT_DIR/"
echo "Done!"
