#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PG_HOST="${PG_HOST:-localhost}"
PG_PORT="${PG_PORT:-5433}"
PG_USER="${PG_USER:-postgres}"
PG_DB="${PG_DB:-tpcds}"
PG_PASSWORD="${PG_PASSWORD:-}"
export PGPASSWORD="${PGPASSWORD:-$PG_PASSWORD}"
PG_PASSWORD="$PGPASSWORD"
TPCDS_SF="${TPCDS_SF:-5}"

if [[ -n "${QUERY_DIR:-}" ]]; then
    QUERY_DIR="$QUERY_DIR"
elif [[ -d "$SCRIPT_DIR/generated_queries_pg_sf${TPCDS_SF}" ]]; then
    QUERY_DIR="$SCRIPT_DIR/generated_queries_pg_sf${TPCDS_SF}"
elif [[ -d "$SCRIPT_DIR/generated_queries_pg" ]]; then
    QUERY_DIR="$SCRIPT_DIR/generated_queries_pg"
elif [[ -d "$SCRIPT_DIR/generated_queries" ]]; then
    QUERY_DIR="$SCRIPT_DIR/generated_queries"
else
    QUERY_DIR="$SCRIPT_DIR/generated_queries_pg_sf${TPCDS_SF}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results/tpcds_histogram_comparison_sf${TPCDS_SF}}"
DRIFT_ROUNDS="${DRIFT_ROUNDS:-10}"
DRIFT_RATIO="${DRIFT_RATIO:-0.05}"
TIMEOUT="${TIMEOUT:-30}"
DRIFT_STATE_TABLE="${DRIFT_STATE_TABLE:-tpcds_experiment_state}"
DRIFT_STATE_KEY="${DRIFT_STATE_KEY:-tpcds_histogram_scd2_sf${TPCDS_SF}}"
SLOWDOWN_THRESHOLD="${SLOWDOWN_THRESHOLD:-0.20}"
SUMMARY_SCRIPT="$SCRIPT_DIR/tools/summarize_strategy_regressions.py"

if [[ -n "${RUNNER_PATH:-}" ]]; then
    RUNNER="$RUNNER_PATH"
elif [[ -f "$SCRIPT_DIR/experiment/run_simple_pg_experiment.py" ]]; then
    RUNNER="$SCRIPT_DIR/experiment/run_simple_pg_experiment.py"
elif [[ -f "$REPO_ROOT/job_experiment/experiment/run_simple_pg_experiment.py" ]]; then
    RUNNER="$REPO_ROOT/job_experiment/experiment/run_simple_pg_experiment.py"
elif [[ -f "$REPO_ROOT/legacy_experiments_backup/2026-03-12/job_experiment/experiment/run_simple_pg_experiment.py" ]]; then
    RUNNER="$REPO_ROOT/legacy_experiments_backup/2026-03-12/job_experiment/experiment/run_simple_pg_experiment.py"
else
    echo "Experiment runner not found. Set RUNNER_PATH explicitly."
    exit 1
fi

TPCDS_TABLES=(item customer store_sales)

_psql_scalar() {
    local sql="$1"
    psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -Atqc "$sql"
}

ensure_drift_state_table() {
    _psql_scalar "
        CREATE TABLE IF NOT EXISTS ${DRIFT_STATE_TABLE} (
            experiment_name text PRIMARY KEY,
            drift_injected_at timestamptz NOT NULL,
            drift_rounds integer NOT NULL,
            drift_ratio double precision NOT NULL
        );
    " > /dev/null
}

recorded_drift_count() {
    _psql_scalar "
        SELECT COUNT(*)
        FROM ${DRIFT_STATE_TABLE}
        WHERE experiment_name = '${DRIFT_STATE_KEY}';
    "
}

mark_drift_injected() {
    _psql_scalar "
        INSERT INTO ${DRIFT_STATE_TABLE} (experiment_name, drift_injected_at, drift_rounds, drift_ratio)
        VALUES ('${DRIFT_STATE_KEY}', now(), ${DRIFT_ROUNDS}, ${DRIFT_RATIO})
        ON CONFLICT (experiment_name)
        DO UPDATE SET
            drift_injected_at = EXCLUDED.drift_injected_at,
            drift_rounds = EXCLUDED.drift_rounds,
            drift_ratio = EXCLUDED.drift_ratio;
    " > /dev/null
}

run_queries() {
    local strategy="$1"
    local run_name="$2"

    echo "Running queries for $run_name..."
    python3 "$RUNNER" \
        --host "$PG_HOST" \
        --port "$PG_PORT" \
        --dbname "$PG_DB" \
        --user "$PG_USER" \
        --password "$PG_PASSWORD" \
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
echo "TPC-DS SCD Type 2 Histogram Benchmark"
echo "=================================="
echo "Database: $PG_USER@$PG_HOST:$PG_PORT/$PG_DB"
echo "Scale:    SF=$TPCDS_SF"
echo "Runner:   $RUNNER"
echo "Queries:  $QUERY_DIR"
echo "Output:   $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

if [[ ! -d "$QUERY_DIR" ]]; then
    echo "Query directory not found: $QUERY_DIR"
    echo "Set QUERY_DIR to a directory containing generated TPC-DS .sql files."
    exit 1
fi

ensure_drift_state_table

echo "Checking drift status..."
DRIFT_RECORDED_COUNT="$(recorded_drift_count 2>/dev/null || echo 0)"
DRIFT_RECORDED_COUNT="$(echo "$DRIFT_RECORDED_COUNT" | xargs)"

if [[ "$DRIFT_RECORDED_COUNT" -eq 0 ]]; then
    echo "  ⚠ No experiment drift marker found. Injecting SCD2 drift ($DRIFT_ROUNDS rounds)..."
    python3 "$SCRIPT_DIR/drift/inject_scd2_drift_pg.py" \
        --dbname "$PG_DB" \
        --user "$PG_USER" \
        --password "$PG_PASSWORD" \
        --host "$PG_HOST" \
        --port "$PG_PORT" \
        --rounds "$DRIFT_ROUNDS" \
        --drift-ratio "$DRIFT_RATIO"
    mark_drift_injected
    echo "  ✓ SCD2 Drift injected"
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
echo "Setting analyze_only_histogram = on and running ANALYZE..."
for table in "${TPCDS_TABLES[@]}"; do
    psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" \
        -c "SET analyze_only_histogram = on; ANALYZE $table;" > /dev/null 2>&1 || true
done
run_queries "histogram_only" "OASIS Updates"

echo ""
echo "=================================="
echo "Step 3: Full ANALYZE"
echo "=================================="
echo "Setting analyze_only_histogram = off and running ANALYZE..."
for table in "${TPCDS_TABLES[@]}"; do
    psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" \
        -c "SET analyze_only_histogram = off; ANALYZE $table;" > /dev/null 2>&1 || true
done
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
printf "| %-14s | %11ss | %8s |\n" "OASIS" "$histogram_only_TIME" "$histogram_only_QERROR"
printf "| %-14s | %11ss | %8s |\n" "Full ANALYZE" "$full_analyze_TIME" "$full_analyze_QERROR"
echo "+----------------+---------------+----------+"

echo ""
echo "Detailed results saved to: $OUTPUT_DIR/"
echo "Done!"
