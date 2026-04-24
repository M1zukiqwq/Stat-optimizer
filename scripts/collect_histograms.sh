#!/bin/bash
# Experiment workflow script for OASIS histogram collection
# This script automates the process of:
# 1. Running ANALYZE on a table
# 2. Collecting KLL histograms from Puffin files
# 3. Clearing observations (creating fresh JSON files)
# 4. Ready for query execution to collect new observations

set -e

# Configuration
PRESTO_CLI="${PRESTO_CLI:-presto-cli}"
WAREHOUSE="${ICEBERG_WAREHOUSE:-/path/to/warehouse}"
FEEDBACK_DIR="${ML_FEEDBACK_DIR:-/tmp/ml-feedback}"
CATALOG="${ICEBERG_CATALOG:-hive}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] TABLE_NAME

Collect KLL histograms after ANALYZE for OASIS experiments.

OPTIONS:
    -w, --warehouse PATH     Iceberg warehouse path (default: $WAREHOUSE)
    -f, --feedback-dir PATH  ML feedback directory (default: $FEEDBACK_DIR)
    -c, --catalog NAME       Catalog name (default: $CATALOG)
    --skip-analyze           Skip ANALYZE step (only collect histograms)
    --python                 Use Python script instead of Java tool
    -h, --help               Show this help message

EXAMPLES:
    # Full workflow: ANALYZE + collect histograms
    $0 my_database.my_table

    # Only collect histograms (ANALYZE already done)
    $0 --skip-analyze my_database.my_table

    # Use Python script
    $0 --python my_database.my_table

ENVIRONMENT VARIABLES:
    PRESTO_CLI              Path to presto-cli (default: presto-cli)
    ICEBERG_WAREHOUSE       Iceberg warehouse path
    ML_FEEDBACK_DIR         ML feedback output directory
    ICEBERG_CATALOG         Catalog name

EOF
    exit 1
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
SKIP_ANALYZE=false
USE_PYTHON=false
TABLE_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--warehouse)
            WAREHOUSE="$2"
            shift 2
            ;;
        -f|--feedback-dir)
            FEEDBACK_DIR="$2"
            shift 2
            ;;
        -c|--catalog)
            CATALOG="$2"
            shift 2
            ;;
        --skip-analyze)
            SKIP_ANALYZE=true
            shift
            ;;
        --python)
            USE_PYTHON=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            TABLE_NAME="$1"
            shift
            ;;
    esac
done

if [ -z "$TABLE_NAME" ]; then
    log_error "Table name is required"
    usage
fi

log_info "Starting histogram collection for table: $TABLE_NAME"
log_info "Warehouse: $WAREHOUSE"
log_info "Feedback directory: $FEEDBACK_DIR"

# Step 1: Run ANALYZE (unless skipped)
if [ "$SKIP_ANALYZE" = false ]; then
    log_info "Step 1/3: Running ANALYZE on $TABLE_NAME..."

    if ! $PRESTO_CLI --execute "ANALYZE $TABLE_NAME"; then
        log_error "ANALYZE failed"
        exit 1
    fi

    log_info "ANALYZE completed successfully"

    # Wait a bit for statistics to be written
    log_info "Waiting 2 seconds for statistics to be written..."
    sleep 2
else
    log_warn "Skipping ANALYZE step"
fi

# Step 2: Collect histograms
log_info "Step 2/3: Collecting KLL histograms from Puffin files..."

if [ "$USE_PYTHON" = true ]; then
    # Use Python script
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PYTHON_SCRIPT="$SCRIPT_DIR/sync_histograms.py"

    if [ ! -f "$PYTHON_SCRIPT" ]; then
        log_error "Python script not found: $PYTHON_SCRIPT"
        exit 1
    fi

    log_info "Using Python script: $PYTHON_SCRIPT"

    if ! python3 "$PYTHON_SCRIPT" \
        --catalog-uri "thrift://localhost:9083" \
        --warehouse "$WAREHOUSE" \
        --feedback-dir "$FEEDBACK_DIR" \
        --tables "$TABLE_NAME"; then
        log_error "Python script failed"
        exit 1
    fi
else
    # Use Java tool (TODO: implement or use simplified approach)
    log_warn "Java tool not yet implemented, using simplified approach"

    # For now, create placeholder JSON files
    # In real implementation, this would call HistogramCollector.java

    # Extract simple table name (last part after dot)
    SIMPLE_TABLE="${TABLE_NAME##*.}"
    TABLE_DIR="$FEEDBACK_DIR/$SIMPLE_TABLE"

    mkdir -p "$TABLE_DIR"

    log_info "Creating placeholder histogram files in $TABLE_DIR"
    log_warn "Note: This creates placeholder data. Implement Java tool for real histograms."

    # Create a sample JSON file (you would replace this with actual histogram extraction)
    cat > "$TABLE_DIR/sample_column.json" <<'EOF'
{
  "prior_kll": {
    "min": 0.0,
    "max": 1.0,
    "null_fraction": 0.0,
    "quantile_levels": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    "quantile_values": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
  },
  "observations": []
}
EOF
fi

log_info "Histogram collection completed"

# Step 3: Verify output
log_info "Step 3/3: Verifying output..."

SIMPLE_TABLE="${TABLE_NAME##*.}"
TABLE_DIR="$FEEDBACK_DIR/$SIMPLE_TABLE"

if [ -d "$TABLE_DIR" ]; then
    FILE_COUNT=$(find "$TABLE_DIR" -name "*.json" | wc -l)
    log_info "Found $FILE_COUNT JSON files in $TABLE_DIR"

    # Show first file as example
    FIRST_FILE=$(find "$TABLE_DIR" -name "*.json" | head -1)
    if [ -n "$FIRST_FILE" ]; then
        log_info "Example file: $FIRST_FILE"
        echo "---"
        head -20 "$FIRST_FILE"
        echo "---"
    fi
else
    log_warn "Output directory not found: $TABLE_DIR"
fi

echo ""
log_info "✓ Histogram collection complete!"
log_info ""
log_info "Next steps:"
log_info "  1. Run queries to collect observations:"
log_info "     $PRESTO_CLI --execute \"SELECT * FROM $TABLE_NAME WHERE column < 100\""
log_info ""
log_info "  2. Check collected data:"
log_info "     cat $TABLE_DIR/column_name.json"
log_info ""
log_info "  3. Run ML pipeline:"
log_info "     cd presto-cdf-simulation/cdf_kll_ml_pipeline"
log_info "     python3 predict_histogram.py --input $TABLE_DIR/column_name.json ..."
