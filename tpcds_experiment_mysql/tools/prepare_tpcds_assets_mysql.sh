#!/usr/bin/env bash
set -euo pipefail

KIT_DIR=""
SCALE=5
DATA_DIR=""
RAW_QUERY_DIR=""
FIXED_QUERY_DIR=""
DIALECT="netezza"

usage() {
  cat <<USAGE
Usage: $(basename "$0") --kit-dir PATH [options]

Options:
  --kit-dir PATH         Path to tpcds-kit root
  --scale N              TPC-DS scale factor (default: 5)
  --data-dir PATH        Output directory for .dat files
  --raw-query-dir PATH   Output directory for raw generated SQL files
  --fixed-query-dir PATH Output directory for MySQL-fixed SQL files
  --dialect NAME         dsqgen dialect (default: netezza)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kit-dir) KIT_DIR="$2"; shift 2 ;;
    --scale) SCALE="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --raw-query-dir) RAW_QUERY_DIR="$2"; shift 2 ;;
    --fixed-query-dir) FIXED_QUERY_DIR="$2"; shift 2 ;;
    --dialect) DIALECT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$KIT_DIR" ]]; then
  echo "--kit-dir is required"
  usage
  exit 1
fi

KIT_DIR="$(cd "$KIT_DIR" && pwd)"
TOOLS_DIR="$KIT_DIR/tools"
QUERY_TEMPLATES_DIR="$KIT_DIR/query_templates"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FIXER="$SCRIPT_DIR/fix_tpcds_queries_for_mysql.py"

if [[ -z "$DATA_DIR" ]]; then
  DATA_DIR="$KIT_DIR/generated_data_sf${SCALE}"
fi
if [[ -z "$RAW_QUERY_DIR" ]]; then
  RAW_QUERY_DIR="$EXPERIMENT_ROOT/generated_queries_mysql_raw_sf${SCALE}"
fi
if [[ -z "$FIXED_QUERY_DIR" ]]; then
  FIXED_QUERY_DIR="$EXPERIMENT_ROOT/generated_queries_mysql_sf${SCALE}"
fi

build_tools() {
  if [[ -x "$TOOLS_DIR/dsdgen" && -x "$TOOLS_DIR/dsqgen" ]]; then
    echo "TPC-DS tools already built"
    return
  fi

  pushd "$TOOLS_DIR" >/dev/null
  if [[ "$(uname -s)" == "Darwin" ]]; then
    perl -0pi -e "s/#ifdef USE_VALUES_H\n#include <values\\.h>\n#endif/#if defined(USE_VALUES_H) && !defined(MACOS)\n#include <values.h>\n#endif\n#ifdef MACOS\n#include <float.h>\n#endif/" porting.h
    make clean >/dev/null 2>&1 || true
    make OS=MACOS MACOS_CFLAGS='-D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -DYYDEBUG -DMACOS -std=gnu89 -g -Wall'
  else
    make clean >/dev/null 2>&1 || true
    make OS=LINUX
  fi
  popd >/dev/null
}

generate_data() {
  mkdir -p "$DATA_DIR"
  pushd "$TOOLS_DIR" >/dev/null
  ./dsdgen -SCALE "$SCALE" -FORCE Y -DIR "$DATA_DIR"
  popd >/dev/null
}

generate_queries() {
  rm -rf "$RAW_QUERY_DIR"
  mkdir -p "$RAW_QUERY_DIR"
  pushd "$TOOLS_DIR" >/dev/null
  for i in $(seq 1 99); do
    ./dsqgen \
      -DIRECTORY "$QUERY_TEMPLATES_DIR" \
      -TEMPLATE "query${i}.tpl" \
      -COUNT 1 \
      -SCALE "$SCALE" \
      -DIALECT "$DIALECT" \
      -OUTPUT_DIR "$RAW_QUERY_DIR" >/dev/null
    mv "$RAW_QUERY_DIR/query_0.sql" "$RAW_QUERY_DIR/query${i}.sql"
  done
  popd >/dev/null
}

fix_queries() {
  python3 "$FIXER" --input-dir "$RAW_QUERY_DIR" --output-dir "$FIXED_QUERY_DIR"
}

build_tools
generate_data
generate_queries
fix_queries

echo "Done."
echo "Data dir:        $DATA_DIR"
echo "Raw query dir:   $RAW_QUERY_DIR"
echo "Fixed query dir: $FIXED_QUERY_DIR"
