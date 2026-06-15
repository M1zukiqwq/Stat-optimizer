#!/usr/bin/env bash
# Load the JOB/IMDB dataset into a PostgreSQL database for the cross-column experiment.
#
# Prerequisites: a running PostgreSQL (>=10), and the IMDB tables as delimited files named
# <table>.dat in $IMDB_DAT_DIR (one file per table; see schema_load.sql for the 21 tables).
# The files are assumed pipe-delimited with empty fields = NULL (the format used in the paper).
# For the canonical comma-CSV IMDB dump (gregrahn/join-order-benchmark + imdb.tgz), set
# IMDB_DELIM=',' (and note those CSVs use '"' quoting + '\' escape, so adjust the \copy
# options below if a row fails to parse).
#
# Connection comes from standard libpq env vars (PGHOST/PGPORT/PGUSER) plus:
#   PSQL_BIN      psql binary            (default: psql)
#   PGDATABASE    target database        (default: imdb)
#   IMDB_DAT_DIR  dir with <table>.dat   (REQUIRED)
#   IMDB_DELIM    field delimiter        (default: |)
#
# Example:
#   PSQL_BIN=/usr/lib/postgresql/16/bin/psql PGHOST=/tmp PGPORT=5432 \
#   IMDB_DAT_DIR=~/job/imdb_pipe ./load_imdb.sh
set -euo pipefail
cd "$(dirname "$0")"
PSQL_BIN="${PSQL_BIN:-psql}"
DB="${PGDATABASE:-imdb}"
DELIM="${IMDB_DELIM:-|}"
: "${IMDB_DAT_DIR:?set IMDB_DAT_DIR to the directory holding <table>.dat files}"

TABLES="aka_name aka_title cast_info char_name comp_cast_type company_name company_type \
complete_cast info_type keyword kind_type link_type movie_companies movie_info \
movie_info_idx movie_keyword movie_link name person_info role_type title"

P="$PSQL_BIN -v ON_ERROR_STOP=1"
echo "=== create database $DB (if absent) + schema ==="
$PSQL_BIN -d postgres -c "CREATE DATABASE $DB" 2>/dev/null || echo "  ($DB already exists)"
$P -d "$DB" -q -f schema_load.sql

echo "=== COPY tables from $IMDB_DAT_DIR (delim='$DELIM') ==="
LOAD=$(mktemp)
{
  echo "\\set ON_ERROR_STOP on"
  echo "SET synchronous_commit = off;"
  for t in $TABLES; do
    echo "\\echo loading $t"
    echo "\\copy $t FROM '$IMDB_DAT_DIR/$t.dat' WITH (FORMAT csv, DELIMITER e'$DELIM', NULL '', QUOTE e'\\b', ESCAPE e'\\b')"
  done
} > "$LOAD"
$P -d "$DB" -f "$LOAD"
rm -f "$LOAD"

echo "=== indexes (speed up the experiment's COUNT predicates) + ANALYZE ==="
$P -d "$DB" -c "CREATE INDEX IF NOT EXISTS ix_title_ky  ON title(kind_id, production_year)"
$P -d "$DB" -c "CREATE INDEX IF NOT EXISTS ix_akat_ky   ON aka_title(kind_id, production_year)"
$P -d "$DB" -c "CREATE INDEX IF NOT EXISTS ix_cast_rn   ON cast_info(role_id, nr_order)"
$P -d "$DB" -c "CREATE INDEX IF NOT EXISTS ix_mc_tc     ON movie_companies(company_type_id, company_id)"
$P -d "$DB" -c "CREATE INDEX IF NOT EXISTS ix_pi_ip     ON person_info(info_type_id, person_id)"
$P -d "$DB" -c "CREATE INDEX IF NOT EXISTS ix_title_se  ON title(season_nr, episode_nr)"
$P -d "$DB" -c "ANALYZE"
echo "=== done. now run: python3 crosscol_pg_inject_experiment.py ==="
