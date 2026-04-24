#!/bin/bash
#
# Load fixed IMDB CSV data into PostgreSQL
#

set -e

export PATH="/Users/qichutian/postgres/pgsql14/bin:$PATH"

DATA_DIR="${DATA_DIR:-./imdb_data/fixed}"

echo "Loading fixed IMDB data into PostgreSQL..."
echo "Data directory: $DATA_DIR"
echo ""

# Check if fixed directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Fixed data directory not found: $DATA_DIR"
    exit 1
fi

# Function to load a table
load_table() {
    local table=$1
    local file=$2
    
    if [ ! -f "$DATA_DIR/$file" ]; then
        echo "  Warning: $file not found, skipping $table"
        return
    fi
    
    echo "  Loading $table..."
    psql -d imdb -c "\\COPY $table FROM '$DATA_DIR/$file' WITH (FORMAT csv, HEADER false, NULL '')" 2>&1 | grep -E "(COPY|ERROR)" || true
}

echo "Loading dimension tables (small)..."
load_table "comp_cast_type" "comp_cast_type.csv"
load_table "company_type" "company_type.csv"
load_table "info_type" "info_type.csv"
load_table "kind_type" "kind_type.csv"
load_table "link_type" "link_type.csv"
load_table "role_type" "role_type.csv"

echo ""
echo "Loading medium tables..."
load_table "keyword" "keyword.csv"
load_table "company_name" "company_name.csv"
load_table "movie_link" "movie_link.csv"
load_table "complete_cast" "complete_cast.csv"

echo ""
echo "Loading name tables..."
load_table "name" "name.csv"
load_table "char_name" "char_name.csv"
load_table "aka_name" "aka_name.csv"

echo ""
echo "Loading title tables..."
load_table "title" "title.csv"
load_table "aka_title" "aka_title.csv"

echo ""
echo "Loading movie relationship tables..."
load_table "movie_companies" "movie_companies.csv"
load_table "movie_keyword" "movie_keyword.csv"
load_table "movie_info_idx" "movie_info_idx.csv"

echo ""
echo "Loading large info tables..."
load_table "person_info" "person_info.csv"
load_table "movie_info" "movie_info.csv"

echo ""
echo "Loading largest table (cast_info)..."
load_table "cast_info" "cast_info.csv"

echo ""
echo "Running ANALYZE on all tables..."
psql -d imdb <<'EOF'
ANALYZE title;
ANALYZE cast_info;
ANALYZE movie_info;
ANALYZE movie_keyword;
ANALYZE name;
ANALYZE char_name;
ANALYZE person_info;
ANALYZE movie_companies;
ANALYZE movie_info_idx;
ANALYZE aka_name;
ANALYZE aka_title;
ANALYZE complete_cast;
ANALYZE movie_link;
ANALYZE company_name;
ANALYZE keyword;
ANALYZE comp_cast_type;
ANALYZE company_type;
ANALYZE info_type;
ANALYZE kind_type;
ANALYZE link_type;
ANALYZE role_type;
EOF

echo ""
echo "Table row counts:"
psql -d imdb -c "
SELECT 
    relname as table_name,
    n_live_tup as row_count
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_live_tup DESC;
"

echo ""
echo "✓ Data loading complete!"
