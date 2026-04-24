#!/usr/bin/env python3
"""
Load IMDB data using Python with robust CSV parsing
Bypasses COPY command limitations
"""

import csv
import psycopg2
import sys
from pathlib import Path

# Increase field size limit
csv.field_size_limit(sys.maxsize)

# Table configuration: (name, columns, batch_size)
TABLES = [
    ('comp_cast_type', 2, 1000),
    ('company_type', 2, 1000),
    ('info_type', 2, 1000),
    ('kind_type', 2, 1000),
    ('link_type', 2, 1000),
    ('role_type', 2, 1000),
    ('keyword', 3, 10000),
    ('company_name', 7, 10000),
    ('movie_link', 4, 10000),
    ('complete_cast', 3, 10000),
    ('name', 9, 10000),
    ('char_name', 6, 10000),
    ('aka_name', 8, 10000),
    ('title', 12, 10000),
    ('aka_title', 12, 10000),
    ('movie_companies', 5, 10000),
    ('movie_keyword', 3, 10000),
    ('movie_info_idx', 5, 10000),
    ('person_info', 5, 5000),
    ('movie_info', 5, 5000),
    ('cast_info', 7, 5000),
]


def load_table(conn, cursor, table_name, expected_cols, batch_size, data_dir):
    """Load a single table"""
    csv_file = data_dir / f'{table_name}.csv'
    
    if not csv_file.exists():
        print(f"  SKIP: {csv_file} not found")
        return 0
    
    print(f"Loading {table_name}...")
    
    # Truncate table first
    cursor.execute(f"TRUNCATE TABLE {table_name}")
    conn.commit()
    
    total = 0
    errors = 0
    batch = []
    
    with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"', doublequote=True)
        
        for row_num, row in enumerate(reader, 1):
            try:
                # Handle column count mismatch
                if len(row) != expected_cols:
                    if len(row) > expected_cols:
                        # Merge extra columns into the last one
                        row = row[:expected_cols-1] + [','.join(row[expected_cols-1:])]
                    else:
                        # Pad with empty values
                        row = row + [''] * (expected_cols - len(row))
                
                # Clean data
                cleaned = []
                for val in row:
                    if val is None:
                        cleaned.append(None)
                    else:
                        # Clean the value
                        val = val.replace('\x00', '')  # Remove null bytes
                        val = ' '.join(val.split())     # Normalize whitespace
                        cleaned.append(val if val else None)
                
                batch.append(cleaned)
                
                if len(batch) >= batch_size:
                    insert_batch(conn, cursor, table_name, batch)
                    total += len(batch)
                    batch = []
                    if total % 100000 == 0:
                        print(f"  {total:,} rows...")
                        
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Error row {row_num}: {e}")
                continue
    
    # Insert remaining
    if batch:
        insert_batch(conn, cursor, table_name, batch)
        total += len(batch)
    
    print(f"  ✓ Loaded {total:,} rows ({errors} errors)")
    return total


def insert_batch(conn, cursor, table_name, batch):
    """Insert a batch of rows"""
    if not batch:
        return
    
    placeholders = ','.join(['%s'] * len(batch[0]))
    sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
    
    try:
        for row in batch:
            cursor.execute(sql, row)
        conn.commit()
    except Exception as e:
        conn.rollback()
        # Try row by row
        for row in batch:
            try:
                cursor.execute(sql, row)
                conn.commit()
            except:
                conn.rollback()


def main():
    data_dir = Path(__file__).parent / 'imdb_data'
    
    conn = psycopg2.connect(
        host='localhost',
        port=5433,
        dbname='imdb',
        user='postgres'
    )
    conn.autocommit = False
    cursor = conn.cursor()
    
    total_rows = 0
    
    print("Loading IMDB data with Python")
    print("="*70)
    
    for table, cols, batch in TABLES:
        try:
            rows = load_table(conn, cursor, table, cols, batch, data_dir)
            total_rows += rows
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print("="*70)
    print(f"Total: {total_rows:,} rows loaded")
    
    # Run ANALYZE
    print("\nRunning ANALYZE...")
    for table, _, _ in TABLES:
        try:
            cursor.execute(f"ANALYZE {table}")
            conn.commit()
        except:
            pass
    
    # Show counts
    print("\nTable counts:")
    cursor.execute("""
        SELECT relname, n_live_tup 
        FROM pg_stat_user_tables 
        WHERE schemaname = 'public'
        ORDER BY n_live_tup DESC
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,}")
    
    cursor.close()
    conn.close()
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
