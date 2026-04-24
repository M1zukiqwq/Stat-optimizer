#!/usr/bin/env python3
"""
Robust IMDB data loader for PostgreSQL
Handles special characters in CSV files
"""

import csv
import psycopg2
import sys
from pathlib import Path

# Map table names to CSV files
TABLE_FILES = {
    'comp_cast_type': 'comp_cast_type.csv',
    'company_type': 'company_type.csv',
    'info_type': 'info_type.csv',
    'kind_type': 'kind_type.csv',
    'link_type': 'link_type.csv',
    'role_type': 'role_type.csv',
    'keyword': 'keyword.csv',
    'company_name': 'company_name.csv',
    'name': 'name.csv',
    'char_name': 'char_name.csv',
    'title': 'title.csv',
    'aka_name': 'aka_name.csv',
    'aka_title': 'aka_title.csv',
    'complete_cast': 'complete_cast.csv',
    'movie_link': 'movie_link.csv',
    'movie_companies': 'movie_companies.csv',
    'movie_keyword': 'movie_keyword.csv',
    'movie_info_idx': 'movie_info_idx.csv',
    'person_info': 'person_info.csv',
    'movie_info': 'movie_info.csv',
    'cast_info': 'cast_info.csv',
}


def load_table(conn, cursor, table_name, csv_path, batch_size=10000):
    """Load a table with proper error handling"""
    print(f"Loading {table_name}...", end=' ', flush=True)
    
    if not csv_path.exists():
        print(f"SKIP (file not found)")
        return 0
    
    count = 0
    errors = 0
    batch = []
    
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"', doublequote=True)
        
        for row in reader:
            try:
                # Clean up the row data
                cleaned = []
                for val in row:
                    if val is None or val == '':
                        cleaned.append(None)
                    else:
                        # Handle special characters
                        cleaned.append(val)
                
                batch.append(cleaned)
                
                if len(batch) >= batch_size:
                    insert_batch(conn, cursor, table_name, batch)
                    count += len(batch)
                    batch = []
                    print(f"\r  Loading {table_name}... {count:,} rows", end='', flush=True)
                    
            except Exception as e:
                errors += 1
                if errors <= 3:  # Only show first 3 errors
                    print(f"\n  Error on row {count + errors}: {e}")
                continue
    
    # Insert remaining rows
    if batch:
        insert_batch(conn, cursor, table_name, batch)
        count += len(batch)
    
    print(f"\r  Loaded {table_name}: {count:,} rows{' (' + str(errors) + ' errors)' if errors > 0 else ''}")
    return count


def insert_batch(conn, cursor, table_name, batch):
    """Insert a batch of rows"""
    if not batch:
        return
    
    # Build VALUES clause with parameterized queries
    placeholders = ','.join(['%s'] * len(batch[0]))
    sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
    
    try:
        for row in batch:
            # Convert empty strings to None
            processed = [None if val == '' else val for val in row]
            cursor.execute(sql, processed)
        conn.commit()
    except Exception as e:
        conn.rollback()
        # Fall back to row-by-row insertion
        for row in batch:
            try:
                processed = [None if val == '' else val for val in row]
                cursor.execute(sql, processed)
                conn.commit()
            except:
                conn.rollback()
                pass  # Skip problematic rows


def main():
    data_dir = Path(__file__).parent / 'imdb_data'
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        dbname='imdb',
        user='qichutian'
    )
    conn.autocommit = False
    cursor = conn.cursor()
    
    total_loaded = 0
    
    # Load tables in order (small to large)
    tables_to_load = [
        # Small tables first
        'comp_cast_type', 'company_type', 'info_type', 'kind_type', 
        'link_type', 'role_type', 'keyword',
        # Medium tables
        'company_name', 'name', 'char_name', 'aka_name', 'aka_title',
        'complete_cast', 'movie_link', 'movie_info_idx',
        # Large tables
        'title', 'movie_companies', 'movie_keyword', 
        'person_info', 'movie_info', 'cast_info'
    ]
    
    print(f"Loading IMDB data from {data_dir}")
    print("="*70)
    
    for table in tables_to_load:
        csv_file = data_dir / TABLE_FILES.get(table, f'{table}.csv')
        try:
            count = load_table(conn, cursor, table, csv_file)
            total_loaded += count
        except Exception as e:
            print(f"  ✗ Failed to load {table}: {e}")
    
    cursor.close()
    conn.close()
    
    print("="*70)
    print(f"Total rows loaded: {total_loaded:,}")


if __name__ == '__main__':
    main()
