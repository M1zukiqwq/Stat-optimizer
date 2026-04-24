#!/usr/bin/env python3
"""
Fix CSV files with nested quotes and special characters for PostgreSQL COPY
"""

import csv
import sys
from pathlib import Path

# Increase CSV field size limit for large text fields
csv.field_size_limit(sys.maxsize)


def fix_csv_file(input_file, output_file):
    """
    Read CSV file and write a clean version suitable for PostgreSQL COPY
    """
    print(f"Fixing {input_file.name}...")
    
    fixed_rows = 0
    total_rows = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='replace') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        # Use csv.reader with flexible parsing
        reader = csv.reader(infile, delimiter=',', quotechar='"', doublequote=True, 
                           skipinitialspace=False, strict=False)
        writer = csv.writer(outfile, delimiter=',', quotechar='"', 
                           quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        
        for row_num, row in enumerate(reader, 1):
            total_rows += 1
            
            try:
                # Clean each field
                cleaned = []
                for field in row:
                    if field is None:
                        cleaned.append('')
                    else:
                        # Remove problematic characters and normalize
                        # Replace embedded newlines and carriage returns
                        field = field.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
                        # Remove null bytes
                        field = field.replace('\x00', '')
                        # Strip leading/trailing whitespace
                        field = field.strip()
                        cleaned.append(field)
                
                writer.writerow(cleaned)
                fixed_rows += 1
                
                if row_num % 100000 == 0:
                    print(f"  Processed {row_num:,} rows...")
                    
            except Exception as e:
                print(f"  Warning: Row {row_num} failed: {e}")
                # Write empty row to maintain count
                continue
    
    print(f"  Done: {fixed_rows:,} / {total_rows:,} rows written")
    return fixed_rows


def main():
    data_dir = Path(__file__).parent / 'imdb_data'
    fixed_dir = data_dir / 'fixed'
    fixed_dir.mkdir(exist_ok=True)
    
    # Files known to have issues
    files_to_fix = [
        'company_name.csv',
        'name.csv', 
        'char_name.csv',
        'aka_name.csv',
        'aka_title.csv',
        'title.csv',
        'movie_info.csv',
        'person_info.csv',
        'cast_info.csv',
        'movie_companies.csv',
        'movie_keyword.csv',
    ]
    
    print("Fixing CSV files for PostgreSQL compatibility")
    print("="*70)
    
    total_rows = 0
    for filename in files_to_fix:
        input_file = data_dir / filename
        if input_file.exists():
            output_file = fixed_dir / filename
            rows = fix_csv_file(input_file, output_file)
            total_rows += rows
        else:
            print(f"Skipping {filename} (not found)")
    
    print("="*70)
    print(f"Total rows processed: {total_rows:,}")
    print(f"Fixed files are in: {fixed_dir}")


if __name__ == '__main__':
    main()
