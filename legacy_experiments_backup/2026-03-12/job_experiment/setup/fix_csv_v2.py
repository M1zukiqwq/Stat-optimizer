#!/usr/bin/env python3
"""
Advanced CSV fixer for IMDB data files
Handles complex nested quotes and multi-line fields
"""

import re
import sys
from pathlib import Path


def parse_csv_line(line):
    """
    Parse a CSV line respecting quoted fields
    Returns list of fields or None if line is incomplete
    """
    fields = []
    current = []
    in_quotes = False
    i = 0
    
    while i < len(line):
        char = line[i]
        
        if char == '"':
            if in_quotes and i + 1 < len(line) and line[i + 1] == '"':
                # Escaped quote
                current.append('"')
                i += 2
                continue
            elif in_quotes:
                in_quotes = False
            else:
                in_quotes = True
        elif char == ',' and not in_quotes:
            fields.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
        
        i += 1
    
    if in_quotes:
        return None  # Line is incomplete (multi-line field)
    
    fields.append(''.join(current).strip())
    return fields


def fix_csv_file(input_file, output_file, expected_columns):
    """
    Read CSV file and write a clean version
    """
    print(f"Fixing {input_file.name} (expecting {expected_columns} columns)...")
    
    total_rows = 0
    fixed_rows = 0
    skipped_rows = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='replace') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        buffer = ""
        line_num = 0
        
        for line in infile:
            line_num += 1
            
            if buffer:
                buffer += " "  # Replace newline with space for multi-line fields
            buffer += line.rstrip('\r\n')
            
            # Try to parse the buffer
            fields = parse_csv_line(buffer)
            
            if fields is None:
                # Incomplete line (multi-line field), continue buffering
                continue
            
            # We have a complete row
            total_rows += 1
            
            # Validate column count
            if len(fields) != expected_columns:
                # Try to fix by merging/splitting
                if len(fields) > expected_columns:
                    # Too many columns - merge extras into the last expected column
                    fixed = fields[:expected_columns-1] + [','.join(fields[expected_columns-1:])]
                    fields = fixed
                else:
                    # Too few columns - skip this row
                    print(f"  Warning: Row {line_num} has {len(fields)} columns, expected {expected_columns}")
                    skipped_rows += 1
                    buffer = ""
                    continue
            
            # Clean each field
            cleaned = []
            for field in fields:
                # Remove null bytes
                field = field.replace('\x00', '')
                # Normalize whitespace
                field = ' '.join(field.split())
                cleaned.append(field)
            
            # Write the fixed row
            outfile.write(','.join(cleaned) + '\n')
            fixed_rows += 1
            buffer = ""
            
            if total_rows % 100000 == 0:
                print(f"  Processed {total_rows:,} rows...")
        
        # Handle any remaining buffered content
        if buffer:
            fields = parse_csv_line(buffer)
            if fields and len(fields) == expected_columns:
                cleaned = [' '.join(f.split()).replace('\x00', '') for f in fields]
                outfile.write(','.join(cleaned) + '\n')
                fixed_rows += 1
    
    print(f"  Done: {fixed_rows:,} rows written, {skipped_rows} skipped")
    return fixed_rows


def main():
    data_dir = Path(__file__).parent / 'imdb_data'
    fixed_dir = data_dir / 'fixed_v2'
    fixed_dir.mkdir(exist_ok=True)
    
    # Table to column count mapping
    tables = [
        ('comp_cast_type', 2),
        ('company_type', 2),
        ('info_type', 2),
        ('kind_type', 2),
        ('link_type', 2),
        ('role_type', 2),
        ('keyword', 3),
        ('company_name', 7),
        ('movie_link', 4),
        ('complete_cast', 3),
        ('name', 9),
        ('char_name', 6),
        ('aka_name', 8),
        ('title', 12),
        ('aka_title', 12),
        ('movie_companies', 5),
        ('movie_keyword', 3),
        ('movie_info_idx', 5),
        ('person_info', 5),
        ('movie_info', 5),
        ('cast_info', 7),
    ]
    
    print("Advanced CSV Fixer v2")
    print("="*70)
    
    total_rows = 0
    for table, columns in tables:
        input_file = data_dir / f'{table}.csv'
        output_file = fixed_dir / f'{table}.csv'
        
        if input_file.exists():
            try:
                rows = fix_csv_file(input_file, output_file, columns)
                total_rows += rows
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        else:
            print(f"Skipping {table}.csv (not found)")
    
    print("="*70)
    print(f"Total rows processed: {total_rows:,}")
    print(f"Fixed files are in: {fixed_dir}")


if __name__ == '__main__':
    main()
