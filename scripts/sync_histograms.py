#!/usr/bin/env python3
"""
Sync KLL histograms from Iceberg Puffin files to ML feedback JSON files.

This script reads KLL sketch statistics from Iceberg tables and writes them
to the JSON files created by MLFeedbackEventListener, adding the 'prior_kll'
field required by the OASIS ML pipeline.

Usage:
    python3 sync_histograms.py --catalog-uri thrift://localhost:9083 \\
                                --warehouse /path/to/warehouse \\
                                --feedback-dir /tmp/ml-feedback \\
                                --tables table1,table2
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from pyiceberg.catalog import load_catalog
    from pyiceberg.table import Table
    from pyiceberg.io import FileIO
except ImportError:
    print("Error: pyiceberg is not installed. Install it with:")
    print("  pip install pyiceberg")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_kll_quantiles(kll_bytes: bytes, num_quantiles: int = 9) -> Dict:
    """
    Extract quantile levels and values from KLL sketch bytes.

    Note: This is a placeholder. Full implementation requires:
    1. Apache DataSketches Python bindings (datasketches-python)
    2. Or calling Java code via py4j
    3. Or parsing the binary format directly

    For now, returns a dummy structure.
    """
    # TODO: Implement actual KLL sketch parsing
    # This would require datasketches-python or similar
    logger.warning("KLL sketch parsing not yet implemented, returning placeholder")

    return {
        "min": 0.0,
        "max": 1.0,
        "null_fraction": 0.0,
        "quantile_levels": [i / (num_quantiles + 1) for i in range(1, num_quantiles + 1)],
        "quantile_values": [i / (num_quantiles + 1) for i in range(1, num_quantiles + 1)]
    }


def get_latest_statistics_file(table: Table) -> Optional[any]:
    """Get the most recent statistics file for a table."""
    stats_files = list(table.statistics_files())
    if not stats_files:
        return None

    # Sort by snapshot ID (descending) to get the latest
    return max(stats_files, key=lambda f: f.snapshot_id)


def read_kll_histograms_from_puffin(table: Table, stats_file: any) -> Dict[str, Dict]:
    """
    Read KLL histograms from a Puffin statistics file.

    Returns:
        Dict mapping column names to histogram data
    """
    histograms = {}

    try:
        # Get the file IO from the table
        file_io = table.io

        # Read the Puffin file
        input_file = file_io.new_input(stats_file.path)

        # TODO: Parse Puffin file format
        # This requires implementing Puffin reader in Python or using Java interop
        logger.warning(f"Puffin file reading not yet implemented: {stats_file.path}")

        # Placeholder: Extract column names from table schema
        for field in table.schema().fields:
            if field.field_type.is_primitive:
                histograms[field.name] = extract_kll_quantiles(b"")

    except Exception as e:
        logger.error(f"Failed to read Puffin file {stats_file.path}: {e}")

    return histograms


def update_feedback_json(feedback_dir: Path, table_name: str,
                         column_name: str, histogram: Dict) -> bool:
    """
    Update a feedback JSON file with histogram data.

    Args:
        feedback_dir: Base directory for feedback files
        table_name: Table name
        column_name: Column name
        histogram: Histogram data dict with min, max, quantile_levels, quantile_values

    Returns:
        True if successful, False otherwise
    """
    # Sanitize names (match Java sanitize() logic)
    safe_table = table_name.replace('/', '_').replace('.', '_')
    safe_column = column_name.replace('/', '_').replace('.', '_')

    table_dir = feedback_dir / safe_table
    table_dir.mkdir(parents=True, exist_ok=True)

    json_file = table_dir / f"{safe_column}.json"

    try:
        # Read existing content
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
        else:
            data = {"observations": []}

        # Add or update prior_kll
        data["prior_kll"] = histogram

        # Write back
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Updated histogram for {table_name}.{column_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to update {json_file}: {e}")
        return False


def sync_table_histograms(catalog: any, table_name: str,
                          feedback_dir: Path) -> int:
    """
    Sync histograms for a single table.

    Returns:
        Number of columns updated
    """
    try:
        # Load the table
        table = catalog.load_table(table_name)
        logger.info(f"Processing table: {table_name}")

        # Get the latest statistics file
        stats_file = get_latest_statistics_file(table)
        if not stats_file:
            logger.warning(f"No statistics file found for {table_name}")
            return 0

        logger.info(f"Using statistics file: {stats_file.path}")

        # Read KLL histograms
        histograms = read_kll_histograms_from_puffin(table, stats_file)

        # Update JSON files
        updated_count = 0
        for column_name, histogram in histograms.items():
            if update_feedback_json(feedback_dir, table_name, column_name, histogram):
                updated_count += 1

        return updated_count

    except Exception as e:
        logger.error(f"Failed to process table {table_name}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description='Sync KLL histograms from Iceberg to ML feedback JSON files'
    )
    parser.add_argument(
        '--catalog-uri',
        required=True,
        help='Iceberg catalog URI (e.g., thrift://localhost:9083)'
    )
    parser.add_argument(
        '--warehouse',
        required=True,
        help='Warehouse path (e.g., /path/to/warehouse)'
    )
    parser.add_argument(
        '--feedback-dir',
        required=True,
        help='ML feedback output directory (e.g., /tmp/ml-feedback)'
    )
    parser.add_argument(
        '--tables',
        help='Comma-separated list of tables to sync (default: all tables)'
    )
    parser.add_argument(
        '--catalog-type',
        default='hive',
        choices=['hive', 'hadoop', 'rest'],
        help='Catalog type (default: hive)'
    )

    args = parser.parse_args()

    feedback_dir = Path(args.feedback_dir)
    if not feedback_dir.exists():
        logger.error(f"Feedback directory does not exist: {feedback_dir}")
        sys.exit(1)

    # Load Iceberg catalog
    try:
        if args.catalog_type == 'hive':
            catalog = load_catalog(
                'default',
                **{
                    'type': 'hive',
                    'uri': args.catalog_uri,
                    'warehouse': args.warehouse
                }
            )
        elif args.catalog_type == 'hadoop':
            catalog = load_catalog(
                'default',
                **{
                    'type': 'hadoop',
                    'warehouse': args.warehouse
                }
            )
        else:
            logger.error(f"Unsupported catalog type: {args.catalog_type}")
            sys.exit(1)

        logger.info(f"Connected to Iceberg catalog: {args.catalog_uri}")

    except Exception as e:
        logger.error(f"Failed to connect to catalog: {e}")
        sys.exit(1)

    # Get list of tables to sync
    if args.tables:
        tables = [t.strip() for t in args.tables.split(',')]
    else:
        # List all tables in the catalog
        try:
            tables = catalog.list_tables()
            logger.info(f"Found {len(tables)} tables in catalog")
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            sys.exit(1)

    # Sync each table
    total_updated = 0
    for table_name in tables:
        updated = sync_table_histograms(catalog, table_name, feedback_dir)
        total_updated += updated

    logger.info(f"Sync complete. Updated {total_updated} columns across {len(tables)} tables.")


if __name__ == '__main__':
    main()
