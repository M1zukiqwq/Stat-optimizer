#!/usr/bin/env python3
"""
加载 IMDB CSV 数据到 Iceberg 表
"""

import argparse
import csv
import sys
from pathlib import Path

import prestodb


# 表名到 CSV 文件的映射
TABLE_FILES = {
    'title': 'title.csv',
    'cast_info': 'cast_info.csv',
    'movie_info': 'movie_info.csv',
    'movie_companies': 'movie_companies.csv',
    'movie_keyword': 'movie_keyword.csv',
    'person_info': 'person_info.csv',
    'movie_info_idx': 'movie_info_idx.csv',
    'aka_title': 'aka_title.csv',
    'aka_name': 'aka_name.csv',
    'complete_cast': 'complete_cast.csv',
    'comp_cast_type': 'comp_cast_type.csv',
    'company_name': 'company_name.csv',
    'company_type': 'company_type.csv',
    'info_type': 'info_type.csv',
    'keyword': 'keyword.csv',
    'kind_type': 'kind_type.csv',
    'link_type': 'link_type.csv',
    'name': 'name.csv',
    'role_type': 'role_type.csv',
    'char_name': 'char_name.csv',
    'movie_link': 'movie_link.csv',
}


def load_table(cursor, table_name: str, csv_path: Path, batch_size: int = 10000):
    """加载单个表的数据"""
    print(f"Loading {table_name}...")

    if not csv_path.exists():
        print(f"  ✗ CSV file not found: {csv_path}")
        return

    # 读取 CSV 并批量插入
    with open(csv_path, 'r', encoding='latin1') as f:
        reader = csv.reader(f)

        # 跳过表头
        next(reader, None)

        batch = []
        total_rows = 0

        for row in reader:
            # 转义单引号
            escaped_row = [val.replace("'", "''") if val else 'NULL' for val in row]
            batch.append(escaped_row)

            if len(batch) >= batch_size:
                insert_batch(cursor, table_name, batch)
                total_rows += len(batch)
                print(f"  Inserted {total_rows:,} rows...", end='\r')
                batch = []

        # 插入剩余的行
        if batch:
            insert_batch(cursor, table_name, batch)
            total_rows += len(batch)

        print(f"  ✓ Loaded {total_rows:,} rows")


def try_convert_numeric(val):
    """尝试转换为数字，如果失败则返回原始值"""
    if val == 'NULL' or val is None or val == '':
        return 'NULL'
    try:
        # 尝试转为 int
        int(val)
        return val  # 保持字符串形式，但不加引号
    except ValueError:
        try:
            # 尝试转为 float
            float(val)
            return val  # 保持字符串形式，但不加引号
        except ValueError:
            # 是字符串，需要加引号
            return None  # 标记为字符串

def insert_batch(cursor, table_name: str, batch: list):
    """批量插入数据"""
    if not batch:
        return

    # 构建 VALUES 子句
    values_list = []
    for row in batch:
        values = []
        for val in row:
            if val == 'NULL' or val is None or val == '':
                values.append('NULL')
            else:
                # 检查是否为数字
                is_numeric = False
                try:
                    int(val)
                    is_numeric = True
                except ValueError:
                    try:
                        float(val)
                        is_numeric = True
                    except ValueError:
                        pass
                
                if is_numeric:
                    values.append(val)  # 数字不加引号
                else:
                    values.append(f"'{val}'")  # 字符串加引号
        
        values_list.append(f"({', '.join(values)})")

    values_clause = ',\n'.join(values_list)

    sql = f"INSERT INTO {table_name} VALUES\n{values_clause}"

    try:
        cursor.execute(sql)
    except Exception as e:
        print(f"\n  ✗ Error inserting batch: {e}")
        # 尝试逐行插入（慢但更健壮）
        for row in batch:
            values = []
            for val in row:
                if val == 'NULL' or val is None or val == '':
                    values.append('NULL')
                else:
                    try:
                        int(val)
                        values.append(val)
                    except ValueError:
                        try:
                            float(val)
                            values.append(val)
                        except ValueError:
                            values.append(f"'{val}'")
            try:
                cursor.execute(f"INSERT INTO {table_name} VALUES ({', '.join(values)})")
            except Exception as e2:
                pass  # 静默跳过单行错误


def main():
    parser = argparse.ArgumentParser(description='Load IMDB data into Iceberg')
    parser.add_argument('--data-dir', required=True, help='Directory containing IMDB CSV files')
    parser.add_argument('--presto-host', default='localhost:8080', help='Presto host:port')
    parser.add_argument('--catalog', default='iceberg', help='Catalog name')
    parser.add_argument('--schema', default='imdb', help='Schema name')
    parser.add_argument('--tables', nargs='+', help='Specific tables to load (default: all)')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size for inserts')
    parser.add_argument('--user', default='tianqc', help='Presto user')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # 连接 Presto
    host, port = args.presto_host.split(':')
    conn = prestodb.dbapi.connect(
        host=host,
        port=int(port),
        user=args.user,
        catalog=args.catalog,
        schema=args.schema,
        http_scheme='http'
    )
    cursor = conn.cursor()

    # 确定要加载的表
    tables_to_load = args.tables if args.tables else TABLE_FILES.keys()

    print(f"Loading {len(tables_to_load)} tables into {args.catalog}.{args.schema}")
    print("="*70)

    for table_name in tables_to_load:
        if table_name not in TABLE_FILES:
            print(f"Warning: Unknown table {table_name}, skipping")
            continue

        csv_file = TABLE_FILES[table_name]
        csv_path = data_dir / csv_file

        try:
            load_table(cursor, f"{args.catalog}.{args.schema}.{table_name}",
                      csv_path, args.batch_size)
            conn.commit()
        except Exception as e:
            print(f"  ✗ Error loading {table_name}: {e}")
            conn.rollback()

    cursor.close()
    conn.close()

    print("="*70)
    print("✓ Data loading complete")


if __name__ == '__main__':
    main()
