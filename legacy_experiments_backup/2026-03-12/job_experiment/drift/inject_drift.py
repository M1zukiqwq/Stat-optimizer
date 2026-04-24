#!/usr/bin/env python3
"""
JOB Benchmark Drift Injection Script

模拟与消融实验 q=15 相当的漂移强度，按表规模比例注入 DML 操作。

消融实验参数（10K rows 初始表）：
- q=15: 每个 observation 前执行 15 轮 compound drift
- 每轮: INSERT 10-100, DELETE 10-100, UPDATE 10-100, NULL_CHANGE ±50
- 平均每轮影响: ~200 rows (2% of 10K)
- 15 轮累计: ~3000 rows (30% of 10K)

JOB 等效策略：
- 每轮影响表的 2%
- 执行 15 轮
- 累计影响 30% 的行数
"""

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import prestodb


class DriftInjector:
    def __init__(self, host: str, port: int, catalog: str, schema: str, user: str = 'tianqc'):
        self.conn = prestodb.dbapi.connect(
            host=host,
            port=port,
            user=user,
            catalog=catalog,
            schema=schema,
            http_scheme='http'
        )
        self.cursor = self.conn.cursor()
        self.catalog = catalog
        self.schema = schema

    def get_table_info(self, table_name: str) -> Dict:
        """获取表的行数和数值列信息"""
        # 获取行数
        self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = self.cursor.fetchone()[0]

        # 获取数值列（用于 UPDATE 操作）
        self.cursor.execute(f"""
            SELECT column_name, data_type
            FROM {self.catalog}.information_schema.columns
            WHERE table_schema = '{self.schema}'
              AND table_name = '{table_name}'
              AND data_type IN ('integer', 'bigint', 'double', 'real', 'decimal')
        """)
        numeric_columns = [(row[0], row[1]) for row in self.cursor.fetchall()]

        return {
            'row_count': row_count,
            'numeric_columns': numeric_columns
        }

    def inject_drift_round(self, table_name: str, round_num: int, drift_ratio: float = 0.02,
                          enable_delete: bool = True, enable_update: bool = True):
        """
        执行一轮漂移操作

        Args:
            table_name: 表名
            round_num: 轮次编号
            drift_ratio: 漂移比例（默认 2%，对应消融实验的平均强度）
            enable_delete: 是否启用 DELETE 操作（Iceberg v1 不支持）
            enable_update: 是否启用 UPDATE 操作（Iceberg v1 不支持）
        """
        info = self.get_table_info(table_name)
        row_count = info['row_count']

        # 计算批次大小（至少 100 行，避免小表操作过小）
        batch_size = max(100, int(row_count * drift_ratio))

        print(f"  Round {round_num}: {table_name} ({row_count:,} rows) -> batch_size={batch_size:,}")

        # 分配操作权重（与消融实验一致）
        if enable_delete and enable_update:
            # 完整模式：INSERT 40%, DELETE 30%, UPDATE 30%
            insert_count = int(batch_size * 0.4)
            delete_count = int(batch_size * 0.3)
            update_count = int(batch_size * 0.3)
        elif not enable_delete and not enable_update:
            # 仅 INSERT 模式（Iceberg v1 兼容）
            insert_count = batch_size
            delete_count = 0
            update_count = 0
        else:
            # 部分启用
            if enable_delete:
                insert_count = int(batch_size * 0.6)
                delete_count = int(batch_size * 0.4)
                update_count = 0
            else:
                insert_count = int(batch_size * 0.6)
                delete_count = 0
                update_count = int(batch_size * 0.4)

        try:
            # INSERT: 复制现有行并添加随机扰动
            self._insert_rows(table_name, insert_count, info)

            # DELETE: 随机删除（可选）
            if enable_delete and delete_count > 0:
                self._delete_rows(table_name, delete_count)

            # UPDATE: 随机更新数值列（可选）
            if enable_update and update_count > 0:
                self._update_rows(table_name, update_count, info)

            # Presto 不需要显式 commit，自动提交
            ops = []
            if insert_count > 0:
                ops.append(f"INSERT {insert_count:,}")
            if delete_count > 0:
                ops.append(f"DELETE {delete_count:,}")
            if update_count > 0:
                ops.append(f"UPDATE {update_count:,}")
            print(f"    ✓ {', '.join(ops)}")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            # Presto 不支持 rollback，每个语句自动提交
            raise

    def _insert_rows(self, table_name: str, count: int, info: Dict):
        """
        INSERT: 插入分布反转的数据

        核心原理：让原本稀少的值变得大量出现。
        直方图记录的是漂移前的分布，漂移后实际分布与直方图严重不符：
        - 直方图说 "production_year=1900 只有 50 行" → 实际有 500,000 行
        - CBO 估算 WHERE production_year=1900 返回 50 行 → 实际 500,000 行
        - 导致选择 Nested Loop Join 而非 Hash Join → 性能灾难

        策略：从原始数据中最稀少的值（分布尾部）采样并大量复制，
        而非从热门值采样（热门值直方图本来就知道它们频繁）。
        """
        if count == 0:
            return

        if table_name == 'title':
            # title 表：大量插入 production_year 极早期的电影（原本非常稀少）
            # 原始数据中 1900-1950 的电影很少，直方图对这个范围的桶很粗
            # 漂移后这个范围突然有大量数据，直方图完全失准
            self.cursor.execute(f"""
                INSERT INTO {table_name}
                SELECT
                    CAST((SELECT COALESCE(MAX(id), 0) FROM {table_name}) + row_number() OVER () AS INTEGER) as id,
                    title || ' (drift)',
                    imdb_index,
                    kind_id,
                    1920 + CAST(rand() * 30 AS INTEGER) as production_year,
                    imdb_id,
                    phonetic_code,
                    episode_of_id,
                    season_nr,
                    episode_nr,
                    series_years,
                    md5sum
                FROM {table_name}
                ORDER BY rand()
                LIMIT {count}
            """)

        elif table_name == 'cast_info':
            # cast_info 表：集中在原本 cast 很少的 movie_id 上
            # 直方图认为这些 movie_id 只有 1-2 条 cast 记录
            # 漂移后每个 movie_id 有几千条 → join 基数严重低估
            self.cursor.execute(f"""
                INSERT INTO {table_name}
                SELECT
                    CAST((SELECT COALESCE(MAX(id), 0) FROM {table_name}) + row_number() OVER () AS INTEGER) as id,
                    person_id,
                    movie_id,
                    person_role_id,
                    note,
                    nr_order,
                    role_id
                FROM {table_name}
                WHERE movie_id IN (
                    SELECT movie_id FROM {table_name}
                    GROUP BY movie_id
                    HAVING COUNT(*) <= 3
                    ORDER BY rand()
                    LIMIT 100
                )
                ORDER BY rand()
                LIMIT {count}
            """)

        elif table_name == 'movie_info':
            # movie_info 表：集中在最稀少的 info_type_id 上
            # 直方图认为这些 info_type_id 只有几百行，漂移后变成几十万行
            self.cursor.execute(f"""
                INSERT INTO {table_name}
                SELECT
                    CAST((SELECT COALESCE(MAX(id), 0) FROM {table_name}) + row_number() OVER () AS INTEGER) as id,
                    movie_id,
                    info_type_id,
                    info || ' (drift)',
                    note
                FROM {table_name}
                WHERE info_type_id IN (
                    SELECT info_type_id FROM {table_name}
                    GROUP BY info_type_id
                    ORDER BY COUNT(*) ASC
                    LIMIT 3
                )
                ORDER BY rand()
                LIMIT {count}
            """)

        elif table_name == 'movie_keyword':
            # movie_keyword 表：集中在最稀少的 keyword_id 上
            self.cursor.execute(f"""
                INSERT INTO {table_name}
                SELECT
                    CAST((SELECT COALESCE(MAX(id), 0) FROM {table_name}) + row_number() OVER () AS INTEGER) as id,
                    movie_id,
                    keyword_id
                FROM {table_name}
                WHERE keyword_id IN (
                    SELECT keyword_id FROM {table_name}
                    GROUP BY keyword_id
                    ORDER BY COUNT(*) ASC
                    LIMIT 20
                )
                ORDER BY rand()
                LIMIT {count}
            """)

        elif table_name == 'movie_companies':
            # movie_companies 表：集中在最稀少的 company_type_id 上
            self.cursor.execute(f"""
                INSERT INTO {table_name}
                SELECT
                    CAST((SELECT COALESCE(MAX(id), 0) FROM {table_name}) + row_number() OVER () AS INTEGER) as id,
                    movie_id,
                    company_id,
                    company_type_id,
                    note
                FROM {table_name}
                WHERE company_type_id IN (
                    SELECT company_type_id FROM {table_name}
                    GROUP BY company_type_id
                    ORDER BY COUNT(*) ASC
                    LIMIT 1
                )
                ORDER BY rand()
                LIMIT {count}
            """)

        elif table_name == 'name':
            # name 表：从 ID 最小的 5% 采样（直方图对尾部分辨率低）
            self.cursor.execute(f"""
                INSERT INTO {table_name}
                SELECT
                    CAST((SELECT COALESCE(MAX(id), 0) FROM {table_name}) + row_number() OVER () AS INTEGER) as id,
                    name || ' (drift)',
                    imdb_index,
                    imdb_id,
                    gender,
                    name_pcode_cf,
                    name_pcode_nf,
                    surname_pcode,
                    md5sum
                FROM {table_name}
                WHERE id < (SELECT CAST(MAX(id) * 0.05 AS INTEGER) FROM {table_name})
                ORDER BY rand()
                LIMIT {count}
            """)

        elif table_name == 'movie_info_idx':
            # movie_info_idx 表：集中在最稀少的 info_type_id
            self.cursor.execute(f"""
                INSERT INTO {table_name}
                SELECT
                    CAST((SELECT COALESCE(MAX(id), 0) FROM {table_name}) + row_number() OVER () AS INTEGER) as id,
                    movie_id,
                    info_type_id,
                    info,
                    note
                FROM {table_name}
                WHERE info_type_id IN (
                    SELECT info_type_id FROM {table_name}
                    GROUP BY info_type_id
                    ORDER BY COUNT(*) ASC
                    LIMIT 2
                )
                ORDER BY rand()
                LIMIT {count}
            """)

        else:
            # 通用策略：从 ID 最小的 5% 采样（分布头部偏移）
            # 直方图对极端范围的分辨率最低，偏移效果最大
            self.cursor.execute(f"""
                SELECT column_name
                FROM {self.catalog}.information_schema.columns
                WHERE table_schema = '{self.schema}'
                  AND table_name = '{table_name}'
                ORDER BY ordinal_position
            """)
            columns = [row[0] for row in self.cursor.fetchall()]

            other_cols = ', '.join(columns[1:])
            self.cursor.execute(f"""
                INSERT INTO {table_name}
                SELECT
                    CAST((SELECT COALESCE(MAX({columns[0]}), 0) FROM {table_name}) + row_number() OVER () AS INTEGER) as {columns[0]},
                    {other_cols}
                FROM {table_name}
                WHERE {columns[0]} < (SELECT CAST(MAX({columns[0]}) * 0.05 AS INTEGER) FROM {table_name})
                ORDER BY rand()
                LIMIT {count}
            """)

    def _delete_rows(self, table_name: str, count: int):
        """
        DELETE: 定向删除原本频繁的值（让多的变少）

        与 INSERT（让少的变多）配合，形成双向分布反转：
        - INSERT: 稀少值 → 大量出现（CBO 低估）
        - DELETE: 频繁值 → 大量减少（CBO 高估）

        CBO 同时在两个方向上犯错，效果叠加。
        """
        if count == 0:
            return

        # 获取主键列名
        self.cursor.execute(f"""
            SELECT column_name
            FROM {self.catalog}.information_schema.columns
            WHERE table_schema = '{self.schema}'
              AND table_name = '{table_name}'
            ORDER BY ordinal_position
            LIMIT 1
        """)
        pk_column = self.cursor.fetchone()[0]

        if table_name == 'title':
            # 删除 production_year 最密集的年份（2000-2010 附近）
            # 直方图认为这些年份有大量数据，删除后实际很少
            self.cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE {pk_column} IN (
                    SELECT {pk_column}
                    FROM {table_name}
                    WHERE production_year BETWEEN 2000 AND 2012
                    ORDER BY rand()
                    LIMIT {count}
                )
            """)

        elif table_name == 'cast_info':
            # 删除 cast 最多的热门 movie_id 的记录
            # 直方图认为这些 movie_id 有几千条 cast，删除后只剩几百条
            self.cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE {pk_column} IN (
                    SELECT {pk_column}
                    FROM {table_name}
                    WHERE movie_id IN (
                        SELECT movie_id FROM {table_name}
                        GROUP BY movie_id
                        ORDER BY COUNT(*) DESC
                        LIMIT 200
                    )
                    ORDER BY rand()
                    LIMIT {count}
                )
            """)

        elif table_name == 'movie_info':
            # 删除最频繁的 info_type_id 的记录
            self.cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE {pk_column} IN (
                    SELECT {pk_column}
                    FROM {table_name}
                    WHERE info_type_id IN (
                        SELECT info_type_id FROM {table_name}
                        GROUP BY info_type_id
                        ORDER BY COUNT(*) DESC
                        LIMIT 3
                    )
                    ORDER BY rand()
                    LIMIT {count}
                )
            """)

        elif table_name == 'movie_keyword':
            # 删除最频繁的 keyword_id 的记录
            self.cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE {pk_column} IN (
                    SELECT {pk_column}
                    FROM {table_name}
                    WHERE keyword_id IN (
                        SELECT keyword_id FROM {table_name}
                        GROUP BY keyword_id
                        ORDER BY COUNT(*) DESC
                        LIMIT 10
                    )
                    ORDER BY rand()
                    LIMIT {count}
                )
            """)

        elif table_name == 'movie_companies':
            # 删除最频繁的 company_type_id 的记录
            self.cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE {pk_column} IN (
                    SELECT {pk_column}
                    FROM {table_name}
                    WHERE company_type_id IN (
                        SELECT company_type_id FROM {table_name}
                        GROUP BY company_type_id
                        ORDER BY COUNT(*) DESC
                        LIMIT 1
                    )
                    ORDER BY rand()
                    LIMIT {count}
                )
            """)

        elif table_name == 'name':
            # 删除 ID 中间密集区域的数据
            self.cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE {pk_column} IN (
                    SELECT {pk_column}
                    FROM {table_name}
                    WHERE {pk_column} BETWEEN
                        (SELECT CAST(MAX({pk_column}) * 0.4 AS INTEGER) FROM {table_name})
                        AND (SELECT CAST(MAX({pk_column}) * 0.6 AS INTEGER) FROM {table_name})
                    ORDER BY rand()
                    LIMIT {count}
                )
            """)

        elif table_name == 'movie_info_idx':
            # 删除最频繁的 info_type_id 的记录
            self.cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE {pk_column} IN (
                    SELECT {pk_column}
                    FROM {table_name}
                    WHERE info_type_id IN (
                        SELECT info_type_id FROM {table_name}
                        GROUP BY info_type_id
                        ORDER BY COUNT(*) DESC
                        LIMIT 2
                    )
                    ORDER BY rand()
                    LIMIT {count}
                )
            """)

        else:
            # 通用策略：删除 ID 中间密集区域（40%-60%）
            self.cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE {pk_column} IN (
                    SELECT {pk_column}
                    FROM {table_name}
                    WHERE {pk_column} BETWEEN
                        (SELECT CAST(MAX({pk_column}) * 0.4 AS INTEGER) FROM {table_name})
                        AND (SELECT CAST(MAX({pk_column}) * 0.6 AS INTEGER) FROM {table_name})
                    ORDER BY rand()
                    LIMIT {count}
                )
            """)

    def _update_rows(self, table_name: str, count: int, info: Dict):
        """UPDATE: 随机更新数值列"""
        if count == 0 or not info['numeric_columns']:
            return

        # 随机选择一个数值列
        col_name, col_type = random.choice(info['numeric_columns'])

        # 获取主键列名
        self.cursor.execute(f"""
            SELECT column_name
            FROM {self.catalog}.information_schema.columns
            WHERE table_schema = '{self.schema}'
              AND table_name = '{table_name}'
            ORDER BY ordinal_position
            LIMIT 1
        """)
        pk_column = self.cursor.fetchone()[0]

        # 更新：添加 ±10% 的随机扰动
        if 'int' in col_type.lower():
            perturbation = f"CAST({col_name} * (1 + (rand() - 0.5) * 0.2) AS {col_type})"
        else:
            perturbation = f"{col_name} * (1 + (rand() - 0.5) * 0.2)"

        self.cursor.execute(f"""
            UPDATE {table_name}
            SET {col_name} = {perturbation}
            WHERE {pk_column} IN (
                SELECT {pk_column}
                FROM {table_name}
                ORDER BY rand()
                LIMIT {count}
            )
        """)

    def close(self):
        self.cursor.close()
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='Inject drift into JOB benchmark tables (equivalent to q=15 in ablation study)'
    )
    parser.add_argument('--host', default='localhost', help='Presto host')
    parser.add_argument('--port', type=int, default=8080, help='Presto port')
    parser.add_argument('--catalog', default='iceberg', help='Catalog name')
    parser.add_argument('--schema', default='imdb', help='Schema name')
    parser.add_argument('--user', default='tianqc', help='Presto user (default: tianqc)')
    parser.add_argument('--rounds', type=int, default=15,
                        help='Number of drift rounds (default: 15, equivalent to q=15)')
    parser.add_argument('--drift-ratio', type=float, default=0.02,
                        help='Drift ratio per round (default: 0.02 = 2%%)')
    parser.add_argument('--tables', nargs='+',
                        help='Specific tables to drift (default: all JOB tables)')
    parser.add_argument('--output', type=str, help='Output log file')
    parser.add_argument('--no-delete', action='store_true',
                        help='Disable DELETE operations (for Iceberg v1 compatibility)')
    parser.add_argument('--no-update', action='store_true',
                        help='Disable UPDATE operations (for Iceberg v1 compatibility)')

    args = parser.parse_args()

    # 默认 JOB 核心表（按重要性排序）
    # 包含所有事实表，维度表通常不需要漂移
    default_tables = [
        # 大事实表（优先漂移，影响最大）
        'cast_info',       # 36M rows - 最大
        'movie_info',      # 15M rows
        'movie_keyword',   # 4.5M rows
        'name',            # 4.2M rows - 重要！
        'char_name',       # 3.1M rows
        'person_info',     # 2.9M rows
        'movie_companies', # 2.6M rows
        'title',           # 2.5M rows - 最重要

        # 中等事实表
        'movie_info_idx',  # 1.4M rows
        'aka_name',        # 901K rows
        'aka_title',       # 361K rows
        'complete_cast',   # 135K rows
        'movie_link',      # 29K rows

        # 维度表（可选，通常不漂移）
        # 'company_name',  # 235K rows
        # 'keyword',       # 135K rows
        # 'comp_cast_type', 'company_type', 'info_type',
        # 'kind_type', 'link_type', 'role_type'  # 小维度表
    ]

    tables = args.tables if args.tables else default_tables

    # 确定启用的操作
    enable_delete = not args.no_delete
    enable_update = not args.no_update

    print("="*70)
    print("JOB Benchmark Drift Injection")
    print("="*70)
    print(f"Target: Equivalent to ablation study q={args.rounds}")
    print(f"Strategy: {args.drift_ratio*100:.1f}% per round × {args.rounds} rounds = {args.drift_ratio*args.rounds*100:.1f}% total")
    print(f"Tables: {', '.join(tables)}")

    # 显示操作模式
    ops = []
    ops.append("INSERT")
    if enable_delete:
        ops.append("DELETE")
    if enable_update:
        ops.append("UPDATE")
    print(f"Operations: {', '.join(ops)}")
    if not enable_delete or not enable_update:
        print(f"⚠️  Running in Iceberg v1 compatibility mode")

    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    injector = DriftInjector(args.host, args.port, args.catalog, args.schema, user=args.user)

    try:
        for round_num in range(1, args.rounds + 1):
            print(f"\n=== Drift Round {round_num}/{args.rounds} ===")

            for table_name in tables:
                try:
                    injector.inject_drift_round(table_name, round_num, args.drift_ratio,
                                               enable_delete=enable_delete,
                                               enable_update=enable_update)
                except Exception as e:
                    print(f"  ✗ Failed to drift {table_name}: {e}")
                    continue

        print("\n" + "="*70)
        print("✓ Drift injection complete")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        # 输出最终统计
        print("\nFinal table sizes:")
        for table_name in tables:
            try:
                info = injector.get_table_info(table_name)
                print(f"  {table_name}: {info['row_count']:,} rows")
            except:
                pass

    finally:
        injector.close()


if __name__ == '__main__':
    main()
