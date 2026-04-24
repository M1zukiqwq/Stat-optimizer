#!/usr/bin/env python3
"""
PostgreSQL 数据漂移注入脚本

模拟 distribution-reversal drift：
- INSERT 针对稀有值
- DELETE 针对频繁值

目标表：title, movie_info, cast_info, movie_keyword
"""

import argparse
import random
from typing import List, Tuple

import psycopg2


class DriftInjector:
    def __init__(self, db_name: str, host: str = 'localhost', port: int = 5432, user: str = None):
        """初始化 PostgreSQL 连接"""
        import os
        self.conn = psycopg2.connect(
            dbname=db_name,
            host=host,
            port=port,
            user=user or os.getenv('USER')
        )
        self.conn.autocommit = False
        self.cursor = self.conn.cursor()

    def get_table_row_count(self, table_name: str) -> int:
        """获取表的行数"""
        self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return self.cursor.fetchone()[0]

    def inject_drift_round(self, table_name: str, drift_ratio: float, round_num: int):
        """
        注入一轮数据漂移

        Args:
            table_name: 表名
            drift_ratio: 漂移比例（例如 0.02 表示 2%）
            round_num: 轮次编号
        """
        print(f"\n[Round {round_num}] Injecting drift into {table_name} (ratio={drift_ratio})")

        # 获取表的行数
        total_rows = self.get_table_row_count(table_name)
        drift_count = int(total_rows * drift_ratio)

        print(f"  Total rows: {total_rows:,}")
        print(f"  Drift count: {drift_count:,}")

        # 根据表类型选择漂移策略
        if table_name == 'title':
            self._inject_title_drift(drift_count, round_num)
        elif table_name == 'movie_info':
            self._inject_movie_info_drift(drift_count, round_num)
        elif table_name == 'cast_info':
            self._inject_cast_info_drift(drift_count, round_num)
        elif table_name == 'movie_keyword':
            self._inject_movie_keyword_drift(drift_count, round_num)
        else:
            print(f"  ⚠ Unsupported table: {table_name}")
            return

        self.conn.commit()
        print(f"  ✓ Drift injected successfully")

    def _inject_title_drift(self, drift_count: int, round_num: int):
        """
        title 表漂移策略：
        - DELETE: 删除 production_year 频繁值（例如 2000-2010）
        - INSERT: 插入 production_year 稀有值（例如 1920-1950）
        """
        # DELETE: 删除频繁年份的电影
        delete_years = list(range(2000, 2011))  # 2000-2010
        delete_sql = f"""
        DELETE FROM title
        WHERE id IN (
            SELECT id FROM title
            WHERE production_year = ANY(%s)
            ORDER BY RANDOM()
            LIMIT %s
        )
        """
        self.cursor.execute(delete_sql, (delete_years, drift_count // 2))
        deleted = self.cursor.rowcount
        print(f"    Deleted {deleted} rows (frequent years: 2000-2010)")

        # INSERT: 插入稀有年份的电影
        insert_years = list(range(1920, 1951))  # 1920-1950
        insert_sql = """
        INSERT INTO title (id, title, kind_id, production_year)
        SELECT
            (SELECT MAX(id) FROM title) + ROW_NUMBER() OVER (),
            'Synthetic Movie ' || ROW_NUMBER() OVER (),
            1,  -- kind_id = 1 (movie)
            (ARRAY[%s])[1 + (RANDOM() * %s)::int]
        FROM generate_series(1, %s)
        """
        self.cursor.execute(insert_sql, (insert_years, len(insert_years) - 1, drift_count // 2))
        inserted = self.cursor.rowcount
        print(f"    Inserted {inserted} rows (rare years: 1920-1950)")

    def _inject_movie_info_drift(self, drift_count: int, round_num: int):
        """
        movie_info 表漂移策略：
        - DELETE: 删除 info_type_id 频繁值
        - INSERT: 插入 info_type_id 稀有值
        """
        # 获取频繁和稀有的 info_type_id
        self.cursor.execute("""
            SELECT info_type_id, COUNT(*) as cnt
            FROM movie_info
            GROUP BY info_type_id
            ORDER BY cnt DESC
        """)
        info_types = self.cursor.fetchall()

        frequent_types = [t[0] for t in info_types[:5]]  # 前 5 个频繁类型
        rare_types = [t[0] for t in info_types[-5:]]     # 后 5 个稀有类型

        # DELETE: 删除频繁类型
        delete_sql = """
        DELETE FROM movie_info
        WHERE id IN (
            SELECT id FROM movie_info
            WHERE info_type_id = ANY(%s)
            ORDER BY RANDOM()
            LIMIT %s
        )
        """
        self.cursor.execute(delete_sql, (frequent_types, drift_count // 2))
        deleted = self.cursor.rowcount
        print(f"    Deleted {deleted} rows (frequent info_type_id)")

        # INSERT: 插入稀有类型
        insert_sql = """
        INSERT INTO movie_info (id, movie_id, info_type_id, info)
        SELECT
            (SELECT MAX(id) FROM movie_info) + ROW_NUMBER() OVER (),
            (SELECT id FROM title ORDER BY RANDOM() LIMIT 1),
            (ARRAY[%s])[1 + (RANDOM() * %s)::int],
            'Synthetic info ' || ROW_NUMBER() OVER ()
        FROM generate_series(1, %s)
        """
        self.cursor.execute(insert_sql, (rare_types, len(rare_types) - 1, drift_count // 2))
        inserted = self.cursor.rowcount
        print(f"    Inserted {inserted} rows (rare info_type_id)")

    def _inject_cast_info_drift(self, drift_count: int, round_num: int):
        """
        cast_info 表漂移策略：
        - DELETE: 删除 role_id 频繁值
        - INSERT: 插入 role_id 稀有值
        """
        # 获取频繁和稀有的 role_id
        self.cursor.execute("""
            SELECT role_id, COUNT(*) as cnt
            FROM cast_info
            GROUP BY role_id
            ORDER BY cnt DESC
        """)
        roles = self.cursor.fetchall()

        frequent_roles = [r[0] for r in roles[:2]]  # 前 2 个频繁角色
        rare_roles = [r[0] for r in roles[-2:]]     # 后 2 个稀有角色

        # DELETE: 删除频繁角色
        delete_sql = """
        DELETE FROM cast_info
        WHERE id IN (
            SELECT id FROM cast_info
            WHERE role_id = ANY(%s)
            ORDER BY RANDOM()
            LIMIT %s
        )
        """
        self.cursor.execute(delete_sql, (frequent_roles, drift_count // 2))
        deleted = self.cursor.rowcount
        print(f"    Deleted {deleted} rows (frequent role_id)")

        # INSERT: 插入稀有角色
        insert_sql = """
        INSERT INTO cast_info (id, person_id, movie_id, role_id)
        SELECT
            (SELECT MAX(id) FROM cast_info) + ROW_NUMBER() OVER (),
            (SELECT id FROM name ORDER BY RANDOM() LIMIT 1),
            (SELECT id FROM title ORDER BY RANDOM() LIMIT 1),
            (ARRAY[%s])[1 + (RANDOM() * %s)::int]
        FROM generate_series(1, %s)
        """
        self.cursor.execute(insert_sql, (rare_roles, len(rare_roles) - 1, drift_count // 2))
        inserted = self.cursor.rowcount
        print(f"    Inserted {inserted} rows (rare role_id)")

    def _inject_movie_keyword_drift(self, drift_count: int, round_num: int):
        """
        movie_keyword 表漂移策略：
        - DELETE: 删除 keyword_id 频繁值
        - INSERT: 插入 keyword_id 稀有值
        """
        # 获取频繁和稀有的 keyword_id
        self.cursor.execute("""
            SELECT keyword_id, COUNT(*) as cnt
            FROM movie_keyword
            GROUP BY keyword_id
            ORDER BY cnt DESC
        """)
        keywords = self.cursor.fetchall()

        frequent_keywords = [k[0] for k in keywords[:10]]  # 前 10 个频繁关键词
        rare_keywords = [k[0] for k in keywords[-10:]]     # 后 10 个稀有关键词

        # DELETE: 删除频繁关键词
        delete_sql = """
        DELETE FROM movie_keyword
        WHERE id IN (
            SELECT id FROM movie_keyword
            WHERE keyword_id = ANY(%s)
            ORDER BY RANDOM()
            LIMIT %s
        )
        """
        self.cursor.execute(delete_sql, (frequent_keywords, drift_count // 2))
        deleted = self.cursor.rowcount
        print(f"    Deleted {deleted} rows (frequent keyword_id)")

        # INSERT: 插入稀有关键词
        insert_sql = """
        INSERT INTO movie_keyword (id, movie_id, keyword_id)
        SELECT
            (SELECT MAX(id) FROM movie_keyword) + ROW_NUMBER() OVER (),
            (SELECT id FROM title ORDER BY RANDOM() LIMIT 1),
            (ARRAY[%s])[1 + (RANDOM() * %s)::int]
        FROM generate_series(1, %s)
        """
        self.cursor.execute(insert_sql, (rare_keywords, len(rare_keywords) - 1, drift_count // 2))
        inserted = self.cursor.rowcount
        print(f"    Inserted {inserted} rows (rare keyword_id)")

    def run_drift_injection(self, drift_rounds: int, drift_ratio: float):
        """
        运行多轮数据漂移注入

        Args:
            drift_rounds: 漂移轮数
            drift_ratio: 每轮漂移比例
        """
        print(f"\n{'='*70}")
        print(f"Starting drift injection: {drift_rounds} rounds, {drift_ratio*100}% per round")
        print(f"{'='*70}")

        target_tables = ['title', 'movie_info', 'cast_info', 'movie_keyword']

        for round_num in range(1, drift_rounds + 1):
            print(f"\n{'='*70}")
            print(f"Round {round_num}/{drift_rounds}")
            print(f"{'='*70}")

            for table in target_tables:
                try:
                    self.inject_drift_round(table, drift_ratio, round_num)
                except Exception as e:
                    print(f"  ✗ Error injecting drift into {table}: {e}")
                    self.conn.rollback()

        print(f"\n{'='*70}")
        print(f"✓ Drift injection completed: {drift_rounds} rounds")
        print(f"{'='*70}")

    def close(self):
        self.cursor.close()
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description='Inject data drift into PostgreSQL IMDB database')
    parser.add_argument('--db-name', required=True, help='Database name')
    parser.add_argument('--host', default='localhost', help='PostgreSQL host')
    parser.add_argument('--port', type=int, default=5432, help='PostgreSQL port')
    parser.add_argument('--user', default=None, help='PostgreSQL user (default: current user)')
    parser.add_argument('--drift-rounds', type=int, default=15, help='Number of drift rounds')
    parser.add_argument('--drift-ratio', type=float, default=0.02, help='Drift ratio per round (e.g., 0.02 = 2%)')

    args = parser.parse_args()

    injector = DriftInjector(
        db_name=args.db_name,
        host=args.host,
        port=args.port,
        user=args.user
    )

    try:
        injector.run_drift_injection(
            drift_rounds=args.drift_rounds,
            drift_ratio=args.drift_ratio
        )
    finally:
        injector.close()


if __name__ == '__main__':
    main()
