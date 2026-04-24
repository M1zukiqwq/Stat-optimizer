#!/usr/bin/env python3
"""
PostgreSQL 漂移注入脚本
模拟 q=15 轮漂移：分布反转策略
- INSERT: 让稀少值变多（1920-1950 老电影）
- DELETE: 让频繁值变少（2000-2012 新电影）
每轮影响约 2% 数据，15 轮累计约 30%
"""

import argparse
import psycopg2
import sys

DRIFT_ROUNDS = 15
DRIFT_RATIO = 0.02  # 每轮影响 2%

# 每张表的漂移策略：(insert_where, delete_where, insert_template)
DRIFT_CONFIG = {
    'title': {
        'delete_where': "production_year BETWEEN 2000 AND 2012",
        'insert_sql': """
            INSERT INTO title (id, title, imdb_index, kind_id, production_year,
                               imdb_id, phonetic_code, episode_of_id, season_nr,
                               episode_nr, series_years, md5sum)
            SELECT
                {max_id} + row_number() OVER (),
                'Drift Old Movie ' || gs,
                NULL, 1,
                1920 + (gs % 31),
                NULL, NULL, NULL, NULL, NULL, NULL,
                md5(gs::text)
            FROM generate_series(1, {batch_size}) gs
        """,
    },
    'cast_info': {
        'delete_where': "role_id IN (1, 2) AND nr_order > 5",
        'insert_sql': """
            INSERT INTO cast_info (id, person_id, movie_id, person_role_id,
                                   note, nr_order, role_id)
            SELECT
                {max_id} + row_number() OVER (),
                (SELECT id FROM name OFFSET floor(random()*4000000)::int LIMIT 1),
                (SELECT id FROM title WHERE production_year BETWEEN 1920 AND 1950
                 OFFSET floor(random()*80000)::int LIMIT 1),
                NULL, NULL,
                gs % 10 + 1,
                gs % 12 + 1
            FROM generate_series(1, {batch_size}) gs
        """,
    },
    'movie_info': {
        'delete_where': "info_type_id BETWEEN 1 AND 5 AND info ~ '^[0-9]'",
        'insert_sql': """
            INSERT INTO movie_info (id, movie_id, info_type_id, info, note)
            SELECT
                {max_id} + row_number() OVER (),
                (SELECT id FROM title WHERE production_year BETWEEN 1920 AND 1950
                 OFFSET floor(random()*80000)::int LIMIT 1),
                gs % 10 + 1,
                'drift info ' || gs,
                NULL
            FROM generate_series(1, {batch_size}) gs
        """,
    },
    'name': {
        # 删除姓名以常见字母开头的人（模拟名人减少，无名者增多）
        'delete_where': "name LIKE 'A%' AND gender IS NULL",
        'insert_sql': """
            INSERT INTO name (id, name, imdb_index, imdb_id, gender,
                              name_pcode_cf, name_pcode_nf, surname_pcode, md5sum)
            SELECT
                {max_id} + row_number() OVER (),
                'Drift Person Z' || gs,
                NULL, NULL, NULL, NULL, NULL, NULL,
                md5(('name'||gs)::text)
            FROM generate_series(1, {batch_size}) gs
        """,
    },
    'movie_companies': {
        # 删除大型发行公司记录（company_type_id=2 通常是发行商）
        'delete_where': "company_type_id = 2",
        'insert_sql': """
            INSERT INTO movie_companies (id, movie_id, company_id, company_type_id, note)
            SELECT
                {max_id} + row_number() OVER (),
                (SELECT id FROM title WHERE production_year BETWEEN 1920 AND 1950
                 OFFSET floor(random()*80000)::int LIMIT 1),
                (SELECT id FROM company_name OFFSET floor(random()*200000)::int LIMIT 1),
                1,
                NULL
            FROM generate_series(1, {batch_size}) gs
        """,
    },
    'movie_keyword': {
        # 删除高频关键词关联（keyword_id 越小通常越常见）
        'delete_where': "keyword_id < 1000",
        'insert_sql': """
            INSERT INTO movie_keyword (id, movie_id, keyword_id)
            SELECT
                {max_id} + row_number() OVER (),
                (SELECT id FROM title WHERE production_year BETWEEN 1920 AND 1950
                 OFFSET floor(random()*80000)::int LIMIT 1),
                50000 + (gs % 80000)
            FROM generate_series(1, {batch_size}) gs
        """,
    },
}


def inject_drift(conn, rounds: int, drift_ratio: float, tables: list):
    cur = conn.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")

    for rnd in range(1, rounds + 1):
        print(f"\n--- Round {rnd}/{rounds} ---")
        for table in tables:
            if table not in DRIFT_CONFIG:
                continue
            cfg = DRIFT_CONFIG[table]

            # 计算批次大小
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            total = cur.fetchone()[0]
            batch_size = max(100, int(total * drift_ratio))

            # DELETE
            del_sql = f"DELETE FROM {table} WHERE {cfg['delete_where']} AND id IN (SELECT id FROM {table} WHERE {cfg['delete_where']} LIMIT {batch_size})"
            cur.execute(del_sql)
            deleted = cur.rowcount
            conn.commit()

            # INSERT — 先锁住 MAX(id) 再插入，避免并发冲突
            cur.execute(f"SELECT MAX(id) FROM {table}")
            max_id = cur.fetchone()[0] or 0
            ins_sql = cfg['insert_sql'].format(batch_size=batch_size, max_id=max_id)
            cur.execute(ins_sql)
            inserted = cur.rowcount
            conn.commit()

            print(f"  {table}: deleted={deleted}, inserted={inserted}")

    cur.close()
    print("\nDrift injection complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbname', default='imdb')
    parser.add_argument('--user', default='qichutian')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=5432)
    parser.add_argument('--rounds', type=int, default=DRIFT_ROUNDS)
    parser.add_argument('--drift-ratio', type=float, default=DRIFT_RATIO)
    parser.add_argument('--tables', nargs='+',
                        default=['title', 'cast_info', 'movie_info',
                                 'name', 'movie_companies', 'movie_keyword'])
    args = parser.parse_args()

    conn = psycopg2.connect(
        host=args.host, port=args.port,
        dbname=args.dbname, user=args.user
    )
    conn.autocommit = False

    try:
        inject_drift(conn, args.rounds, args.drift_ratio, args.tables)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
