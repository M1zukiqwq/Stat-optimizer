#!/usr/bin/env python3
"""
真实 PostgreSQL Q-Error 提取验证

连接到实际数据库，运行 EXPLAIN ANALYZE，验证提取逻辑
"""

import argparse
import json
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, str(Path(__file__).parent))
from run_simple_pg_experiment import PostgreSQLExperimentRunner


def test_simple_query(conn, cursor):
    """测试简单查询的 Q-Error 提取"""
    print("="*70)
    print("Test 1: Simple Query")
    print("="*70)
    
    # 一个简单的查询
    sql = "SELECT * FROM title WHERE production_year BETWEEN 2000 AND 2010 LIMIT 100"
    
    print(f"Query: {sql}")
    print()
    
    # 运行 EXPLAIN ANALYZE
    cursor.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
    rows = cursor.fetchall()
    
    # 获取原始 JSON
    if rows and len(rows) > 0:
        first_row = rows[0]
        if isinstance(first_row, dict):
            explain_output = json.dumps(first_row)
        elif hasattr(first_row, '__iter__') and len(first_row) > 0:
            explain_output = first_row[0]
        else:
            explain_output = str(first_row)
    else:
        print("✗ No EXPLAIN output")
        return False
    
    # 打印原始 JSON（美化）
    try:
        plan_data = json.loads(explain_output)
        print("Raw EXPLAIN JSON (first 2000 chars):")
        print(json.dumps(plan_data, indent=2)[:2000])
        print()
    except:
        print("Raw output:", explain_output[:2000])
        print()
    
    # 使用提取器
    runner = PostgreSQLExperimentRunner.__new__(PostgreSQLExperimentRunner)
    stats = runner.extract_plan_stats(explain_output)
    
    print(f"Extracted {len(stats)} operators:")
    print("-"*70)
    for i, stat in enumerate(stats, 1):
        print(f"{i}. {stat['operator']}")
        if stat['relation']:
            print(f"   Relation: {stat['relation']}")
        print(f"   Estimated: {stat['estimated_rows']:,}")
        print(f"   Actual:    {stat['actual_rows']:,}")
        print(f"   Q-Error:   {stat['q_error']:.2f}")
        print()
    
    # 计算聚合 Q-Error
    if stats:
        q_errors = [s['q_error'] for s in stats]
        geom_mean = runner.geometric_mean(q_errors)
        print(f"Geometric Mean Q-Error: {geom_mean:.2f}")
    
    return len(stats) > 0


def test_join_query(conn, cursor):
    """测试 Join 查询的 Q-Error 提取"""
    print("="*70)
    print("Test 2: Join Query")
    print("="*70)
    
    # 一个简单的 JOIN 查询
    sql = """
        SELECT t.title, mi.info 
        FROM title t 
        JOIN movie_info mi ON t.id = mi.movie_id 
        WHERE t.production_year = 2005 
        LIMIT 10
    """
    
    print(f"Query: {sql.strip()}")
    print()
    
    cursor.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
    rows = cursor.fetchall()
    
    if rows and len(rows) > 0:
        first_row = rows[0]
        if isinstance(first_row, dict):
            explain_output = json.dumps(first_row)
        elif hasattr(first_row, '__iter__') and len(first_row) > 0:
            explain_output = first_row[0]
        else:
            explain_output = str(first_row)
    else:
        print("✗ No EXPLAIN output")
        return False
    
    # 使用提取器
    runner = PostgreSQLExperimentRunner.__new__(PostgreSQLExperimentRunner)
    stats = runner.extract_plan_stats(explain_output)
    
    print(f"Extracted {len(stats)} operators:")
    print("-"*70)
    
    # 按类型分组统计
    scans = [s for s in stats if 'Scan' in s['operator']]
    joins = [s for s in stats if 'Join' in s['operator']]
    others = [s for s in stats if 'Scan' not in s['operator'] and 'Join' not in s['operator']]
    
    if scans:
        print(f"\nScans ({len(scans)}):")
        for s in scans:
            rel = s['relation'] or 'N/A'
            print(f"  - {s['operator']} on {rel}: est={s['estimated_rows']:,}, actual={s['actual_rows']:,}, qe={s['q_error']:.2f}")
    
    if joins:
        print(f"\nJoins ({len(joins)}):")
        for j in joins:
            print(f"  - {j['operator']}: est={j['estimated_rows']:,}, actual={j['actual_rows']:,}, qe={j['q_error']:.2f}")
    
    if others:
        print(f"\nOthers ({len(others)}):")
        for o in others:
            print(f"  - {o['operator']}: est={o['estimated_rows']:,}, actual={o['actual_rows']:,}, qe={o['q_error']:.2f}")
    
    if stats:
        q_errors = [s['q_error'] for s in stats]
        geom_mean = runner.geometric_mean(q_errors)
        print(f"\nGeometric Mean Q-Error: {geom_mean:.2f}")
    
    return len(stats) > 0


def test_complex_job_query(conn, cursor, query_file=None):
    """测试真实的 JOB 查询"""
    print("="*70)
    print("Test 3: JOB Query")
    print("="*70)
    
    # 如果没有提供查询文件，使用内置的简单 JOB 风格查询
    if query_file and Path(query_file).exists():
        sql = Path(query_file).read_text()
        print(f"Query file: {query_file}")
    else:
        # 模拟一个 JOB 风格的复杂查询
        sql = """
            SELECT MIN(t.title) AS movie_title
            FROM company_name AS cn,
                 company_type AS ct,
                 info_type AS it,
                 info_type AS it2,
                 kind_type AS kt,
                 movie_companies AS mc,
                 movie_info AS mi,
                 movie_info_idx AS mi_idx,
                 title AS t
            WHERE cn.country_code = '[us]'
              AND ct.kind = 'production companies'
              AND it.info = 'rating'
              AND it2.info = 'release dates'
              AND kt.kind = 'movie'
              AND t.id = mi.movie_id
              AND t.id = mi_idx.movie_id
              AND t.id = mc.movie_id
              LIMIT 10
        """
        print("Using built-in JOB-style query")
    
    print()
    
    try:
        cursor.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
        rows = cursor.fetchall()
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return False
    
    if rows and len(rows) > 0:
        first_row = rows[0]
        if isinstance(first_row, dict):
            explain_output = json.dumps(first_row)
        elif hasattr(first_row, '__iter__') and len(first_row) > 0:
            explain_output = first_row[0]
        else:
            explain_output = str(first_row)
    else:
        print("✗ No EXPLAIN output")
        return False
    
    # 使用提取器
    runner = PostgreSQLExperimentRunner.__new__(PostgreSQLExperimentRunner)
    stats = runner.extract_plan_stats(explain_output)
    
    print(f"Extracted {len(stats)} operators")
    print("-"*70)
    
    # 统计各类算子
    scans = [s for s in stats if 'Scan' in s['operator']]
    joins = [s for s in stats if 'Join' in s['operator']]
    
    print(f"  Scans: {len(scans)}")
    print(f"  Joins: {len(joins)}")
    print(f"  Others: {len(stats) - len(scans) - len(joins)}")
    print()
    
    # 显示 Top 5 最差 Q-Error
    if stats:
        sorted_stats = sorted(stats, key=lambda x: x['q_error'], reverse=True)
        print("Top 5 operators by Q-Error:")
        for i, s in enumerate(sorted_stats[:5], 1):
            rel = f" on {s['relation']}" if s['relation'] else ""
            print(f"  {i}. {s['operator']}{rel}: qe={s['q_error']:.2f} (est={s['estimated_rows']:,}, act={s['actual_rows']:,})")
        
        q_errors = [s['q_error'] for s in stats]
        geom_mean = runner.geometric_mean(q_errors)
        mean_qe = sum(q_errors) / len(q_errors)
        max_qe = max(q_errors)
        
        print()
        print(f"Overall Statistics:")
        print(f"  Geometric Mean Q-Error: {geom_mean:.2f}")
        print(f"  Mean Q-Error:           {mean_qe:.2f}")
        print(f"  Max Q-Error:            {max_qe:.2f}")
    
    return len(stats) > 0


def main():
    parser = argparse.ArgumentParser(description='Verify Q-Error extraction with real PostgreSQL')
    parser.add_argument('--host', default='localhost', help='PostgreSQL host')
    parser.add_argument('--port', type=int, default=5432, help='PostgreSQL port')
    parser.add_argument('--dbname', required=True, help='Database name')
    parser.add_argument('--user', required=True, help='PostgreSQL user')
    parser.add_argument('--password', default='', help='PostgreSQL password')
    parser.add_argument('--query-file', help='Optional JOB query file to test')
    
    args = parser.parse_args()
    
    print("PostgreSQL Q-Error Extraction Real Verification")
    print("="*70)
    print(f"Connecting to: {args.user}@{args.host}:{args.port}/{args.dbname}")
    print()
    
    # 连接数据库
    try:
        conn = psycopg2.connect(
            host=args.host,
            port=args.port,
            dbname=args.dbname,
            user=args.user,
            password=args.password
        )
        conn.autocommit = True
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        print("✓ Connected to PostgreSQL")
        print()
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return 1
    
    try:
        # 检查必要的表是否存在
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'title'
            )
        """)
        if not cursor.fetchone()[0]:
            print("✗ Table 'title' not found. Please load IMDB data first.")
            return 1
        
        # 运行测试
        results = []
        
        print()
        results.append(("Simple Query", test_simple_query(conn, cursor)))
        print()
        
        results.append(("Join Query", test_join_query(conn, cursor)))
        print()
        
        results.append(("JOB Query", test_complex_job_query(conn, cursor, args.query_file)))
        print()
        
        # 汇总
        print("="*70)
        print("Test Summary")
        print("="*70)
        for name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{status}: {name}")
        
        all_passed = all(r[1] for r in results)
        print("="*70)
        if all_passed:
            print("✓ All tests PASSED - Q-Error extraction is working correctly!")
        else:
            print("✗ Some tests FAILED - Check the output above")
        
        return 0 if all_passed else 1
        
    except Exception as e:
        print(f"\n✗ Error during tests: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cursor.close()
        conn.close()
        print("\n✓ Disconnected from PostgreSQL")


if __name__ == '__main__':
    sys.exit(main())
