#!/usr/bin/env python3
"""
使用 TPC-DS 数据库验证 Q-Error 提取
"""

import json
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, str(Path(__file__).parent))
from run_simple_pg_experiment import PostgreSQLExperimentRunner


def test_simple_scan(conn, cursor):
    """测试简单扫描"""
    print("="*70)
    print("Test 1: Simple Seq Scan")
    print("="*70)
    
    sql = "SELECT * FROM item WHERE i_current_price > 50 LIMIT 100"
    print(f"Query: {sql}")
    print()
    
    cursor.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
    rows = cursor.fetchall()
    
    first_row = rows[0] if rows else None
    if first_row:
        if isinstance(first_row, dict):
            # RealDictCursor 返回字典
            explain_output = list(first_row.values())[0]
        else:
            explain_output = first_row[0]
    else:
        explain_output = '[]'
    
    # 打印原始计划
    try:
        plan_data = json.loads(explain_output)
        print("Plan structure:")
        print(json.dumps(plan_data, indent=2)[:2000])
        print()
    except:
        print("Raw output:", explain_output[:1000])
    
    # 提取 Q-Error
    runner = PostgreSQLExperimentRunner.__new__(PostgreSQLExperimentRunner)
    stats = runner.extract_plan_stats(explain_output)
    
    print(f"\nExtracted {len(stats)} operators:")
    for stat in stats:
        rel = stat['relation'] or 'N/A'
        print(f"  - {stat['operator']} on {rel}")
        print(f"    Est: {stat['estimated_rows']:,}, Actual: {stat['actual_rows']:,}, Q-Error: {stat['q_error']:.2f}")
    
    return len(stats) > 0


def test_join_query(conn, cursor):
    """测试 JOIN 查询"""
    print("\n" + "="*70)
    print("Test 2: Join Query")
    print("="*70)
    
    sql = """
        SELECT COUNT(*)
        FROM store_sales ss
        JOIN item i ON ss.ss_item_sk = i.i_item_sk
        WHERE i.i_current_price > 100
    """
    print(f"Query: {sql.strip()}")
    print()
    
    cursor.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
    rows = cursor.fetchall()
    
    first_row = rows[0] if rows else None
    if first_row:
        if isinstance(first_row, dict):
            # RealDictCursor 返回字典
            explain_output = list(first_row.values())[0]
        else:
            explain_output = first_row[0]
    else:
        explain_output = '[]'
    
    # 提取 Q-Error
    runner = PostgreSQLExperimentRunner.__new__(PostgreSQLExperimentRunner)
    stats = runner.extract_plan_stats(explain_output)
    
    print(f"Extracted {len(stats)} operators:")
    
    # 分组统计
    scans = [s for s in stats if 'Scan' in s['operator']]
    joins = [s for s in stats if 'Join' in s['operator']]
    
    if scans:
        print(f"\nScans ({len(scans)}):")
        for s in scans:
            rel = s['relation'] or 'N/A'
            print(f"  - {s['operator']} on {rel}: est={s['estimated_rows']:,}, act={s['actual_rows']:,}, qe={s['q_error']:.2f}")
    
    if joins:
        print(f"\nJoins ({len(joins)}):")
        for j in joins:
            print(f"  - {j['operator']}: est={j['estimated_rows']:,}, act={j['actual_rows']:,}, qe={j['q_error']:.2f}")
    
    if stats:
        q_errors = [s['q_error'] for s in stats]
        geom_mean = runner.geometric_mean(q_errors)
        print(f"\nGeometric Mean Q-Error: {geom_mean:.2f}")
    
    return len(stats) > 0


def test_index_scan(conn, cursor):
    """测试索引扫描"""
    print("\n" + "="*70)
    print("Test 3: Index Scan")
    print("="*70)
    
    sql = "SELECT * FROM item WHERE i_item_sk = 1000"
    print(f"Query: {sql}")
    print()
    
    cursor.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
    rows = cursor.fetchall()
    
    first_row = rows[0] if rows else None
    if first_row:
        if isinstance(first_row, dict):
            # RealDictCursor 返回字典
            explain_output = list(first_row.values())[0]
        else:
            explain_output = first_row[0]
    else:
        explain_output = '[]'
    
    runner = PostgreSQLExperimentRunner.__new__(PostgreSQLExperimentRunner)
    stats = runner.extract_plan_stats(explain_output)
    
    print(f"Extracted {len(stats)} operators:")
    for stat in stats:
        rel = stat['relation'] or 'N/A'
        print(f"  - {stat['operator']} on {rel}: est={stat['estimated_rows']:,}, act={stat['actual_rows']:,}, qe={stat['q_error']:.2f}")
    
    return len(stats) > 0


def test_complex_query(conn, cursor):
    """测试复杂查询"""
    print("\n" + "="*70)
    print("Test 4: Complex Query with Multiple Joins")
    print("="*70)
    
    sql = """
        SELECT 
            d.d_year,
            i.i_category,
            COUNT(*) as cnt
        FROM store_sales ss
        JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
        JOIN item i ON ss.ss_item_sk = i.i_item_sk
        WHERE d.d_year = 2000
          AND i.i_current_price > 50
        GROUP BY d.d_year, i.i_category
        LIMIT 100
    """
    print(f"Query: {sql.strip()}")
    print()
    
    cursor.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
    rows = cursor.fetchall()
    
    first_row = rows[0] if rows else None
    if first_row:
        if isinstance(first_row, dict):
            # RealDictCursor 返回字典
            explain_output = list(first_row.values())[0]
        else:
            explain_output = first_row[0]
    else:
        explain_output = '[]'
    
    runner = PostgreSQLExperimentRunner.__new__(PostgreSQLExperimentRunner)
    stats = runner.extract_plan_stats(explain_output)
    
    print(f"Extracted {len(stats)} operators:")
    
    # 显示 Top 5 最差 Q-Error
    if stats:
        sorted_stats = sorted(stats, key=lambda x: x['q_error'], reverse=True)
        print("\nTop 5 by Q-Error:")
        for i, s in enumerate(sorted_stats[:5], 1):
            rel = f" on {s['relation']}" if s['relation'] else ""
            print(f"  {i}. {s['operator']}{rel}: qe={s['q_error']:.2f} (est={s['estimated_rows']:,}, act={s['actual_rows']:,})")
        
        q_errors = [s['q_error'] for s in stats]
        geom_mean = runner.geometric_mean(q_errors)
        mean_qe = sum(q_errors) / len(q_errors)
        max_qe = max(q_errors)
        
        print(f"\nOverall:")
        print(f"  Geometric Mean Q-Error: {geom_mean:.2f}")
        print(f"  Mean Q-Error:           {mean_qe:.2f}")
        print(f"  Max Q-Error:            {max_qe:.2f}")
    
    return len(stats) > 0


def main():
    print("PostgreSQL Q-Error Extraction Verification (TPC-DS)")
    print("="*70)
    
    # 连接数据库
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            dbname='tpcds',
            user='qichutian'
        )
        conn.autocommit = True
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        print("✓ Connected to PostgreSQL (tpcds database)")
        print()
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return 1
    
    try:
        # 禁用并行查询以避免配置问题
        cursor.execute("SET max_parallel_workers_per_gather = 0")
        
        results = []
        
        results.append(("Simple Scan", test_simple_scan(conn, cursor)))
        results.append(("Join Query", test_join_query(conn, cursor)))
        results.append(("Index Scan", test_index_scan(conn, cursor)))
        results.append(("Complex Query", test_complex_query(conn, cursor)))
        
        print("\n" + "="*70)
        print("Test Summary")
        print("="*70)
        for name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{status}: {name}")
        
        all_passed = all(r[1] for r in results)
        print("="*70)
        if all_passed:
            print("✓ All tests PASSED!")
        else:
            print("✗ Some tests FAILED")
        
        return 0 if all_passed else 1
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cursor.close()
        conn.close()
        print("\n✓ Disconnected from PostgreSQL")


if __name__ == '__main__':
    sys.exit(main())
