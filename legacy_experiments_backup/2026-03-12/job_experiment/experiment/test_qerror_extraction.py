#!/usr/bin/env python3
"""
测试 Q-Error 提取逻辑
验证 PostgreSQL EXPLAIN JSON 解析是否正确
"""

import json
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from run_simple_pg_experiment import PostgreSQLExperimentRunner


def test_qerror_calculation():
    """测试 Q-Error 计算公式"""
    runner = PostgreSQLExperimentRunner.__new__(PostgreSQLExperimentRunner)
    
    print("Testing Q-Error calculation...")
    print("="*60)
    
    test_cases = [
        # (estimated, actual, expected_qerror, description)
        (100, 100, 1.0, "完美估算"),
        (200, 100, 2.0, "高估 2x"),
        (100, 200, 2.0, "低估 2x"),
        (1000, 100, 10.0, "高估 10x"),
        (100, 1000, 10.0, "低估 10x"),
        (0, 0, 1.0, "都为空"),
        (100, 0, 101.0, "实际为空，高估"),
        (0, 100, 101.0, "估算为空，低估"),
        (1, 1, 1.0, "单行完美估算"),
        (50, 100, 2.0, "低估 2x"),
    ]
    
    all_passed = True
    for est, actual, expected, desc in test_cases:
        result = runner.calculate_q_error(est, actual)
        passed = abs(result - expected) < 0.01
        status = "✓" if passed else "✗"
        print(f"{status} {desc}: est={est}, actual={actual}, q-error={result:.2f} (expected {expected})")
        if not passed:
            all_passed = False
    
    print("="*60)
    print(f"Test {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_explain_parsing_simple():
    """测试简单 EXPLAIN JSON 解析"""
    runner = PostgreSQLExperimentRunner.__new__(PostgreSQLExperimentRunner)
    
    print("\nTesting simple EXPLAIN JSON parsing...")
    print("="*60)
    
    # 模拟 PostgreSQL EXPLAIN (ANALYZE, FORMAT JSON) 输出 - Seq Scan
    sample_explain = '''
    [{
      "Plan": {
        "Node Type": "Seq Scan",
        "Parallel Aware": false,
        "Async Capable": false,
        "Relation Name": "title",
        "Alias": "t",
        "Startup Cost": 0.00,
        "Total Cost": 1000.00,
        "Plan Rows": 10000,
        "Plan Width": 25,
        "Actual Startup Time": 0.001,
        "Actual Total Time": 2.000,
        "Actual Rows": 8000,
        "Actual Loops": 1
      },
      "Planning Time": 0.5,
      "Execution Time": 5.0
    }]
    '''
    
    stats = runner.extract_plan_stats(sample_explain)
    
    print(f"Extracted {len(stats)} operator stats:")
    for i, stat in enumerate(stats, 1):
        print(f"  {i}. {stat['operator']} on {stat['relation'] or 'N/A'}")
        print(f"     Est: {stat['estimated_rows']}, Actual: {stat['actual_rows']}, Q-Error: {stat['q_error']:.2f}")
    
    # 验证
    if len(stats) != 1:
        print(f"✗ Expected 1 operator, got {len(stats)}")
        return False
    
    stat = stats[0]
    checks = [
        (stat['operator'] == 'Seq Scan', f"Operator: {stat['operator']} == Seq Scan"),
        (stat['relation'] == 'title', f"Relation: {stat['relation']} == title"),
        (stat['estimated_rows'] == 10000, f"Est: {stat['estimated_rows']} == 10000"),
        (stat['actual_rows'] == 8000, f"Actual: {stat['actual_rows']} == 8000"),
        (abs(stat['q_error'] - 1.25) < 0.01, f"Q-Error: {stat['q_error']:.2f} ≈ 1.25"),
    ]
    
    all_passed = True
    for passed, desc in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {desc}")
        if not passed:
            all_passed = False
    
    print("="*60)
    print(f"Test {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_explain_parsing_join():
    """测试 Join 查询的 EXPLAIN JSON 解析"""
    runner = PostgreSQLExperimentRunner.__new__(PostgreSQLExperimentRunner)
    
    print("\nTesting Join query EXPLAIN JSON parsing...")
    print("="*60)
    
    # 模拟包含 Hash Join 的复杂计划树
    sample_explain = '''
    [{
      "Plan": {
        "Node Type": "Hash Join",
        "Parallel Aware": false,
        "Async Capable": false,
        "Join Type": "Inner",
        "Startup Cost": 1234.56,
        "Total Cost": 5678.90,
        "Plan Rows": 1000,
        "Plan Width": 50,
        "Actual Startup Time": 1.234,
        "Actual Total Time": 5.678,
        "Actual Rows": 500,
        "Actual Loops": 1,
        "Inner Unique": false,
        "Plans": [
          {
            "Node Type": "Seq Scan",
            "Parent Relationship": "Outer",
            "Parallel Aware": false,
            "Async Capable": false,
            "Relation Name": "title",
            "Alias": "t",
            "Startup Cost": 0.00,
            "Total Cost": 1000.00,
            "Plan Rows": 10000,
            "Plan Width": 25,
            "Actual Startup Time": 0.001,
            "Actual Total Time": 2.000,
            "Actual Rows": 8000,
            "Actual Loops": 1
          },
          {
            "Node Type": "Hash",
            "Parent Relationship": "Inner",
            "Parallel Aware": false,
            "Async Capable": false,
            "Startup Cost": 500.00,
            "Total Cost": 500.00,
            "Plan Rows": 100,
            "Plan Width": 25,
            "Actual Startup Time": 0.500,
            "Actual Total Time": 0.500,
            "Actual Rows": 200,
            "Actual Loops": 1,
            "Plans": [
              {
                "Node Type": "Index Scan",
                "Parent Relationship": "Outer",
                "Parallel Aware": false,
                "Async Capable": false,
                "Scan Direction": "Forward",
                "Index Name": "idx_movie_info_movie_id",
                "Relation Name": "movie_info",
                "Alias": "mi",
                "Startup Cost": 0.29,
                "Total Cost": 400.00,
                "Plan Rows": 100,
                "Plan Width": 25,
                "Actual Startup Time": 0.100,
                "Actual Total Time": 1.000,
                "Actual Rows": 200,
                "Actual Loops": 1
              }
            ]
          }
        ]
      },
      "Planning Time": 0.5,
      "Triggers": [],
      "Execution Time": 10.0
    }]
    '''
    
    stats = runner.extract_plan_stats(sample_explain)
    
    print(f"Extracted {len(stats)} operator stats:")
    for i, stat in enumerate(stats, 1):
        rel = stat['relation'] if stat['relation'] else 'N/A'
        print(f"  {i}. {stat['operator']} on {rel}: qe={stat['q_error']:.2f}")
    
    # 验证提取结果（使用集合比较，因为遍历顺序可能不同）
    expected_ops = {
        ("Seq Scan", "title", 10000, 8000, 1.25),  # 10000/8000 = 1.25
        ("Index Scan", "movie_info", 100, 200, 2.0),  # 200/100 = 2.0
        ("Hash", None, 100, 200, 2.0),  # 200/100 = 2.0
        ("Hash Join", None, 1000, 500, 2.0),  # 1000/500 = 2.0
    }
    
    if len(stats) != len(expected_ops):
        print(f"✗ Expected {len(expected_ops)} operators, got {len(stats)}")
        return False
    
    print("\nValidation:")
    all_passed = True
    
    # 将 stats 转换为可比较的集合
    actual_ops = set()
    for stat in stats:
        rel = stat['relation'] if stat['relation'] else None
        actual_ops.add((
            stat['operator'],
            rel,
            stat['estimated_rows'],
            stat['actual_rows'],
            round(stat['q_error'], 2)
        ))
    
    for expected in expected_ops:
        op_type, relation, est, actual, expected_qe = expected
        
        found = False
        for actual_op in actual_ops:
            (a_op_type, a_relation, a_est, a_actual, a_qe) = actual_op
            
            if (a_op_type == op_type and 
                a_relation == relation and
                a_est == est and 
                a_actual == actual and
                abs(a_qe - expected_qe) < 0.01):
                found = True
                print(f"✓ Found {op_type}: est={est}, actual={actual}, q-error={a_qe}")
                break
        
        if not found:
            print(f"✗ Missing or mismatch: {expected}")
            all_passed = False
    
    print("="*60)
    print(f"Test {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_real_postgres_format():
    """测试真实 PostgreSQL 输出格式（从实际数据库捕获）"""
    runner = PostgreSQLExperimentRunner.__new__(PostgreSQLExperimentRunner)
    
    print("\nTesting real PostgreSQL EXPLAIN format...")
    print("="*60)
    
    # 这是从真实 PostgreSQL 14 捕获的 EXPLAIN (ANALYZE, FORMAT JSON) 输出
    # 查询: SELECT COUNT(*) FROM title WHERE production_year > 2000
    real_postgres_output = '''
    [{
      "Plan": {
        "Node Type": "Aggregate",
        "Strategy": "Plain",
        "Partial Mode": "Simple",
        "Parallel Aware": false,
        "Async Capable": false,
        "Startup Cost": 0.00,
        "Total Cost": 31092.18,
        "Plan Rows": 1,
        "Plan Width": 8,
        "Actual Startup Time": 125.234,
        "Actual Total Time": 125.235,
        "Actual Rows": 1,
        "Actual Loops": 1,
        "Plans": [{
          "Node Type": "Seq Scan",
          "Parent Relationship": "Outer",
          "Parallel Aware": false,
          "Async Capable": false,
          "Relation Name": "title",
          "Alias": "title",
          "Startup Cost": 0.00,
          "Total Cost": 28387.10,
          "Plan Rows": 1082031,
          "Plan Width": 0,
          "Actual Startup Time": 0.023,
          "Actual Total Time": 98.456,
          "Actual Rows": 1150234,
          "Actual Loops": 1,
          "Filter": "(production_year > 2000)",
          "Rows Removed by Filter": 1371376
        }]
      },
      "Planning Time": 0.123,
      "Triggers": [],
      "Execution Time": 125.345
    }]
    '''
    
    stats = runner.extract_plan_stats(real_postgres_output)
    
    print(f"Extracted {len(stats)} operators from real PostgreSQL output")
    
    # 期望提取到 Seq Scan
    seq_scans = [s for s in stats if s['operator'] == 'Seq Scan']
    
    if not seq_scans:
        print("✗ No Seq Scan found")
        return False
    
    scan = seq_scans[0]
    print(f"✓ Found Seq Scan on {scan['relation']}")
    print(f"  Estimated: {scan['estimated_rows']:,}")
    print(f"  Actual:    {scan['actual_rows']:,}")
    print(f"  Q-Error:   {scan['q_error']:.2f}")
    
    # 验证数值
    # 从 JSON 中: Plan Rows = 1082031, Actual Rows = 1150234
    expected_est = 1082031
    expected_act = 1150234
    expected_qe = max(expected_est/expected_act, expected_act/expected_est)
    
    checks = [
        (scan['estimated_rows'] == expected_est, f"Est rows: {scan['estimated_rows']} == {expected_est}"),
        (scan['actual_rows'] == expected_act, f"Actual rows: {scan['actual_rows']} == {expected_act}"),
        (abs(scan['q_error'] - expected_qe) < 0.01, f"Q-Error: {scan['q_error']:.2f} ≈ {expected_qe:.2f}"),
    ]
    
    all_passed = True
    for passed, desc in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {desc}")
        if not passed:
            all_passed = False
    
    print("="*60)
    print(f"Test {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_index_scan_format():
    """测试 Index Scan 格式"""
    runner = PostgreSQLExperimentRunner.__new__(PostgreSQLExperimentRunner)
    
    print("\nTesting Index Scan format...")
    print("="*60)
    
    # Index Scan 的真实格式
    index_scan_output = '''
    [{
      "Plan": {
        "Node Type": "Index Scan",
        "Parallel Aware": false,
        "Async Capable": false,
        "Scan Direction": "Forward",
        "Index Name": "title_pkey",
        "Relation Name": "title",
        "Alias": "t",
        "Startup Cost": 0.42,
        "Total Cost": 8.44,
        "Plan Rows": 1,
        "Plan Width": 94,
        "Actual Startup Time": 0.008,
        "Actual Total Time": 0.009,
        "Actual Rows": 1,
        "Actual Loops": 1,
        "Index Cond": "(id = 12345)"
      },
      "Planning Time": 0.5,
      "Execution Time": 0.1
    }]
    '''
    
    stats = runner.extract_plan_stats(index_scan_output)
    
    if not stats:
        print("✗ No stats extracted")
        return False
    
    stat = stats[0]
    print(f"✓ Extracted Index Scan on {stat['relation']}")
    print(f"  Estimated: {stat['estimated_rows']}, Actual: {stat['actual_rows']}")
    print(f"  Q-Error: {stat['q_error']:.2f}")
    
    # Index Scan 应该在目标算子列表中
    if stat['operator'] != 'Index Scan':
        print(f"✗ Expected Index Scan, got {stat['operator']}")
        return False
    
    print("="*60)
    print("Test PASSED")
    return True


def test_nested_loop_format():
    """测试 Nested Loop Join 格式"""
    runner = PostgreSQLExperimentRunner.__new__(PostgreSQLExperimentRunner)
    
    print("\nTesting Nested Loop Join format...")
    print("="*60)
    
    nested_loop_output = '''
    [{
      "Plan": {
        "Node Type": "Nested Loop",
        "Parallel Aware": false,
        "Async Capable": false,
        "Join Type": "Inner",
        "Startup Cost": 0.84,
        "Total Cost": 1234.56,
        "Plan Rows": 100,
        "Plan Width": 100,
        "Actual Startup Time": 0.1,
        "Actual Total Time": 50.5,
        "Actual Rows": 250,
        "Actual Loops": 1,
        "Plans": [
          {
            "Node Type": "Index Scan",
            "Parent Relationship": "Outer",
            "Relation Name": "title",
            "Plan Rows": 10,
            "Actual Rows": 10
          },
          {
            "Node Type": "Index Scan",
            "Parent Relationship": "Inner",
            "Relation Name": "movie_info",
            "Plan Rows": 10,
            "Actual Rows": 25
          }
        ]
      },
      "Execution Time": 50.6
    }]
    '''
    
    stats = runner.extract_plan_stats(nested_loop_output)
    
    # 应该提取到 Nested Loop + 2 个 Index Scan
    if len(stats) != 3:
        print(f"✗ Expected 3 operators, got {len(stats)}")
        return False
    
    nested_loops = [s for s in stats if s['operator'] == 'Nested Loop']
    if not nested_loops:
        print("✗ No Nested Loop found")
        return False
    
    nl = nested_loops[0]
    print(f"✓ Extracted Nested Loop Join")
    print(f"  Estimated: {nl['estimated_rows']}, Actual: {nl['actual_rows']}")
    print(f"  Q-Error: {nl['q_error']:.2f}")
    
    # Q-Error 应该是 250/100 = 2.5
    expected_qe = 2.5
    if abs(nl['q_error'] - expected_qe) > 0.01:
        print(f"✗ Q-Error mismatch: {nl['q_error']:.2f} != {expected_qe}")
        return False
    
    print("="*60)
    print("Test PASSED")
    return True


def test_geometric_mean():
    """测试几何平均计算"""
    runner = PostgreSQLExperimentRunner.__new__(PostgreSQLExperimentRunner)
    
    print("\nTesting geometric mean calculation...")
    print("="*60)
    
    test_cases = [
        ([1.0, 1.0, 1.0], 1.0, "All 1s"),
        ([2.0, 2.0, 2.0], 2.0, "All 2s"),
        ([1.0, 4.0], 2.0, "1 and 4 -> 2"),
        ([2.0, 8.0], 4.0, "2 and 8 -> 4"),
        ([1.0, 2.0, 4.0, 8.0], 2.83, "Multiple values"),
        ([], 1.0, "Empty list"),
    ]
    
    all_passed = True
    for values, expected, desc in test_cases:
        result = runner.geometric_mean(values)
        passed = abs(result - expected) < 0.1
        status = "✓" if passed else "✗"
        print(f"{status} {desc}: {values} -> {result:.2f} (expected {expected})")
        if not passed:
            all_passed = False
    
    print("="*60)
    print(f"Test {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def main():
    print("PostgreSQL Q-Error Extraction Test Suite")
    print("="*60)
    
    results = []
    results.append(("Q-Error Calculation", test_qerror_calculation()))
    results.append(("Simple EXPLAIN Parsing", test_explain_parsing_simple()))
    results.append(("Join Query Parsing", test_explain_parsing_join()))
    results.append(("Real PostgreSQL Format", test_real_postgres_format()))
    results.append(("Index Scan Format", test_index_scan_format()))
    results.append(("Nested Loop Format", test_nested_loop_format()))
    results.append(("Geometric Mean", test_geometric_mean()))
    
    print("\n" + "="*60)
    print("Overall Test Results:")
    print("="*60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    print("="*60)
    print(f"All tests {'PASSED' if all_passed else 'FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
