#!/usr/bin/env python3
"""
PostgreSQL JOB Benchmark 实验运行器

功能：
1. 查询执行时间收集（通过 EXPLAIN ANALYZE）
2. Q-error 指标收集（通过 EXPLAIN ANALYZE 的估算 vs 实际行数）

支持策略：
- baseline: 无漂移，fresh statistics
- stale_prior: 有漂移，不修正
- full_analyze: 有漂移后重新 ANALYZE
"""

import argparse
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor


class PostgresExperimentRunner:
    def __init__(self, db_name: str, host: str = 'localhost', port: int = 5432, user: str = None):
        """初始化 PostgreSQL 连接"""
        import os
        self.conn = psycopg2.connect(
            dbname=db_name,
            host=host,
            port=port,
            user=user or os.getenv('USER')
        )
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()

    def extract_plan_stats(self, explain_output: str) -> List[Dict]:
        """
        从 EXPLAIN ANALYZE 输出中提取算子级别的统计信息

        PostgreSQL EXPLAIN ANALYZE 格式示例：
        Seq Scan on title  (cost=0.00..12345.00 rows=1000 width=50) (actual time=0.123..45.678 rows=987 loops=1)
        Hash Join  (cost=100.00..5000.00 rows=500 width=100) (actual time=1.234..56.789 rows=456 loops=1)
        """
        stats = []

        # 目标算子：Filter 和 Join 相关
        target_operators = {
            'Seq Scan', 'Index Scan', 'Index Only Scan', 'Bitmap Heap Scan',
            'Hash Join', 'Nested Loop', 'Merge Join',
            'Filter'
        }

        lines = explain_output.split('\n')

        for line in lines:
            # 匹配算子行：包含 (cost=... rows=...) (actual ... rows=...)
            # 示例：Seq Scan on title  (cost=0.00..12345.00 rows=1000 width=50) (actual time=0.123..45.678 rows=987 loops=1)

            # 提取算子名称
            operator_match = None
            for op in target_operators:
                if op in line:
                    operator_match = op
                    break

            if not operator_match:
                continue

            # 提取估算行数：rows=1000
            est_match = re.search(r'\(cost=[\d.]+\.\.[\d.]+\s+rows=(\d+)', line)
            # 提取实际行数：actual ... rows=987
            act_match = re.search(r'actual\s+time=[\d.]+\.\.[\d.]+\s+rows=(\d+)', line)

            if est_match and act_match:
                estimated_rows = float(est_match.group(1))
                actual_rows = float(act_match.group(1))

                stats.append({
                    'operator': operator_match,
                    'estimated_rows': estimated_rows,
                    'actual_rows': actual_rows,
                    'q_error': self.calculate_q_error(estimated_rows, actual_rows)
                })

        return stats

    def calculate_q_error(self, estimated: float, actual: float) -> float:
        """Q-error = max(estimated / actual, actual / estimated)"""
        if actual == 0 and estimated == 0:
            return 1.0
        elif actual == 0:
            return estimated + 1
        elif estimated == 0:
            return actual + 1
        else:
            return max(estimated / actual, actual / estimated)

    def geometric_mean(self, values: List[float]) -> float:
        """计算几何平均数"""
        if not values:
            return 0.0
        log_sum = sum(math.log(v) for v in values if v > 0)
        return math.exp(log_sum / len(values))

    def run_query(self, query_id: str, sql: str, collect_qerror: bool = True) -> Dict:
        """
        运行单个查询，收集执行时间和 Q-error

        使用 EXPLAIN ANALYZE 执行查询，一次获取：
        - 执行时间（通过计时）
        - Q-error（通过解析 EXPLAIN ANALYZE 输出）
        """
        result = {
            'query_id': query_id,
            'status': 'unknown',
            'execution_time_ms': None,
            'rows_returned': None,
            'error': None,
            'qerror': None
        }

        try:
            if collect_qerror:
                # 使用 EXPLAIN ANALYZE：同时执行查询并获取计划统计
                explain_sql = f"EXPLAIN ANALYZE {sql}"
                start_time = time.time()
                self.cursor.execute(explain_sql)
                rows = self.cursor.fetchall()
                end_time = time.time()

                explain_output = '\n'.join([row[0] for row in rows])

                result['status'] = 'success'
                result['execution_time_ms'] = int((end_time - start_time) * 1000)

                # 提取 Q-error
                plan_stats = self.extract_plan_stats(explain_output)
                if plan_stats:
                    q_errors = [s['q_error'] for s in plan_stats]
                    result['qerror'] = {
                        'num_operators': len(plan_stats),
                        'mean': sum(q_errors) / len(q_errors),
                        'median': sorted(q_errors)[len(q_errors) // 2],
                        'max': max(q_errors),
                        'min': min(q_errors),
                        'geometric_mean': self.geometric_mean(q_errors),
                        'operator_details': plan_stats
                    }
                else:
                    print(f"\n  ⚠ No Q-error extracted for {query_id}. "
                          f"EXPLAIN output length: {len(explain_output)} chars")

                # 从 EXPLAIN ANALYZE 输出中提取最终输出行数
                # 最后一行通常是：Planning Time: X ms / Execution Time: Y ms
                # 倒数第二行可能是总行数
                final_output = re.search(r'rows=(\d+)', explain_output)
                if final_output:
                    result['rows_returned'] = int(final_output.group(1))
            else:
                # 普通执行模式（不收集 Q-error）
                start_time = time.time()
                self.cursor.execute(sql)
                rows = self.cursor.fetchall()
                end_time = time.time()

                result['status'] = 'success'
                result['execution_time_ms'] = int((end_time - start_time) * 1000)
                result['rows_returned'] = len(rows)

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def run_experiment(self, query_dir: Path, strategy: str, output_dir: Path,
                      collect_qerror: bool = True):
        """运行完整实验"""
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRunning {strategy} experiment on PostgreSQL")
        print(f"Q-error collection: {'enabled' if collect_qerror else 'disabled'}")

        queries = self.load_queries(query_dir)
        print(f"Loaded {len(queries)} queries")

        # 运行查询
        results = []
        for i, (query_id, sql) in enumerate(queries.items(), 1):
            print(f"[{i}/{len(queries)}] Running {query_id}...", end=' ')

            result = self.run_query(query_id, sql, collect_qerror=collect_qerror)
            results.append(result)

            if result['status'] == 'success':
                time_str = f"{result['execution_time_ms']}ms"
                if result['qerror']:
                    qe = result['qerror']
                    print(f"✓ {time_str} | Q-error mean={qe['mean']:.2f} max={qe['max']:.2f}")
                else:
                    print(f"✓ {time_str}")
            else:
                print(f"✗ {result['error']}")

        # 保存结果
        output_file = output_dir / f"{strategy}_results.json"
        output_data = {
            'strategy': strategy,
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(queries),
            'collect_qerror': collect_qerror,
            'database': 'postgresql',
            'results': results
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to {output_file}")

        # 打印摘要
        self.print_summary(results, collect_qerror)

    def load_queries(self, query_dir: Path) -> Dict[str, str]:
        """加载所有 SQL 查询"""
        queries = {}
        for sql_file in sorted(query_dir.glob('*.sql')):
            query_id = sql_file.stem
            with open(sql_file, 'r') as f:
                queries[query_id] = f.read()
        return queries

    def print_summary(self, results: List[Dict], collect_qerror: bool = True):
        """打印结果摘要"""
        success_count = sum(1 for r in results if r['status'] == 'success')
        error_count = sum(1 for r in results if r['status'] == 'error')

        success_times = [r['execution_time_ms'] for r in results if r['status'] == 'success']
        total_time = sum(success_times) if success_times else 0
        avg_time = total_time / len(success_times) if success_times else 0

        print("\n" + "="*70)
        print("Execution Summary:")
        print(f"  Success: {success_count}/{len(results)}")
        print(f"  Error: {error_count}/{len(results)}")
        print(f"  Total time: {total_time/1000:.1f}s")
        print(f"  Avg time: {avg_time:.0f}ms")

        if collect_qerror:
            # Q-error 汇总
            qerror_results = [r for r in results if r['status'] == 'success' and r.get('qerror')]
            if qerror_results:
                mean_qerrors = [r['qerror']['mean'] for r in qerror_results]
                max_qerrors = [r['qerror']['max'] for r in qerror_results]
                geom_qerrors = [r['qerror']['geometric_mean'] for r in qerror_results]

                print(f"\nQ-error Summary ({len(qerror_results)} queries):")
                print(f"  Mean Q-error:")
                print(f"    Average: {sum(mean_qerrors) / len(mean_qerrors):.2f}")
                print(f"    Median:  {sorted(mean_qerrors)[len(mean_qerrors) // 2]:.2f}")
                print(f"    Max:     {max(mean_qerrors):.2f}")
                print(f"  Max Q-error:")
                print(f"    Average: {sum(max_qerrors) / len(max_qerrors):.2f}")
                print(f"    Max:     {max(max_qerrors):.2f}")
                print(f"  Geometric mean Q-error:")
                print(f"    Average: {sum(geom_qerrors) / len(geom_qerrors):.2f}")

        print("="*70)

    def close(self):
        self.cursor.close()
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description='Run JOB benchmark on PostgreSQL')
    parser.add_argument('--db-name', required=True, help='Database name')
    parser.add_argument('--host', default='localhost', help='PostgreSQL host')
    parser.add_argument('--port', type=int, default=5432, help='PostgreSQL port')
    parser.add_argument('--user', default=None, help='PostgreSQL user (default: current user)')
    parser.add_argument('--query-dir', required=True, help='Directory containing SQL queries')
    parser.add_argument('--strategy', required=True,
                       choices=['baseline', 'stale_prior', 'full_analyze'],
                       help='Experiment strategy')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--no-qerror', action='store_true',
                       help='Disable Q-error collection')

    args = parser.parse_args()

    runner = PostgresExperimentRunner(
        db_name=args.db_name,
        host=args.host,
        port=args.port,
        user=args.user
    )

    try:
        runner.run_experiment(
            query_dir=Path(args.query_dir),
            strategy=args.strategy,
            output_dir=Path(args.output_dir),
            collect_qerror=not args.no_qerror
        )
    finally:
        runner.close()


if __name__ == '__main__':
    main()
