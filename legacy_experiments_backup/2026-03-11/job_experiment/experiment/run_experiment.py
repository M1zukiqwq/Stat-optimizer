#!/usr/bin/env python3
"""
JOB Benchmark 实验运行器（合并版）

一次运行同时收集：
1. 查询执行时间（通过 EXPLAIN ANALYZE 的实际执行）
2. Q-error 指标（通过 EXPLAIN ANALYZE 的估算 vs 实际行数）

支持多种策略：
- baseline: 无漂移，fresh statistics
- stale_prior: 有漂移，不修正
- oasis: 有漂移，OASIS 修正
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

import prestodb


class ExperimentRunner:
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

    def set_session_properties(self, strategy: str, enable_correction: bool, use_histograms: bool = True):
        """设置 session 属性"""
        if use_histograms:
            self.cursor.execute("SET SESSION optimizer_use_histograms = true")
            print("  ✓ Enabled optimizer_use_histograms")
        else:
            self.cursor.execute("SET SESSION optimizer_use_histograms = false")
            print("  ✗ Disabled optimizer_use_histograms")

        if strategy == 'warmup' or enable_correction:
            self.cursor.execute("SET SESSION iceberg.ml_feedback_enabled = true")
            self.cursor.execute("SET SESSION iceberg.ml_feedback_output_dir = '/tmp/ml-feedback'")

        if enable_correction:
            self.cursor.execute("SET SESSION iceberg.oasis_model_correction_enabled = true")
            print("  ✓ Enabled OASIS model correction")
        else:
            self.cursor.execute("SET SESSION iceberg.oasis_model_correction_enabled = false")

    # ==================== Q-error 提取 ====================

    def extract_plan_stats(self, explain_output: str) -> List[Dict]:
        """从 EXPLAIN ANALYZE 输出中提取算子级别的统计信息（仅 filter 和 join 算子）"""
        stats = []
        lines = explain_output.split('\n')

        # 只关注 filter 和 join 相关的算子
        target_operators = {
            'Filter', 'ScanFilter', 'ScanFilterProject',
            'Join', 'HashJoin', 'NestedLoopJoin', 'SemiJoin',
            'CrossJoin', 'LookupJoin', 'MergeJoin'
        }

        current_operator = None
        estimated_rows = None
        actual_rows = None

        def flush_operator():
            nonlocal current_operator, estimated_rows, actual_rows
            # 只保存目标算子的统计信息
            if (current_operator and
                current_operator in target_operators and
                estimated_rows is not None and
                actual_rows is not None):
                stats.append({
                    'operator': current_operator,
                    'estimated_rows': estimated_rows,
                    'actual_rows': actual_rows,
                    'q_error': self.calculate_q_error(estimated_rows, actual_rows)
                })

        for line in lines:
            # 匹配算子：支持 "- Operator[" 和 "- Operator" (无括号)
            operator_match = re.search(r'^\s*-\s+(\w+)[\[\s]', line) or \
                             re.search(r'^\s*-\s+(\w+)\s*$', line)
            if operator_match:
                flush_operator()
                current_operator = operator_match.group(1)
                estimated_rows = None
                actual_rows = None

            if 'Estimates:' in line:
                # 支持逗号分隔的数字和 K/M/B 后缀；跳过 "rows: ?"
                est_match = re.search(r'rows:\s*([\d,]+\.?\d*[KMB]?)\b', line)
                if est_match:
                    estimated_rows = self.parse_row_count(est_match.group(1))

            if 'Output:' in line:
                out_match = re.search(r'Output:\s*([\d,]+\.?\d*[KMB]?)\s*rows', line)
                if out_match:
                    actual_rows = self.parse_row_count(out_match.group(1))

        flush_operator()

        return stats

    def parse_row_count(self, count_str: str) -> float:
        """解析行数字符串（支持 K/M/B 后缀和逗号分隔）"""
        count_str = count_str.strip().replace(',', '')
        multiplier = 1
        if count_str.endswith('K'):
            multiplier = 1000
            count_str = count_str[:-1]
        elif count_str.endswith('M'):
            multiplier = 1000000
            count_str = count_str[:-1]
        elif count_str.endswith('B'):
            multiplier = 1000000000
            count_str = count_str[:-1]
        return float(count_str) * multiplier

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

    # ==================== 查询执行 ====================

    def run_query(self, query_id: str, sql: str, collect_qerror: bool = True) -> Dict:
        """
        运行单个查询，同时收集执行时间和 Q-error

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
                    # 诊断：为什么没有提取到 Q-error
                    has_unknown_est = '?' in explain_output and 'rows: ?' in explain_output
                    print(f"\n  ⚠ No Q-error extracted for {query_id}. "
                          f"Unknown estimates: {has_unknown_est}. "
                          f"EXPLAIN output length: {len(explain_output)} chars")

                # 从 EXPLAIN ANALYZE 输出中提取最终输出行数
                final_output = re.search(r'Output:\s*([\d.]+[KMB]?)\s*rows', explain_output)
                if final_output:
                    result['rows_returned'] = int(self.parse_row_count(final_output.group(1)))
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

    # ==================== 实验运行 ====================

    def run_experiment(self, query_dir: Path, strategy: str, output_dir: Path,
                      enable_correction: bool, use_histograms: bool = True,
                      warmup_only: bool = False, warmup_count: int = 20,
                      collect_qerror: bool = True):
        """运行完整实验"""
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSession configuration:")
        self.set_session_properties(strategy, enable_correction, use_histograms)

        queries = self.load_queries(query_dir)
        print(f"Loaded {len(queries)} queries")
        print(f"Q-error collection: {'enabled' if collect_qerror else 'disabled'}")

        if warmup_only:
            queries = dict(list(queries.items())[:warmup_count])
            print(f"Warmup mode: running first {len(queries)} queries")

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
            'use_histograms': use_histograms,
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
    parser = argparse.ArgumentParser(description='Run JOB benchmark experiment (execution time + Q-error)')
    parser.add_argument('--presto-host', required=True, help='Presto host:port')
    parser.add_argument('--catalog', default='iceberg', help='Catalog name')
    parser.add_argument('--schema', default='imdb', help='Schema name')
    parser.add_argument('--user', default='tianqc', help='Presto user (default: tianqc)')
    parser.add_argument('--query-dir', required=True, help='Directory containing SQL queries')
    parser.add_argument('--strategy', required=True,
                       choices=['baseline', 'stale_prior', 'oasis', 'full_analyze', 'warmup'],
                       help='Experiment strategy')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--enable-correction', action='store_true',
                       help='Enable OASIS model correction')
    parser.add_argument('--no-correction', action='store_true',
                       help='Disable OASIS model correction')
    parser.add_argument('--use-histograms', action='store_true', default=True,
                       help='Enable optimizer_use_histograms (default: True)')
    parser.add_argument('--no-histograms', action='store_true',
                       help='Disable optimizer_use_histograms')
    parser.add_argument('--warmup-only', action='store_true',
                       help='Only run warmup queries')
    parser.add_argument('--warmup-count', type=int, default=20,
                       help='Number of warmup queries')
    parser.add_argument('--no-qerror', action='store_true',
                       help='Disable Q-error collection (use plain execution instead of EXPLAIN ANALYZE)')

    args = parser.parse_args()

    host, port = args.presto_host.split(':')

    runner = ExperimentRunner(host, int(port), args.catalog, args.schema, user=args.user)

    try:
        runner.run_experiment(
            query_dir=Path(args.query_dir),
            strategy=args.strategy,
            output_dir=Path(args.output_dir),
            enable_correction=args.enable_correction and not args.no_correction,
            use_histograms=args.use_histograms and not args.no_histograms,
            warmup_only=args.warmup_only,
            warmup_count=args.warmup_count,
            collect_qerror=not args.no_qerror
        )
    finally:
        runner.close()


if __name__ == '__main__':
    main()
