#!/usr/bin/env python3
"""
提取 Presto EXPLAIN ANALYZE 的 Q-error 指标

Q-error = max(estimated / actual, actual / estimated)

对比场景：
1. Before ANALYZE: 使用过期统计信息（有漂移）
2. After ANALYZE: 使用新鲜统计信息（重新 ANALYZE）
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import prestodb


class QErrorExtractor:
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

    def set_session_properties(self, use_histograms: bool = True):
        """设置 session 属性"""
        if use_histograms:
            self.cursor.execute("SET SESSION optimizer_use_histograms = true")
        else:
            self.cursor.execute("SET SESSION optimizer_use_histograms = false")

    def extract_plan_stats(self, explain_output: str) -> List[Dict]:
        """
        从 EXPLAIN ANALYZE 输出中提取统计信息

        Presto 的 EXPLAIN ANALYZE 输出格式示例：
        - InnerJoin[...]
          Estimates: {rows: 1000 (100kB), cpu: ?, memory: ?, network: ?}
          CPU: 10ms, Scheduled: 20ms, Input: 950 rows (95kB); Output: 800 rows (80kB)

        我们需要提取：
        - Operator type (e.g., InnerJoin, TableScan, Filter)
        - Estimated rows (from "Estimates: {rows: X")
        - Actual rows (from "Output: X rows")
        """
        stats = []
        lines = explain_output.split('\n')

        current_operator = None
        estimated_rows = None
        actual_rows = None

        def flush_operator():
            nonlocal current_operator, estimated_rows, actual_rows
            if current_operator and estimated_rows is not None and actual_rows is not None:
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

            # 匹配估算行数（Estimates: {rows: 1000 (100kB), ...}）
            # 支持逗号分隔的数字；跳过 "rows: ?"
            if 'Estimates:' in line:
                est_match = re.search(r'rows:\s*([\d,]+\.?\d*[KMB]?)\b', line)
                if est_match:
                    estimated_rows = self.parse_row_count(est_match.group(1))

            # 匹配实际行数（Output: 800 rows (80kB)）
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
        """
        计算 Q-error

        Q-error = max(estimated / actual, actual / estimated)

        特殊情况：
        - 如果 actual = 0 且 estimated = 0: Q-error = 1.0 (完美估算)
        - 如果 actual = 0 且 estimated > 0: Q-error = estimated + 1 (避免除零)
        - 如果 estimated = 0 且 actual > 0: Q-error = actual + 1 (避免除零)
        """
        if actual == 0 and estimated == 0:
            return 1.0
        elif actual == 0:
            return estimated + 1
        elif estimated == 0:
            return actual + 1
        else:
            return max(estimated / actual, actual / estimated)

    def run_explain_analyze(self, query_id: str, sql: str, timeout: int = 300) -> Optional[str]:
        """运行 EXPLAIN ANALYZE 并返回输出"""
        try:
            explain_sql = f"EXPLAIN ANALYZE {sql}"
            self.cursor.execute(explain_sql)

            # 获取所有行并拼接
            rows = self.cursor.fetchall()
            explain_output = '\n'.join([row[0] for row in rows])

            return explain_output

        except Exception as e:
            print(f"  ✗ Error running EXPLAIN ANALYZE for {query_id}: {e}")
            return None

    def analyze_query(self, query_id: str, sql: str) -> Optional[Dict]:
        """分析单个查询的 Q-error"""
        print(f"  Analyzing {query_id}...", end=' ')

        explain_output = self.run_explain_analyze(query_id, sql)
        if not explain_output:
            print("✗ Failed")
            return None

        stats = self.extract_plan_stats(explain_output)

        if not stats:
            print("✗ No stats extracted")
            return None

        # 计算汇总指标
        q_errors = [s['q_error'] for s in stats]
        result = {
            'query_id': query_id,
            'operator_stats': stats,
            'summary': {
                'num_operators': len(stats),
                'mean_q_error': sum(q_errors) / len(q_errors),
                'median_q_error': sorted(q_errors)[len(q_errors) // 2],
                'max_q_error': max(q_errors),
                'min_q_error': min(q_errors),
                'geometric_mean_q_error': self.geometric_mean(q_errors)
            }
        }

        print(f"✓ Mean Q-error: {result['summary']['mean_q_error']:.2f}, Max: {result['summary']['max_q_error']:.2f}")
        return result

    def geometric_mean(self, values: List[float]) -> float:
        """计算几何平均数"""
        if not values:
            return 0.0
        product = 1.0
        for v in values:
            product *= v
        return product ** (1.0 / len(values))

    def run_experiment(self, query_dir: Path, output_dir: Path, use_histograms: bool = True):
        """运行完整的 Q-error 提取实验"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 设置 session 属性
        print(f"\nSession configuration:")
        self.set_session_properties(use_histograms)
        print(f"  optimizer_use_histograms = {use_histograms}")

        # 加载查询
        queries = self.load_queries(query_dir)
        print(f"\nLoaded {len(queries)} queries")

        # 分析每个查询
        results = []
        for i, (query_id, sql) in enumerate(queries.items(), 1):
            print(f"[{i}/{len(queries)}]", end=' ')
            result = self.analyze_query(query_id, sql)
            if result:
                results.append(result)

        # 保存结果
        output_file = output_dir / 'qerror_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'use_histograms': use_histograms,
                'total_queries': len(queries),
                'successful_queries': len(results),
                'results': results
            }, f, indent=2)

        print(f"\n✓ Results saved to {output_file}")

        # 打印汇总
        self.print_summary(results)

    def load_queries(self, query_dir: Path) -> Dict[str, str]:
        """加载所有 SQL 查询"""
        queries = {}
        for sql_file in sorted(query_dir.glob('*.sql')):
            query_id = sql_file.stem
            with open(sql_file, 'r') as f:
                queries[query_id] = f.read()
        return queries

    def print_summary(self, results: List[Dict]):
        """打印 Q-error 汇总统计"""
        if not results:
            print("\n⚠️  No results to summarize")
            return

        mean_q_errors = [r['summary']['mean_q_error'] for r in results]
        max_q_errors = [r['summary']['max_q_error'] for r in results]
        geom_mean_q_errors = [r['summary']['geometric_mean_q_error'] for r in results]

        print("\n" + "="*70)
        print("Q-error Summary:")
        print(f"  Successful queries: {len(results)}")
        print(f"\n  Mean Q-error across queries:")
        print(f"    Average: {sum(mean_q_errors) / len(mean_q_errors):.2f}")
        print(f"    Median: {sorted(mean_q_errors)[len(mean_q_errors) // 2]:.2f}")
        print(f"    Max: {max(mean_q_errors):.2f}")
        print(f"\n  Max Q-error across queries:")
        print(f"    Average: {sum(max_q_errors) / len(max_q_errors):.2f}")
        print(f"    Median: {sorted(max_q_errors)[len(max_q_errors) // 2]:.2f}")
        print(f"    Max: {max(max_q_errors):.2f}")
        print(f"\n  Geometric mean Q-error:")
        print(f"    Average: {sum(geom_mean_q_errors) / len(geom_mean_q_errors):.2f}")
        print("="*70)

    def close(self):
        self.cursor.close()
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description='Extract Q-error from Presto EXPLAIN ANALYZE')
    parser.add_argument('--presto-host', required=True, help='Presto host:port')
    parser.add_argument('--catalog', default='iceberg', help='Catalog name')
    parser.add_argument('--schema', default='imdb', help='Schema name')
    parser.add_argument('--user', default='tianqc', help='Presto user')
    parser.add_argument('--query-dir', required=True, help='Directory containing SQL queries')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--use-histograms', action='store_true', default=True,
                       help='Enable optimizer_use_histograms (default: True)')
    parser.add_argument('--no-histograms', action='store_true',
                       help='Disable optimizer_use_histograms')

    args = parser.parse_args()

    # 解析 host:port
    host, port = args.presto_host.split(':')

    # 创建 extractor
    extractor = QErrorExtractor(host, int(port), args.catalog, args.schema, user=args.user)

    try:
        extractor.run_experiment(
            query_dir=Path(args.query_dir),
            output_dir=Path(args.output_dir),
            use_histograms=args.use_histograms and not args.no_histograms
        )
    finally:
        extractor.close()


if __name__ == '__main__':
    main()
