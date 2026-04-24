#!/usr/bin/env python3
"""
PostgreSQL 简单端到端实验：Stale Prior vs Full ANALYZE

这是一个简化版本，用于快速验证实验流程：
1. Stale Prior: 注入漂移后，使用过期统计运行 JOB 查询
2. Full ANALYZE: 重新 ANALYZE 后，使用新鲜统计运行 JOB 查询
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


class PostgreSQLExperimentRunner:
    def __init__(self, host: str, port: int, dbname: str, user: str, password: str = ''):
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        self.conn.autocommit = True
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        self.dbname = dbname

    def get_table_stats(self, table_name: str) -> Dict:
        """获取表的统计信息"""
        self.cursor.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
        row_count = self.cursor.fetchone()['cnt']
        
        # 获取最后一次 ANALYZE 时间
        self.cursor.execute("""
            SELECT last_analyze, last_autoanalyze
            FROM pg_stat_user_tables
            WHERE relname = %s
        """, (table_name,))
        result = self.cursor.fetchone()
        last_analyze = result['last_analyze'] if result else None
        
        return {
            'row_count': row_count,
            'last_analyze': str(last_analyze) if last_analyze else 'Never'
        }

    def extract_plan_stats(self, explain_output) -> List[Dict]:
        """
        从 PostgreSQL EXPLAIN (ANALYZE, FORMAT JSON) 输出中提取 Q-error
        
        PostgreSQL JSON 格式:
        [
          {
            "Plan": {
              "Node Type": "...",
              "Parallel Aware": false,
              "Plan Rows": N,
              "Plan Width": N,
              "Actual Rows": N,
              "Actual Time": X..Y,
              "Plans": [ ... ]  # 子节点
            },
            "Planning Time": X,
            "Execution Time": Y
          }
        ]
        
        Args:
            explain_output: JSON 字符串或已解析的 Python 对象（list/dict）
        """
        stats = []
        
        try:
            # 处理两种情况：1) JSON 字符串 2) 已解析的 Python 对象
            if isinstance(explain_output, str):
                plan_data = json.loads(explain_output)
            else:
                plan_data = explain_output
            
            # PostgreSQL 返回的是数组
            if isinstance(plan_data, list) and len(plan_data) > 0:
                root = plan_data[0]
            elif isinstance(plan_data, dict):
                root = plan_data
            else:
                return stats
            
            # 获取 Plan 节点
            plan = root.get('Plan', root)
            
            def traverse_node(node, depth=0):
                """递归遍历计划树"""
                if not isinstance(node, dict):
                    return
                
                node_type = node.get('Node Type', 'Unknown')
                actual_rows = node.get('Actual Rows', 0)
                estimated_rows = node.get('Plan Rows', 0)
                
                # 关注的算子类型（基数估算相关的）
                target_operators = {
                    'Seq Scan', 'Index Scan', 'Index Only Scan', 'Bitmap Heap Scan',
                    'Nested Loop', 'Merge Join', 'Hash Join',
                    'Hash', 'Materialize', 'Sort', 'Gather', 'Gather Merge',
                    'Bitmap Index Scan'
                }
                
                # 扫描节点类型（可能受 LIMIT 影响）
                scan_operators = {
                    'Seq Scan', 'Index Scan', 'Index Only Scan', 'Bitmap Heap Scan',
                    'Bitmap Index Scan'
                }
                
                # 只记录目标算子且估算行数 > 0 的（避免根节点 Total Runtime）
                if node_type in target_operators and estimated_rows > 0:
                    relation = node.get('Relation Name', '')
                    index = node.get('Index Name', '')
                    
                    # 对于扫描节点，如果存在 "Rows Removed by Filter"，
                    # 说明 Actual Rows 可能被 LIMIT 截断，需要加上被过滤的行数
                    actual_rows_for_qe = actual_rows
                    if node_type in scan_operators:
                        rows_removed = node.get('Rows Removed by Filter', 0)
                        if rows_removed > 0:
                            actual_rows_for_qe = actual_rows + rows_removed
                    
                    # 计算 Q-error（使用修正后的实际行数）
                    q_error = self.calculate_q_error(estimated_rows, actual_rows_for_qe)
                    
                    stats.append({
                        'operator': node_type,
                        'estimated_rows': estimated_rows,
                        'actual_rows': actual_rows,
                        'actual_rows_for_qe': actual_rows_for_qe,
                        'rows_removed': node.get('Rows Removed by Filter', 0) if node_type in scan_operators else 0,
                        'q_error': q_error,
                        'relation': relation or index or None,
                        'depth': depth
                    })
                
                # 递归处理子节点
                for child in node.get('Plans', []):
                    traverse_node(child, depth + 1)
            
            traverse_node(plan)
            
        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON parse error: {e}")
        except Exception as e:
            print(f"  ⚠ Error parsing plan: {e}")
        
        return stats

    def calculate_q_error(self, estimated: float, actual: float) -> float:
        """
        计算 Q-error = max(estimated / actual, actual / estimated)
        
        Q-error 定义（来自论文）:
        - 如果 estimated == actual: q-error = 1 (完美估算)
        - 如果 estimated > actual: q-error = estimated / actual
        - 如果 actual > estimated: q-error = actual / estimated
        
        特殊情况处理:
        - actual == 0 且 estimated == 0: q-error = 1 (都为空，算正确)
        - actual == 0 但 estimated > 0: q-error = estimated + 1 (高估)
        - estimated == 0 但 actual > 0: q-error = actual + 1 (低估)
        """
        if actual == 0 and estimated == 0:
            return 1.0
        elif actual == 0:
            # 实际为空但估算有值，高估
            return max(estimated + 1, 2.0)
        elif estimated == 0:
            # 实际有值但估算为空，低估
            return max(actual + 1, 2.0)
        else:
            return max(estimated / actual, actual / estimated)

    def geometric_mean(self, values: List[float]) -> float:
        """计算几何平均数（用于聚合多个 q-error）"""
        if not values:
            return 1.0  # 空集合返回 1（表示完美估算）
        
        # 过滤掉无效值
        valid_values = [v for v in values if v > 0]
        if not valid_values:
            return 1.0
        
        log_sum = sum(math.log(v) for v in valid_values)
        return math.exp(log_sum / len(valid_values))

    def run_query(self, query_id: str, sql: str, timeout: int = 300) -> Dict:
        """运行单个查询，收集执行时间和 Q-error"""
        result = {
            'query_id': query_id,
            'status': 'unknown',
            'execution_time_ms': None,
            'rows_returned': None,
            'error': None,
            'qerror': None,
            'plan_raw': None
        }

        try:
            # 设置超时并禁用并行查询（statement_timeout 单位是毫秒）
            self.cursor.execute(f"SET statement_timeout = {timeout * 1000}")
            self.cursor.execute("SET max_parallel_workers_per_gather = 0")
            self.cursor.execute("SET lock_timeout = '5s'")
            
            # 使用 EXPLAIN (ANALYZE, FORMAT JSON) 获取执行时间和 Q-error
            # VERBOSE 提供更多细节，但这里用不上
            explain_sql = f"EXPLAIN (ANALYZE, FORMAT JSON, BUFFERS OFF) {sql}"
            start_time = time.time()
            self.cursor.execute(explain_sql)
            rows = self.cursor.fetchall()
            end_time = time.time()

            # PostgreSQL 返回的是字典数组，第一列可能是 JSON 字符串或已解析的 Python 对象
            if rows and len(rows) > 0:
                first_row = rows[0]
                if isinstance(first_row, dict):
                    # 使用 RealDictCursor 返回的是 dict
                    # RealDictRow 可能有 'QUERY PLAN' 键
                    if 'QUERY PLAN' in first_row:
                        explain_output = first_row['QUERY PLAN']
                    else:
                        explain_output = first_row
                elif hasattr(first_row, '__iter__') and not isinstance(first_row, str):
                    # 普通 cursor 返回的是 tuple，第一个元素可能是已解析的 JSON
                    explain_output = first_row[0] if len(first_row) > 0 else '[]'
                else:
                    explain_output = first_row if first_row else '[]'
            else:
                explain_output = '[]'

            result['status'] = 'success'
            # 先用 Python wall-clock 作为兜底
            result['execution_time_ms'] = int((end_time - start_time) * 1000)
            # 保存前2000字符用于调试
            if isinstance(explain_output, str):
                result['plan_raw'] = explain_output[:2000]
            else:
                result['plan_raw'] = str(explain_output)[:2000]

            # 提取 Q-error，以及 PG 内部精确执行时间
            plan_stats = self.extract_plan_stats(explain_output)
            if plan_stats:
                q_errors = [s['q_error'] for s in plan_stats]
                result['qerror'] = {
                    'num_operators': len(plan_stats),
                    'mean': sum(q_errors) / len(q_errors),
                    'median': sorted(q_errors)[len(q_errors) // 2] if q_errors else 1.0,
                    'max': max(q_errors) if q_errors else 1.0,
                    'min': min(q_errors) if q_errors else 1.0,
                    'geometric_mean': self.geometric_mean(q_errors),
                    'operator_details': plan_stats[:10]  # 只保存前10个算子的详情
                }
            
            # 提取总返回行数 + PG 内部 Execution Time（精确，不含网络开销）
            try:
                # explain_output 可能是已解析的 Python 对象
                if isinstance(explain_output, str):
                    plan_data = json.loads(explain_output)
                else:
                    plan_data = explain_output
                    
                if isinstance(plan_data, list) and len(plan_data) > 0:
                    root_plan = plan_data[0].get('Plan', {})
                    pg_exec_time = plan_data[0].get('Execution Time')  # 单位：ms（float）
                elif isinstance(plan_data, dict):
                    root_plan = plan_data.get('Plan', plan_data)
                    pg_exec_time = plan_data.get('Execution Time')
                else:
                    root_plan = {}
                    pg_exec_time = None
                result['rows_returned'] = root_plan.get('Actual Rows', 0)
                result['execution_time_total'] = pg_exec_time
                # 用 PG 内部时间覆盖 Python 计时（更精确）
                if pg_exec_time is not None:
                    result['execution_time_ms'] = int(pg_exec_time)
            except:
                pass

        except psycopg2.Error as e:
            err_msg = str(e.pgerror) if e.pgerror else str(e)
            # 检查是否是 statement_timeout
            if 'canceling statement due to statement timeout' in err_msg or \
               'statement timeout' in err_msg.lower():
                result['status'] = 'timeout'
                result['execution_time_ms'] = timeout * 1000
                result['error'] = f'timeout (>{timeout}s)'
                # 超时后需要重置连接状态
                try:
                    self.conn.rollback()
                except:
                    pass
            else:
                result['status'] = 'error'
                result['error'] = err_msg
                try:
                    self.conn.rollback()
                except:
                    pass
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            try:
                self.conn.rollback()
            except:
                pass

        return result

    def run_experiment(self, query_dir: Path, output_dir: Path, 
                       strategy: str, timeout: int = 300,
                       skip_queries: List[str] = None):
        """运行完整实验"""
        output_dir.mkdir(parents=True, exist_ok=True)
        skip_queries = skip_queries or []

        print(f"\nStrategy: {strategy}")
        print(f"Database: {self.dbname}")
        print(f"Timeout: {timeout}s")
        print("="*70)

        queries = self.load_queries(query_dir, skip_queries)
        print(f"Loaded {len(queries)} queries")

        # 打印当前统计状态
        print("\nCurrent table statistics:")
        for table in ['title', 'cast_info', 'movie_info', 'movie_keyword']:
            try:
                stats = self.get_table_stats(table)
                print(f"  {table}: {stats['row_count']:,} rows (ANALYZE: {stats['last_analyze']})")
            except Exception as e:
                pass
        print("="*70)

        # 运行查询
        results = []
        for i, (query_id, sql) in enumerate(queries.items(), 1):
            print(f"[{i}/{len(queries)}] Running {query_id}...", end=' ', flush=True)

            result = self.run_query(query_id, sql, timeout)
            results.append(result)

            if result['status'] == 'success':
                time_str = f"{result['execution_time_ms']}ms"
                if result['qerror']:
                    qe = result['qerror']
                    print(f"✓ {time_str} | ops={qe['num_operators']} q-error={qe['geometric_mean']:.2f}")
                else:
                    print(f"✓ {time_str} (no q-error data)")
            elif result['status'] == 'timeout':
                print(f"⏱ TIMEOUT (>{timeout}s)")
            else:
                error_msg = result['error'][:60] if result['error'] and len(result['error']) > 60 else result['error']
                print(f"✗ {error_msg}")

        # 保存结果
        output_file = output_dir / f"{strategy}_results.json"
        output_data = {
            'strategy': strategy,
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(queries),
            'results': results
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to {output_file}")
        self.print_summary(results)

    def load_queries(self, query_dir: Path, skip_queries: List[str]) -> Dict[str, str]:
        """加载所有 SQL 查询"""
        queries = {}
        for sql_file in sorted(query_dir.glob('*.sql')):
            query_id = sql_file.stem
            if query_id in skip_queries:
                print(f"  Skipping {query_id} (in skip list)")
                continue
            with open(sql_file, 'r') as f:
                queries[query_id] = f.read()
        return queries

    def print_summary(self, results: List[Dict]):
        """打印结果摘要"""
        success_count = sum(1 for r in results if r['status'] == 'success')
        timeout_count = sum(1 for r in results if r['status'] == 'timeout')
        error_count = sum(1 for r in results if r['status'] == 'error')

        # 成功的时间 + 超时的时间（超时按 timeout 秒整计入）
        success_times = [r['execution_time_ms'] for r in results
                         if r['status'] in ('success', 'timeout') and r['execution_time_ms']]
        total_time = sum(success_times) if success_times else 0
        avg_time = (total_time / len(success_times)) if success_times else 0

        print("\n" + "="*70)
        print("Execution Summary:")
        print(f"  Success: {success_count}/{len(results)}")
        print(f"  Timeout: {timeout_count}/{len(results)}")
        print(f"  Error:   {error_count}/{len(results)}")
        print(f"  Total time: {total_time/1000:.1f}s")
        print(f"  Avg time: {avg_time:.0f}ms")

        # Q-error 汇总
        qerror_results = [r for r in results if r['status'] == 'success' and r.get('qerror')]
        if qerror_results:
            all_geom_means = [r['qerror']['geometric_mean'] for r in qerror_results]
            all_max_qerrors = [r['qerror']['max'] for r in qerror_results]
            
            # 计算跨所有查询的几何平均
            overall_geom_mean = self.geometric_mean(all_geom_means)
            overall_max = max(all_max_qerrors) if all_max_qerrors else 1.0

            print(f"\nQ-error Summary ({len(qerror_results)} queries with q-error data):")
            print(f"  Per-query geometric mean:")
            print(f"    Average: {sum(all_geom_means) / len(all_geom_means):.2f}")
            print(f"    Median:  {sorted(all_geom_means)[len(all_geom_means) // 2]:.2f}")
            print(f"  Overall geometric mean: {overall_geom_mean:.2f}")
            print(f"  Max q-error observed:   {overall_max:.2f}")
            
            # 统计 operator 数量
            total_ops = sum(r['qerror']['num_operators'] for r in qerror_results)
            print(f"  Total operators analyzed: {total_ops}")
        else:
            print("\nNo Q-error data collected")

        print("="*70)
        
        # 输出论文 Table 7 格式的数据
        print("\nPaper Table 7 Format:")
        print(f"  {success_count}/{len(results)} queries succeeded")
        print(f"  Total time: {total_time/1000:.1f}s")
        if qerror_results:
            print(f"  Mean Q-Error: {overall_geom_mean:.2f}")

    def close(self):
        self.cursor.close()
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description='PostgreSQL JOB Benchmark Experiment')
    parser.add_argument('--host', default='localhost', help='PostgreSQL host')
    parser.add_argument('--port', type=int, default=5432, help='PostgreSQL port')
    parser.add_argument('--dbname', required=True, help='Database name')
    parser.add_argument('--user', required=True, help='PostgreSQL user')
    parser.add_argument('--password', default='', help='PostgreSQL password')
    parser.add_argument('--query-dir', required=True, help='Directory containing SQL queries')
    parser.add_argument('--strategy', required=True, 
                       choices=['stale_prior', 'full_analyze', 'histogram_only'],
                       help='Experiment strategy')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--timeout', type=int, default=30, help='Query timeout in seconds')
    parser.add_argument('--skip-queries', nargs='+', help='Queries to skip')

    args = parser.parse_args()

    runner = PostgreSQLExperimentRunner(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password
    )

    try:
        runner.run_experiment(
            query_dir=Path(args.query_dir),
            output_dir=Path(args.output_dir),
            strategy=args.strategy,
            timeout=args.timeout,
            skip_queries=args.skip_queries
        )
    finally:
        runner.close()


if __name__ == '__main__':
    main()
