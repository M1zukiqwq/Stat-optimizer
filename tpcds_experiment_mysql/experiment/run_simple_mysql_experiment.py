#!/usr/bin/env python3
"""MySQL end-to-end runner for TPC-DS histogram experiment."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pymysql
from pymysql.cursors import DictCursor


class MySQLExperimentRunner:
    def __init__(self, host: str, port: int, dbname: str, user: str, password: str = 'tianqichu123'):
        self.conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=dbname,
            autocommit=True,
            cursorclass=DictCursor,
            local_infile=True,
        )
        self.cursor = self.conn.cursor()
        self.dbname = dbname

    def get_table_stats(self, table_name: str) -> Dict:
        self.cursor.execute(
            """
            SELECT table_rows
            FROM information_schema.tables
            WHERE table_schema = DATABASE()
              AND table_name = %s
            """,
            (table_name,),
        )
        row = self.cursor.fetchone() or {}
        return {'row_count': row.get('table_rows', 0), 'last_analyze': 'N/A'}

    def calculate_q_error(self, estimated: float, actual: float) -> float:
        if actual == 0 and estimated == 0:
            return 1.0
        if actual == 0:
            return max(estimated + 1, 2.0)
        if estimated == 0:
            return max(actual + 1, 2.0)
        return max(estimated / actual, actual / estimated)

    def geometric_mean(self, values: List[float]) -> float:
        valid = [v for v in values if v > 0]
        if not valid:
            return 1.0
        return math.exp(sum(math.log(v) for v in valid) / len(valid))

    def extract_qerror_from_analyze(self, analyze_output: str) -> List[Dict]:
        stats = []
        pattern = re.compile(r'rows=([0-9]+(?:\.[0-9]+)?)')
        for line in analyze_output.splitlines():
            if 'actual time=' not in line:
                continue
            matches = pattern.findall(line)
            if len(matches) < 2:
                continue
            estimated = float(matches[0])
            actual = float(matches[-1])
            stats.append({
                'estimated_rows': estimated,
                'actual_rows': actual,
                'q_error': self.calculate_q_error(estimated, actual),
                'line': line.strip(),
            })
        return stats

    def run_query(self, query_id: str, sql: str, timeout: int = 300) -> Dict:
        result = {
            'query_id': query_id,
            'status': 'unknown',
            'execution_time_ms': None,
            'rows_returned': None,
            'error': None,
            'qerror': None,
            'plan_raw': None,
        }
        try:
            self.cursor.execute(f"SET SESSION max_execution_time = {timeout * 1000}")
            start_time = time.time()
            self.cursor.execute(f"EXPLAIN ANALYZE {sql}")
            rows = self.cursor.fetchall()
            end_time = time.time()
            explain_text = '\n'.join(str(next(iter(row.values()))) for row in rows)
            result['status'] = 'success'
            result['execution_time_ms'] = int((end_time - start_time) * 1000)
            result['plan_raw'] = explain_text
            line_stats = self.extract_qerror_from_analyze(explain_text)
            if line_stats:
                q_errors = [entry['q_error'] for entry in line_stats]
                result['qerror'] = {
                    'num_operators': len(line_stats),
                    'mean': sum(q_errors) / len(q_errors),
                    'median': sorted(q_errors)[len(q_errors) // 2],
                    'geometric_mean': self.geometric_mean(q_errors),
                    'max': max(q_errors),
                }
        except pymysql.MySQLError as exc:
            message = str(exc)
            lowered = message.lower()
            if 'maximum statement execution time exceeded' in lowered or 'query execution was interrupted' in lowered:
                result['status'] = 'timeout'
                result['execution_time_ms'] = timeout * 1000
                result['error'] = f'timeout (>{timeout}s)'
            else:
                result['status'] = 'error'
                result['error'] = message
        except Exception as exc:
            result['status'] = 'error'
            result['error'] = str(exc)
        return result

    def load_queries(self, query_dir: Path, skip_queries: List[str]) -> Dict[str, str]:
        queries = {}
        for sql_file in sorted(query_dir.glob('*.sql')):
            query_id = sql_file.stem
            if query_id in skip_queries:
                print(f"  Skipping {query_id} (in skip list)")
                continue
            queries[query_id] = sql_file.read_text()
        return queries

    def print_summary(self, results: List[Dict]) -> None:
        success_count = sum(1 for r in results if r['status'] == 'success')
        timeout_count = sum(1 for r in results if r['status'] == 'timeout')
        error_count = sum(1 for r in results if r['status'] == 'error')
        time_values = [r['execution_time_ms'] for r in results if r['status'] in ('success', 'timeout') and r['execution_time_ms']]
        total_time = sum(time_values) if time_values else 0
        avg_time = total_time / len(time_values) if time_values else 0

        print('\n' + '=' * 70)
        print('Execution Summary:')
        print(f'  Success: {success_count}/{len(results)}')
        print(f'  Timeout: {timeout_count}/{len(results)}')
        print(f'  Error:   {error_count}/{len(results)}')
        print(f'  Total time: {total_time/1000:.1f}s')
        print(f'  Avg time: {avg_time:.0f}ms')

        qerror_results = [r for r in results if r['status'] == 'success' and r.get('qerror')]
        if qerror_results:
            geoms = [r['qerror']['geometric_mean'] for r in qerror_results]
            overall_geom_mean = self.geometric_mean(geoms)
            print(f'\nQ-error Summary ({len(qerror_results)} queries with q-error data):')
            print(f'  Per-query geometric mean:')
            print(f'    Average: {sum(geoms) / len(geoms):.2f}')
            print(f'    Median:  {sorted(geoms)[len(geoms) // 2]:.2f}')
            print(f'  Overall geometric mean: {overall_geom_mean:.2f}')
            print(f'  Max q-error observed:   {max(r["qerror"]["max"] for r in qerror_results):.2f}')
            print(f'  Total operators analyzed: {sum(r["qerror"]["num_operators"] for r in qerror_results)}')
            print('\nPaper Table 7 Format:')
            print(f'  {success_count}/{len(results)} queries succeeded')
            print(f'  Total time: {total_time/1000:.1f}s')
            print(f'  Mean Q-Error: {overall_geom_mean:.2f}')
        else:
            print('\nNo Q-error data collected')

    def run_experiment(self, query_dir: Path, output_dir: Path, strategy: str, timeout: int = 300, skip_queries: Optional[List[str]] = None):
        output_dir.mkdir(parents=True, exist_ok=True)
        skip_queries = skip_queries or []

        print(f'\nStrategy: {strategy}')
        print(f'Database: {self.dbname}')
        print(f'Timeout: {timeout}s')
        print('=' * 70)

        queries = self.load_queries(query_dir, skip_queries)
        print(f'Loaded {len(queries)} queries')

        print('\nCurrent table statistics:')
        for table in ['item', 'customer', 'store_sales']:
            stats = self.get_table_stats(table)
            print(f"  {table}: {stats['row_count']:,} rows")
        print('=' * 70)

        results = []
        for index, (query_id, sql) in enumerate(queries.items(), 1):
            print(f'[{index}/{len(queries)}] Running {query_id}...', end=' ', flush=True)
            result = self.run_query(query_id, sql, timeout)
            results.append(result)
            if result['status'] == 'success':
                if result.get('qerror'):
                    print(f"✓ {result['execution_time_ms']}ms | ops={result['qerror']['num_operators']} q-error={result['qerror']['geometric_mean']:.2f}")
                else:
                    print(f"✓ {result['execution_time_ms']}ms (no q-error data)")
            elif result['status'] == 'timeout':
                print(f"⏱ {result['error']}")
            else:
                print(f"✗ {result['error']}")

        output_file = output_dir / f'{strategy}_results.json'
        with open(output_file, 'w') as handle:
            json.dump({
                'strategy': strategy,
                'timestamp': datetime.now().isoformat(),
                'total_queries': len(queries),
                'results': results,
            }, handle, indent=2)

        print(f'\n✓ Results saved to {output_file}')
        self.print_summary(results)

    def close(self) -> None:
        self.cursor.close()
        self.conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='MySQL TPC-DS Benchmark Experiment')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=3306)
    parser.add_argument('--dbname', required=True)
    parser.add_argument('--user', required=True)
    parser.add_argument('--password', default='tianqichu123')
    parser.add_argument('--query-dir', required=True)
    parser.add_argument('--strategy', required=True, choices=['stale_prior', 'full_analyze', 'histogram_only'])
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--skip-queries', nargs='+')
    args = parser.parse_args()

    runner = MySQLExperimentRunner(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )
    try:
        runner.run_experiment(
            query_dir=Path(args.query_dir),
            output_dir=Path(args.output_dir),
            strategy=args.strategy,
            timeout=args.timeout,
            skip_queries=args.skip_queries,
        )
    finally:
        runner.close()


if __name__ == '__main__':
    main()
