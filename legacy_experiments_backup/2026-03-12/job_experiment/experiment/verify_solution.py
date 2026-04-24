#!/usr/bin/env python3
"""
验证 Step 7.5 方案的有效性

此脚本模拟实验流程，检查每一步的 snapshot 和统计文件状态，
验证方案 A 是否能正确解决 Q-error 一模一样的问题。
"""

import argparse
import prestodb
from datetime import datetime
from typing import List, Dict, Optional


class SnapshotVerifier:
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

    def get_current_snapshot(self, table: str) -> Optional[int]:
        """获取表的当前 snapshot ID"""
        try:
            self.cursor.execute(f"""
                SELECT snapshot_id
                FROM "{self.catalog}"."{self.schema}"."{table}$snapshots"
                ORDER BY committed_at DESC
                LIMIT 1
            """)
            result = self.cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            print(f"✗ 获取 snapshot 失败: {e}")
            return None

    def get_statistics_files(self, table: str) -> List[Dict]:
        """获取表的所有统计文件"""
        try:
            self.cursor.execute(f"""
                SELECT snapshot_id, file_path, file_size_in_bytes
                FROM "{self.catalog}"."{self.schema}"."{table}$statistics"
                ORDER BY snapshot_id DESC
            """)
            return [
                {'snapshot_id': row[0], 'file_path': row[1], 'file_size': row[2]}
                for row in self.cursor.fetchall()
            ]
        except Exception as e:
            print(f"✗ 获取统计文件失败: {e}")
            return []

    def get_snapshot_info(self, table: str, snapshot_id: int) -> Optional[Dict]:
        """获取 snapshot 的详细信息"""
        try:
            self.cursor.execute(f"""
                SELECT snapshot_id, committed_at, operation, summary
                FROM "{self.catalog}"."{self.schema}"."{table}$snapshots"
                WHERE snapshot_id = {snapshot_id}
            """)
            result = self.cursor.fetchone()
            if result:
                return {
                    'snapshot_id': result[0],
                    'committed_at': result[1],
                    'operation': result[2],
                    'summary': result[3]
                }
            return None
        except Exception as e:
            print(f"✗ 获取 snapshot 信息失败: {e}")
            return None

    def verify_step(self, step_name: str, table: str, expected_behavior: str):
        """验证某一步的状态"""
        print(f"\n{'='*70}")
        print(f"验证: {step_name}")
        print(f"{'='*70}")

        current_snapshot = self.get_current_snapshot(table)
        if not current_snapshot:
            print(f"✗ 无法获取当前 snapshot")
            return False

        print(f"当前 Snapshot ID: {current_snapshot}")

        stats_files = self.get_statistics_files(table)
        print(f"统计文件数量: {len(stats_files)}")

        if not stats_files:
            print(f"⚠️  没有统计文件")
            return False

        # 找到与当前 snapshot 最接近的统计文件
        closest_stats = None
        min_distance = float('inf')

        for stats in stats_files:
            stats_snapshot_id = stats['snapshot_id']
            # 简化的距离计算（实际 Presto 使用时间戳差异）
            distance = abs(current_snapshot - stats_snapshot_id)

            print(f"  统计文件: Snapshot {stats_snapshot_id}, 距离: {distance}")

            if distance < min_distance:
                min_distance = distance
                closest_stats = stats

        if closest_stats:
            print(f"\n✓ 最接近的统计文件:")
            print(f"  Snapshot ID: {closest_stats['snapshot_id']}")
            print(f"  文件路径: {closest_stats['file_path']}")
            print(f"  距离: {min_distance}")

            if closest_stats['snapshot_id'] == current_snapshot:
                print(f"  ✓ 统计文件与当前 snapshot 完全匹配（距离=0）")
            else:
                print(f"  ⚠️  统计文件与当前 snapshot 不匹配")

        print(f"\n预期行为: {expected_behavior}")

        return True

    def close(self):
        self.cursor.close()
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description='验证 Step 7.5 方案的有效性')
    parser.add_argument('--presto-host', required=True, help='Presto host:port')
    parser.add_argument('--catalog', default='iceberg', help='Catalog name')
    parser.add_argument('--schema', default='imdb', help='Schema name')
    parser.add_argument('--user', default='tianqc', help='Presto user')
    parser.add_argument('--table', default='title', help='Table to verify')
    parser.add_argument('--step', required=True,
                       choices=['after_initial_analyze', 'after_drift', 'after_stale_prior',
                               'after_reanalyze', 'after_trigger_snapshot', 'after_full_analyze'],
                       help='Which step to verify')

    args = parser.parse_args()

    host, port = args.presto_host.split(':')
    verifier = SnapshotVerifier(host, int(port), args.catalog, args.schema, user=args.user)

    step_descriptions = {
        'after_initial_analyze': '初始 ANALYZE 后，应该有统计文件关联到当前 snapshot',
        'after_drift': '数据漂移后，当前 snapshot 应该是新的，但统计文件仍然关联到旧 snapshot',
        'after_stale_prior': 'Stale Prior 测试后，snapshot 不应该变化（只是 SELECT 查询）',
        'after_reanalyze': '重新 ANALYZE 后，应该有新的统计文件关联到当前 snapshot',
        'after_trigger_snapshot': '触发新 snapshot 后，当前 snapshot 应该是新的，统计文件关联到上一个 snapshot',
        'after_full_analyze': 'Full ANALYZE 测试后，应该使用上一个 snapshot 的统计文件（新鲜的）'
    }

    try:
        verifier.verify_step(
            args.step.replace('_', ' ').title(),
            args.table,
            step_descriptions[args.step]
        )
    finally:
        verifier.close()

    print(f"\n{'='*70}")
    print("验证完成")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
