#!/usr/bin/env python3
"""
诊断 Iceberg 统计文件问题

检查：
1. 当前 snapshot ID
2. 统计文件关联的 snapshot ID
3. 时间戳差异
4. 是否存在 snapshot 不匹配问题
"""

import argparse
import prestodb
from datetime import datetime


def diagnose_table(host: str, port: int, catalog: str, schema: str, table: str, user: str = 'tianqc'):
    """诊断单个表的统计文件状态"""
    conn = prestodb.dbapi.connect(
        host=host,
        port=port,
        user=user,
        catalog=catalog,
        schema=schema,
        http_scheme='http'
    )
    cursor = conn.cursor()

    print(f"\n{'='*70}")
    print(f"诊断表: {catalog}.{schema}.{table}")
    print(f"{'='*70}\n")

    # 1. 查询当前 snapshot ID
    try:
        cursor.execute(f"SELECT snapshot_id FROM \"{catalog}\".\"{schema}\".\"{table}$snapshots\" ORDER BY committed_at DESC LIMIT 1")
        current_snapshot = cursor.fetchone()
        if current_snapshot:
            current_snapshot_id = current_snapshot[0]
            print(f"✓ 当前 Snapshot ID: {current_snapshot_id}")
        else:
            print(f"✗ 无法获取当前 snapshot")
            return
    except Exception as e:
        print(f"✗ 查询 snapshot 失败: {e}")
        return

    # 2. 查询最近的 snapshots
    try:
        cursor.execute(f"""
            SELECT snapshot_id, committed_at, operation, summary
            FROM \"{catalog}\".\"{schema}\".\"{table}$snapshots\"
            ORDER BY committed_at DESC
            LIMIT 5
        """)
        snapshots = cursor.fetchall()
        print(f"\n最近 5 个 Snapshots:")
        for i, (snap_id, committed_at, operation, summary) in enumerate(snapshots, 1):
            print(f"  {i}. Snapshot {snap_id}")
            print(f"     时间: {committed_at}")
            print(f"     操作: {operation}")
            if summary:
                # 解析 summary 中的 total-records
                if 'total-records' in summary:
                    import re
                    match = re.search(r'total-records[\'"]?\s*[:=]\s*[\'"]?(\d+)', summary)
                    if match:
                        print(f"     记录数: {match.group(1)}")
            print()
    except Exception as e:
        print(f"✗ 查询 snapshots 失败: {e}")

    # 3. 查询统计文件
    try:
        cursor.execute(f"""
            SELECT snapshot_id, file_path, file_size_in_bytes
            FROM \"{catalog}\".\"{schema}\".\"{table}$statistics\"
            ORDER BY snapshot_id DESC
        """)
        stats_files = cursor.fetchall()

        if not stats_files:
            print(f"⚠️  没有找到统计文件！")
            print(f"   可能原因：")
            print(f"   1. 从未运行过 ANALYZE")
            print(f"   2. ANALYZE 失败")
            print(f"   3. 统计文件被删除")
        else:
            print(f"统计文件 ({len(stats_files)} 个):")
            for i, (snap_id, file_path, file_size) in enumerate(stats_files, 1):
                is_current = "✓ 当前" if snap_id == current_snapshot_id else "✗ 过期"
                print(f"  {i}. {is_current} - Snapshot {snap_id}")
                print(f"     文件: {file_path}")
                print(f"     大小: {file_size} bytes")
                print()

            # 检查是否存在当前 snapshot 的统计文件
            current_stats = [s for s in stats_files if s[0] == current_snapshot_id]
            if not current_stats:
                print(f"⚠️  问题诊断：")
                print(f"   当前 Snapshot ID: {current_snapshot_id}")
                print(f"   统计文件关联的 Snapshot IDs: {[s[0] for s in stats_files]}")
                print(f"   ❌ 当前 snapshot 没有对应的统计文件！")
                print(f"   ")
                print(f"   这就是 Q-error 一模一样的原因：")
                print(f"   1. 数据漂移（INSERT/DELETE）创建了新的 snapshot")
                print(f"   2. ANALYZE 在漂移前运行，关联到旧的 snapshot")
                print(f"   3. 查询时使用新 snapshot，但读取的是旧 snapshot 的统计")
                print(f"   4. 重新 ANALYZE 后，仍然关联到旧 snapshot（如果没有新的数据变更）")
                print(f"   ")
                print(f"   解决方案：")
                print(f"   1. 在数据漂移后立即运行 ANALYZE")
                print(f"   2. 或者在 ANALYZE 后再次运行数据漂移（触发新 snapshot）")
                print(f"   3. 确保 ANALYZE 关联到最新的 snapshot")
            else:
                print(f"✓ 当前 snapshot 有对应的统计文件")
    except Exception as e:
        print(f"✗ 查询统计文件失败: {e}")

    cursor.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='诊断 Iceberg 表的统计文件状态')
    parser.add_argument('--presto-host', required=True, help='Presto host:port')
    parser.add_argument('--catalog', default='iceberg', help='Catalog name')
    parser.add_argument('--schema', default='imdb', help='Schema name')
    parser.add_argument('--user', default='tianqc', help='Presto user')
    parser.add_argument('--tables', nargs='+',
                       default=['title', 'cast_info', 'movie_info', 'movie_companies', 'name', 'movie_keyword'],
                       help='Tables to diagnose')

    args = parser.parse_args()

    host, port = args.presto_host.split(':')

    for table in args.tables:
        diagnose_table(host, int(port), args.catalog, args.schema, table, user=args.user)

    print(f"\n{'='*70}")
    print("诊断完成")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
