#!/usr/bin/env python3
"""
生成 JOB-Enhanced 查询

在原始 JOB 查询的基础上，给 JOIN 列增加范围 filter，
使得直方图能够影响基数估算。
"""

import argparse
import re
from pathlib import Path


# 增强规则：在 WHERE 子句中插入范围 filter
# 使用 > 和 < 而不是 = 或 LIKE，让直方图发挥更大作用
ENHANCEMENT_RULES = {
    # 规则 1：如果查询包含 title 表，增加 production_year filter
    'title': {
        'pattern': r'(WHERE\s+)',
        'filter': 't.production_year > 1920 AND t.production_year < 1950 AND '
    },
    # 规则 2：如果查询包含 cast_info 表，增加 role_id filter
    'cast_info': {
        'pattern': r'(WHERE\s+)',
        'filter': 'ci.role_id > 0 AND ci.role_id < 3 AND '
    },
    # 规则 3：如果查询包含 movie_info 表，增加 info_type_id filter
    'movie_info': {
        'pattern': r'(WHERE\s+)',
        'filter': 'mi.info_type_id > 0 AND mi.info_type_id < 6 AND '
    },
    # 规则 4：如果查询包含 movie_companies 表，增加 company_type_id filter
    'movie_companies': {
        'pattern': r'(WHERE\s+)',
        'filter': 'mc.company_type_id > 0 AND mc.company_type_id < 3 AND '
    },
}


def enhance_query(query_sql: str, query_id: str) -> str:
    """
    增强单个查询

    策略：
    1. 检测查询中使用的表
    2. 根据表名应用相应的范围 filter
    3. 在 WHERE 子句开头插入 filter
    """
    # 检测查询中的表
    tables_in_query = []
    if re.search(r'\btitle\b.*\bAS\s+t\b', query_sql, re.IGNORECASE):
        tables_in_query.append('title')
    if re.search(r'\bcast_info\b.*\bAS\s+ci\b', query_sql, re.IGNORECASE):
        tables_in_query.append('cast_info')
    if re.search(r'\bmovie_info\b.*\bAS\s+mi\b', query_sql, re.IGNORECASE):
        tables_in_query.append('movie_info')
    if re.search(r'\bmovie_companies\b.*\bAS\s+mc\b', query_sql, re.IGNORECASE):
        tables_in_query.append('movie_companies')

    # 如果没有检测到目标表，返回原查询
    if not tables_in_query:
        return None

    # 应用增强规则
    where_match = re.search(r'WHERE\s+', query_sql, re.IGNORECASE)
    if not where_match:
        return None

    # 在 WHERE 后面插入所有适用的 filter
    insert_pos = where_match.end()
    filters = []
    filters_added = []

    for table in tables_in_query:
        if table in ENHANCEMENT_RULES:
            filters.append(ENHANCEMENT_RULES[table]['filter'])
            filters_added.append(table)

    if filters:
        # 插入 filter
        filter_str = ''.join(filters)
        enhanced = query_sql[:insert_pos] + filter_str + query_sql[insert_pos:]

        # 添加注释说明
        comment = f"-- JOB-Enhanced: Added range filters on {', '.join(filters_added)}\n"
        enhanced = comment + enhanced

        return enhanced

    return None


def generate_enhanced_queries(input_dir: Path, output_dir: Path, query_ids: list = None):
    """
    生成 JOB-Enhanced 查询

    Args:
        input_dir: 原始 JOB 查询目录
        output_dir: 输出目录
        query_ids: 要增强的查询 ID 列表（如果为 None，则处理所有查询）
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有 JOB 查询文件
    job_files = sorted(input_dir.glob('*.sql'))

    if not job_files:
        print(f"✗ 没有找到 JOB 查询文件在: {input_dir}")
        return

    print(f"处理 {len(job_files)} 个 JOB 查询...")
    print(f"输出目录: {output_dir}\n")

    enhanced_count = 0
    skipped_count = 0

    for job_file in job_files:
        query_id = job_file.stem

        # 如果指定了 query_ids，只处理指定的查询
        if query_ids and query_id not in query_ids:
            continue

        # 读取原始查询
        with open(job_file, 'r') as f:
            original_sql = f.read()

        # 增强查询
        enhanced_sql = enhance_query(original_sql, query_id)

        if enhanced_sql:
            # 写入增强后的查询
            output_file = output_dir / f"{query_id}.sql"
            with open(output_file, 'w') as f:
                f.write(enhanced_sql)

            print(f"  ✓ {query_id}.sql (enhanced)")
            enhanced_count += 1
        else:
            skipped_count += 1

    print(f"\n{'='*60}")
    print(f"完成！")
    print(f"  增强: {enhanced_count} 个查询")
    print(f"  跳过: {skipped_count} 个查询（不包含目标表）")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='生成 JOB-Enhanced 查询')
    parser.add_argument('--input', default='queries/job',
                       help='原始 JOB 查询目录（默认: queries/job）')
    parser.add_argument('--output', default='queries/job_enhanced',
                       help='输出目录（默认: queries/job_enhanced）')
    parser.add_argument('--queries', nargs='+',
                       help='要增强的查询 ID（例如: 10a 10b 11a）')

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    generate_enhanced_queries(input_dir, output_dir, args.queries)


if __name__ == '__main__':
    main()
