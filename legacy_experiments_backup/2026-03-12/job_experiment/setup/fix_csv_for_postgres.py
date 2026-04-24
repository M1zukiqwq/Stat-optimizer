#!/usr/bin/env python3
"""
修复IMDB CSV文件以便PostgreSQL COPY使用

使用方法:
    python3 fix_csv_for_postgres.py

修复后的文件会保存在 ./imdb_data/fixed/ 目录
"""

import csv
import sys
from pathlib import Path

# 增加CSV字段大小限制
csv.field_size_limit(sys.maxsize)

# 表配置: (表名, 期望列数)
TABLES = [
    ('comp_cast_type', 2),
    ('company_type', 2),
    ('info_type', 2),
    ('kind_type', 2),
    ('link_type', 2),
    ('role_type', 2),
    ('keyword', 3),
    ('company_name', 7),
    ('movie_link', 4),
    ('complete_cast', 3),
    ('name', 9),
    ('char_name', 6),
    ('aka_name', 8),
    ('title', 12),
    ('aka_title', 12),
    ('movie_companies', 5),
    ('movie_keyword', 3),
    ('movie_info_idx', 5),
    ('person_info', 5),
    ('movie_info', 5),
    ('cast_info', 7),
]


def clean_field(value):
    """清理字段值"""
    if value is None:
        return ''
    # 移除null字节
    value = value.replace('\x00', '')
    # 将换行符替换为空格
    value = value.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    # 规范化空格
    value = ' '.join(value.split())
    return value


def fix_row(row, expected_cols):
    """
    修复一行的列数
    - 如果列数过多，合并多余的列到最后一个有效列
    - 如果列数过少，用空字符串填充
    """
    if len(row) == expected_cols:
        return row
    elif len(row) > expected_cols:
        # 合并多余的列
        fixed = row[:expected_cols-1]
        fixed.append(','.join(row[expected_cols-1:]))
        return fixed
    else:
        # 填充空列
        return row + [''] * (expected_cols - len(row))


def process_csv(input_file, output_file, expected_cols):
    """处理单个CSV文件"""
    print(f"  处理中...", end='', flush=True)
    
    total = 0
    fixed = 0
    errors = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='replace') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        writer = csv.writer(outfile, delimiter=',', quotechar='"', 
                           quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        
        # 使用python的csv reader，设置严格的引号处理
        reader = csv.reader(infile, delimiter=',', quotechar='"', 
                           doublequote=True, strict=False)
        
        for row in reader:
            total += 1
            
            try:
                # 修复列数
                row = fix_row(row, expected_cols)
                
                # 清理每个字段
                cleaned = [clean_field(field) for field in row]
                
                # 写入
                writer.writerow(cleaned)
                fixed += 1
                
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"\n    警告: 第{total}行错误: {e}")
                continue
            
            if total % 500000 == 0:
                print(f" {total:,}...", end='', flush=True)
    
    print(f" 完成 ({fixed:,}行, {errors}错误)")
    return fixed


def main():
    data_dir = Path(__file__).parent / 'imdb_data'
    fixed_dir = data_dir / 'fixed'
    fixed_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("IMDB CSV 修复工具")
    print("="*70)
    print(f"输入目录: {data_dir}")
    print(f"输出目录: {fixed_dir}")
    print("")
    
    grand_total = 0
    
    for table_name, expected_cols in TABLES:
        input_file = data_dir / f"{table_name}.csv"
        output_file = fixed_dir / f"{table_name}.csv"
        
        if not input_file.exists():
            print(f"跳过 {table_name}.csv (文件不存在)")
            continue
        
        # 获取文件大小
        file_size = input_file.stat().st_size / (1024*1024)  # MB
        print(f"[{table_name}] ({file_size:.1f} MB, {expected_cols}列)")
        
        try:
            rows = process_csv(input_file, output_file, expected_cols)
            grand_total += rows
        except Exception as e:
            print(f"  ✗ 失败: {e}")
    
    print("="*70)
    print(f"总计: {grand_total:,} 行已修复")
    print(f"修复后的文件在: {fixed_dir}")
    print("")
    print("下一步: 运行 load_fixed_data.sh 加载数据")


if __name__ == '__main__':
    main()
