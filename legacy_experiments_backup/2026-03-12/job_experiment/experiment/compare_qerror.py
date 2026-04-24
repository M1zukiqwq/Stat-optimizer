#!/usr/bin/env python3
"""
对比 ANALYZE 前后的 Q-error 改善

输入：
- before_analyze_qerror.json: 漂移后、ANALYZE 前的 Q-error
- after_analyze_qerror.json: ANALYZE 后的 Q-error

输出：
- Q-error 改善统计
- 论文用的表格和图表数据
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np


def load_qerror_results(result_file: Path) -> Dict:
    """加载 Q-error 结果"""
    with open(result_file, 'r') as f:
        return json.load(f)


def compare_qerror(before: Dict, after: Dict) -> pd.DataFrame:
    """对比两个 Q-error 结果"""
    # 构建查询 ID 到结果的映射
    before_map = {r['query_id']: r for r in before['results']}
    after_map = {r['query_id']: r for r in after['results']}

    # 找到共同的查询
    common_queries = set(before_map.keys()) & set(after_map.keys())

    if not common_queries:
        print("⚠️  No common queries found between before and after results")
        return pd.DataFrame()

    # 构建对比数据
    data = []
    for query_id in sorted(common_queries):
        before_result = before_map[query_id]
        after_result = after_map[query_id]

        before_mean = before_result['summary']['mean_q_error']
        after_mean = after_result['summary']['mean_q_error']
        before_max = before_result['summary']['max_q_error']
        after_max = after_result['summary']['max_q_error']
        before_geom = before_result['summary']['geometric_mean_q_error']
        after_geom = after_result['summary']['geometric_mean_q_error']

        # 计算改善百分比
        mean_improvement = (before_mean - after_mean) / before_mean * 100 if before_mean > 0 else 0
        max_improvement = (before_max - after_max) / before_max * 100 if before_max > 0 else 0
        geom_improvement = (before_geom - after_geom) / before_geom * 100 if before_geom > 0 else 0

        data.append({
            'query_id': query_id,
            'before_mean_qerror': before_mean,
            'after_mean_qerror': after_mean,
            'mean_improvement_pct': mean_improvement,
            'before_max_qerror': before_max,
            'after_max_qerror': after_max,
            'max_improvement_pct': max_improvement,
            'before_geom_qerror': before_geom,
            'after_geom_qerror': after_geom,
            'geom_improvement_pct': geom_improvement
        })

    return pd.DataFrame(data)


def print_summary(df: pd.DataFrame):
    """打印汇总统计"""
    print("\n" + "="*70)
    print("Q-error Improvement Summary")
    print("="*70)

    print(f"\nTotal queries analyzed: {len(df)}")

    # Mean Q-error 统计
    print(f"\nMean Q-error:")
    print(f"  Before ANALYZE: {df['before_mean_qerror'].mean():.2f} (median: {df['before_mean_qerror'].median():.2f})")
    print(f"  After ANALYZE:  {df['after_mean_qerror'].mean():.2f} (median: {df['after_mean_qerror'].median():.2f})")
    print(f"  Average improvement: {df['mean_improvement_pct'].mean():.1f}%")
    print(f"  Queries improved: {(df['mean_improvement_pct'] > 0).sum()} / {len(df)}")
    print(f"  Queries degraded: {(df['mean_improvement_pct'] < 0).sum()} / {len(df)}")

    # Max Q-error 统计
    print(f"\nMax Q-error:")
    print(f"  Before ANALYZE: {df['before_max_qerror'].mean():.2f} (median: {df['before_max_qerror'].median():.2f})")
    print(f"  After ANALYZE:  {df['after_max_qerror'].mean():.2f} (median: {df['after_max_qerror'].median():.2f})")
    print(f"  Average improvement: {df['max_improvement_pct'].mean():.1f}%")

    # Geometric mean Q-error 统计
    print(f"\nGeometric mean Q-error:")
    print(f"  Before ANALYZE: {df['before_geom_qerror'].mean():.2f} (median: {df['before_geom_qerror'].median():.2f})")
    print(f"  After ANALYZE:  {df['after_geom_qerror'].mean():.2f} (median: {df['after_geom_qerror'].median():.2f})")
    print(f"  Average improvement: {df['geom_improvement_pct'].mean():.1f}%")

    # 最大改善和最大退化的查询
    print(f"\nTop 5 most improved queries (by mean Q-error):")
    top_improved = df.nlargest(5, 'mean_improvement_pct')[['query_id', 'before_mean_qerror', 'after_mean_qerror', 'mean_improvement_pct']]
    for _, row in top_improved.iterrows():
        print(f"  {row['query_id']}: {row['before_mean_qerror']:.2f} → {row['after_mean_qerror']:.2f} ({row['mean_improvement_pct']:+.1f}%)")

    if (df['mean_improvement_pct'] < 0).any():
        print(f"\nTop 5 most degraded queries (by mean Q-error):")
        top_degraded = df.nsmallest(5, 'mean_improvement_pct')[['query_id', 'before_mean_qerror', 'after_mean_qerror', 'mean_improvement_pct']]
        for _, row in top_degraded.iterrows():
            print(f"  {row['query_id']}: {row['before_mean_qerror']:.2f} → {row['after_mean_qerror']:.2f} ({row['mean_improvement_pct']:+.1f}%)")

    print("="*70)


def generate_paper_table(df: pd.DataFrame, output_dir: Path):
    """生成论文用的表格"""
    # 汇总统计
    summary_data = {
        'Metric': [
            'Mean Q-error (avg)',
            'Mean Q-error (median)',
            'Max Q-error (avg)',
            'Max Q-error (median)',
            'Geom. mean Q-error (avg)',
            'Geom. mean Q-error (median)'
        ],
        'Before ANALYZE': [
            f"{df['before_mean_qerror'].mean():.2f}",
            f"{df['before_mean_qerror'].median():.2f}",
            f"{df['before_max_qerror'].mean():.2f}",
            f"{df['before_max_qerror'].median():.2f}",
            f"{df['before_geom_qerror'].mean():.2f}",
            f"{df['before_geom_qerror'].median():.2f}"
        ],
        'After ANALYZE': [
            f"{df['after_mean_qerror'].mean():.2f}",
            f"{df['after_mean_qerror'].median():.2f}",
            f"{df['after_max_qerror'].mean():.2f}",
            f"{df['after_max_qerror'].median():.2f}",
            f"{df['after_geom_qerror'].mean():.2f}",
            f"{df['after_geom_qerror'].median():.2f}"
        ],
        'Improvement': [
            f"{df['mean_improvement_pct'].mean():.1f}%",
            f"{(df['before_mean_qerror'].median() - df['after_mean_qerror'].median()) / df['before_mean_qerror'].median() * 100:.1f}%",
            f"{df['max_improvement_pct'].mean():.1f}%",
            f"{(df['before_max_qerror'].median() - df['after_max_qerror'].median()) / df['before_max_qerror'].median() * 100:.1f}%",
            f"{df['geom_improvement_pct'].mean():.1f}%",
            f"{(df['before_geom_qerror'].median() - df['after_geom_qerror'].median()) / df['before_geom_qerror'].median() * 100:.1f}%"
        ]
    }

    summary_df = pd.DataFrame(summary_data)

    # 保存 CSV
    csv_file = output_dir / 'qerror_comparison_summary.csv'
    summary_df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved summary CSV to {csv_file}")

    # 保存 LaTeX
    latex_file = output_dir / 'qerror_comparison_summary.tex'
    with open(latex_file, 'w') as f:
        f.write(summary_df.to_latex(index=False, escape=False))
    print(f"✓ Saved summary LaTeX to {latex_file}")

    # 保存详细的每个查询的对比
    detail_csv = output_dir / 'qerror_comparison_detail.csv'
    df.to_csv(detail_csv, index=False)
    print(f"✓ Saved detail CSV to {detail_csv}")

    # 打印表格
    print("\n" + "="*70)
    print("Q-error Comparison Table (for paper)")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("="*70)


def generate_cdf_data(df: pd.DataFrame, output_dir: Path):
    """生成 CDF 图表数据（用于绘制 Q-error 分布）"""
    # 准备 CDF 数据
    before_mean = sorted(df['before_mean_qerror'].values)
    after_mean = sorted(df['after_mean_qerror'].values)

    cdf_data = {
        'before_qerror': before_mean,
        'before_cdf': [i / len(before_mean) for i in range(1, len(before_mean) + 1)],
        'after_qerror': after_mean,
        'after_cdf': [i / len(after_mean) for i in range(1, len(after_mean) + 1)]
    }

    # 保存为 CSV（方便用 matplotlib/gnuplot 绘图）
    cdf_file = output_dir / 'qerror_cdf_data.csv'
    pd.DataFrame({
        'qerror': before_mean + after_mean,
        'cdf': cdf_data['before_cdf'] + cdf_data['after_cdf'],
        'category': ['Before ANALYZE'] * len(before_mean) + ['After ANALYZE'] * len(after_mean)
    }).to_csv(cdf_file, index=False)

    print(f"✓ Saved CDF data to {cdf_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare Q-error before and after ANALYZE')
    parser.add_argument('--before', required=True, help='Q-error results before ANALYZE (JSON)')
    parser.add_argument('--after', required=True, help='Q-error results after ANALYZE (JSON)')
    parser.add_argument('--output-dir', required=True, help='Output directory')

    args = parser.parse_args()

    # 加载结果
    print("Loading Q-error results...")
    before = load_qerror_results(Path(args.before))
    after = load_qerror_results(Path(args.after))

    print(f"  Before ANALYZE: {len(before['results'])} queries")
    print(f"  After ANALYZE: {len(after['results'])} queries")

    # 对比
    print("\nComparing Q-error...")
    df = compare_qerror(before, after)

    if df.empty:
        print("⚠️  No comparison data available")
        return

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打印汇总
    print_summary(df)

    # 生成论文表格
    generate_paper_table(df, output_dir)

    # 生成 CDF 数据
    generate_cdf_data(df, output_dir)

    print(f"\n✓ All outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
