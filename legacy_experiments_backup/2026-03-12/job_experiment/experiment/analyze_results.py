#!/usr/bin/env python3
"""
分析 JOB 实验结果，生成论文 Table 4
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_results(result_file: Path) -> Dict:
    """加载实验结果"""
    with open(result_file, 'r') as f:
        return json.load(f)


def analyze_strategy(results: Dict) -> Dict:
    """分析单个策略的结果"""
    query_results = results['results']

    success_results = [r for r in query_results if r['status'] == 'success']
    error_results = [r for r in query_results if r['status'] == 'error']

    if not success_results:
        return {
            'total_time_s': 0,
            'avg_time_ms': 0,
            'success_count': 0,
            'error_count': len(error_results),
            'total_queries': len(query_results)
        }

    times = [r['execution_time_ms'] for r in success_results]

    return {
        'total_time_s': sum(times) / 1000,
        'avg_time_ms': sum(times) / len(times),
        'success_count': len(success_results),
        'error_count': len(error_results),
        'total_queries': len(query_results)
    }


def generate_table4(baseline: Dict, stale: Dict, oasis: Dict, full_analyze: Dict) -> pd.DataFrame:
    """生成论文 Table 4"""
    data = []

    for name, stats in [
        ('Baseline (no drift)', baseline),
        ('Stale Prior', stale),
        ('OASIS', oasis),
        ('Full ANALYZE', full_analyze)
    ]:
        # 计算 vs. Stale Prior 的改进
        if name == 'Stale Prior':
            vs_stale = '—'
        else:
            speedup = (stale['total_time_s'] - stats['total_time_s']) / stale['total_time_s'] * 100
            vs_stale = f"{speedup:+.1f}%"

        data.append({
            'Method': name,
            'Total time (s)': f"{stats['total_time_s']:.1f}",
            'Success': f"{stats['success_count']}/{stats['total_queries']}",
            'Errors': stats['error_count'],
            'vs. Stale': vs_stale
        })

    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description='Analyze JOB experiment results')
    parser.add_argument('--baseline', required=True, help='Baseline results directory')
    parser.add_argument('--stale-prior', required=True, help='Stale prior results directory')
    parser.add_argument('--oasis', required=True, help='OASIS results directory')
    parser.add_argument('--full-analyze', required=True, help='Full ANALYZE results directory')
    parser.add_argument('--output', required=True, help='Output directory')

    args = parser.parse_args()

    # 加载结果
    print("Loading results...")
    baseline = load_results(Path(args.baseline) / 'baseline_results.json')
    stale = load_results(Path(args.stale_prior) / 'stale_prior_results.json')
    oasis = load_results(Path(args.oasis) / 'oasis_results.json')
    full_analyze = load_results(Path(args.full_analyze) / 'full_analyze_results.json')

    # 分析
    print("Analyzing...")
    baseline_stats = analyze_strategy(baseline)
    stale_stats = analyze_strategy(stale)
    oasis_stats = analyze_strategy(oasis)
    full_analyze_stats = analyze_strategy(full_analyze)

    # 生成 Table 4
    table4 = generate_table4(baseline_stats, stale_stats, oasis_stats, full_analyze_stats)

    # 保存
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_file = output_dir / 'table4.csv'
    table4.to_csv(csv_file, index=False)
    print(f"\n✓ Saved CSV to {csv_file}")

    # LaTeX
    latex_file = output_dir / 'table4.tex'
    with open(latex_file, 'w') as f:
        f.write(table4.to_latex(index=False, escape=False))
    print(f"✓ Saved LaTeX to {latex_file}")

    # 打印到控制台
    print("\n" + "="*70)
    print("Table 4: JOB Benchmark Results (q=15 drift)")
    print("="*70)
    print(table4.to_string(index=False))
    print("="*70)

    # 打印关键发现
    speedup = (stale_stats['total_time_s'] - oasis_stats['total_time_s']) / stale_stats['total_time_s'] * 100
    print(f"\nKey findings:")
    print(f"  - OASIS speedup vs. Stale Prior: {speedup:+.1f}%")
    print(f"  - OASIS total time: {oasis_stats['total_time_s']:.1f}s")
    print(f"  - Stale Prior total time: {stale_stats['total_time_s']:.1f}s")


if __name__ == '__main__':
    main()
