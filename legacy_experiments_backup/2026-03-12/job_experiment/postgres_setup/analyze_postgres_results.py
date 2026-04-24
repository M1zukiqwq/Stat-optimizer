#!/usr/bin/env python3
"""
PostgreSQL 实验结果分析脚本

比较 baseline, stale_prior, full_analyze 三种策略的：
1. 执行时间
2. Q-error
3. 受益查询比例
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_results(file_path: Path) -> Dict:
    """加载实验结果"""
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_execution_time(baseline: Dict, stale: Dict, full_analyze: Dict):
    """分析执行时间"""
    print("\n" + "="*70)
    print("Execution Time Analysis")
    print("="*70)

    # 提取成功查询的执行时间
    baseline_times = [r['execution_time_ms'] for r in baseline['results'] if r['status'] == 'success']
    stale_times = [r['execution_time_ms'] for r in stale['results'] if r['status'] == 'success']
    full_times = [r['execution_time_ms'] for r in full_analyze['results'] if r['status'] == 'success']

    baseline_total = sum(baseline_times)
    stale_total = sum(stale_times)
    full_total = sum(full_times)

    print(f"\nTotal Execution Time:")
    print(f"  Baseline:      {baseline_total/1000:.1f}s")
    print(f"  Stale Prior:   {stale_total/1000:.1f}s  ({(stale_total/baseline_total - 1)*100:+.1f}%)")
    print(f"  Full ANALYZE:  {full_total/1000:.1f}s  ({(full_total/baseline_total - 1)*100:+.1f}%)")

    print(f"\nAverage Execution Time:")
    print(f"  Baseline:      {baseline_total/len(baseline_times):.0f}ms")
    print(f"  Stale Prior:   {stale_total/len(stale_times):.0f}ms")
    print(f"  Full ANALYZE:  {full_total/len(full_times):.0f}ms")

    # 计算受益查询比例（相对于 Stale Prior）
    speedup_count = 0
    for i in range(len(stale_times)):
        if full_times[i] < stale_times[i] * 0.95:  # 至少 5% 提升
            speedup_count += 1

    print(f"\nQueries with >5% speedup (Full ANALYZE vs Stale Prior):")
    print(f"  Count: {speedup_count}/{len(stale_times)} ({speedup_count/len(stale_times)*100:.1f}%)")


def analyze_qerror(baseline: Dict, stale: Dict, full_analyze: Dict):
    """分析 Q-error"""
    print("\n" + "="*70)
    print("Q-Error Analysis")
    print("="*70)

    # 提取 Q-error
    def extract_qerrors(results: Dict) -> List[float]:
        qerrors = []
        for r in results['results']:
            if r['status'] == 'success' and r.get('qerror'):
                qerrors.append(r['qerror']['mean'])
        return qerrors

    baseline_qerrors = extract_qerrors(baseline)
    stale_qerrors = extract_qerrors(stale)
    full_qerrors = extract_qerrors(full_analyze)

    if not baseline_qerrors or not stale_qerrors or not full_qerrors:
        print("  ⚠ Q-error data not available")
        return

    baseline_mean = sum(baseline_qerrors) / len(baseline_qerrors)
    stale_mean = sum(stale_qerrors) / len(stale_qerrors)
    full_mean = sum(full_qerrors) / len(full_qerrors)

    print(f"\nMean Q-Error:")
    print(f"  Baseline:      {baseline_mean:.2f}")
    print(f"  Stale Prior:   {stale_mean:.2f}  ({(stale_mean/baseline_mean - 1)*100:+.1f}%)")
    print(f"  Full ANALYZE:  {full_mean:.2f}  ({(full_mean/baseline_mean - 1)*100:+.1f}%)")

    print(f"\nQ-Error Reduction (vs Stale Prior):")
    print(f"  Full ANALYZE:  {(1 - full_mean/stale_mean)*100:.1f}%")

    # 计算中位数
    baseline_median = sorted(baseline_qerrors)[len(baseline_qerrors) // 2]
    stale_median = sorted(stale_qerrors)[len(stale_qerrors) // 2]
    full_median = sorted(full_qerrors)[len(full_qerrors) // 2]

    print(f"\nMedian Q-Error:")
    print(f"  Baseline:      {baseline_median:.2f}")
    print(f"  Stale Prior:   {stale_median:.2f}")
    print(f"  Full ANALYZE:  {full_median:.2f}")

    # 计算最大值
    baseline_max = max(baseline_qerrors)
    stale_max = max(stale_qerrors)
    full_max = max(full_qerrors)

    print(f"\nMax Q-Error:")
    print(f"  Baseline:      {baseline_max:.2f}")
    print(f"  Stale Prior:   {stale_max:.2f}")
    print(f"  Full ANALYZE:  {full_max:.2f}")


def generate_latex_table(baseline: Dict, stale: Dict, full_analyze: Dict, output_file: Path):
    """生成 LaTeX 表格"""
    # 执行时间
    baseline_times = [r['execution_time_ms'] for r in baseline['results'] if r['status'] == 'success']
    stale_times = [r['execution_time_ms'] for r in stale['results'] if r['status'] == 'success']
    full_times = [r['execution_time_ms'] for r in full_analyze['results'] if r['status'] == 'success']

    baseline_total = sum(baseline_times) / 1000
    stale_total = sum(stale_times) / 1000
    full_total = sum(full_times) / 1000

    # Q-error
    def extract_qerrors(results: Dict) -> List[float]:
        qerrors = []
        for r in results['results']:
            if r['status'] == 'success' and r.get('qerror'):
                qerrors.append(r['qerror']['mean'])
        return qerrors

    baseline_qerrors = extract_qerrors(baseline)
    stale_qerrors = extract_qerrors(stale)
    full_qerrors = extract_qerrors(full_analyze)

    baseline_qerror = sum(baseline_qerrors) / len(baseline_qerrors) if baseline_qerrors else 0
    stale_qerror = sum(stale_qerrors) / len(stale_qerrors) if stale_qerrors else 0
    full_qerror = sum(full_qerrors) / len(full_qerrors) if full_qerrors else 0

    # 生成 LaTeX 表格
    latex = f"""
% Execution Time Table
\\begin{{table}}[!htb]
  \\centering
  \\caption{{PostgreSQL JOB benchmark execution time (113 queries, IMDB dataset).}}
  \\label{{tab:postgres_job_time}}
  \\begin{{tabular}}{{l r r}}
    \\toprule
    Method & Total time (s) & vs.~Baseline \\\\
    \\midrule
    Baseline       & {baseline_total:.1f} & — \\\\
    Stale Prior    & {stale_total:.1f} & {(stale_total/baseline_total - 1)*100:+.1f}\\% \\\\
    Full ANALYZE   & {full_total:.1f} & {(full_total/baseline_total - 1)*100:+.1f}\\% \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}

% Q-Error Table
\\begin{{table}}[!htb]
  \\centering
  \\caption{{PostgreSQL JOB benchmark Q-Error (mean across histogram-dependent operators).}}
  \\label{{tab:postgres_job_qerror}}
  \\begin{{tabular}}{{l r r}}
    \\toprule
    Method & Mean Q-Error & vs.~Baseline \\\\
    \\midrule
    Baseline       & {baseline_qerror:.2f} & — \\\\
    Stale Prior    & {stale_qerror:.2f} & {(stale_qerror/baseline_qerror - 1)*100:+.1f}\\% \\\\
    Full ANALYZE   & {full_qerror:.2f} & {(full_qerror/baseline_qerror - 1)*100:+.1f}\\% \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""

    with open(output_file, 'w') as f:
        f.write(latex)

    print(f"\n✓ LaTeX tables saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze PostgreSQL experiment results')
    parser.add_argument('--baseline', required=True, help='Baseline results JSON')
    parser.add_argument('--stale', required=True, help='Stale Prior results JSON')
    parser.add_argument('--full-analyze', required=True, help='Full ANALYZE results JSON')
    parser.add_argument('--output', required=True, help='Output comparison JSON')
    parser.add_argument('--latex', default=None, help='Output LaTeX tables file')

    args = parser.parse_args()

    # 加载结果
    baseline = load_results(Path(args.baseline))
    stale = load_results(Path(args.stale))
    full_analyze = load_results(Path(args.full_analyze))

    # 分析执行时间
    analyze_execution_time(baseline, stale, full_analyze)

    # 分析 Q-error
    analyze_qerror(baseline, stale, full_analyze)

    # 生成 LaTeX 表格
    if args.latex:
        generate_latex_table(baseline, stale, full_analyze, Path(args.latex))

    # 保存比较结果
    comparison = {
        'baseline': {
            'total_time_s': sum([r['execution_time_ms'] for r in baseline['results'] if r['status'] == 'success']) / 1000,
            'mean_qerror': sum([r['qerror']['mean'] for r in baseline['results'] if r['status'] == 'success' and r.get('qerror')]) / len([r for r in baseline['results'] if r['status'] == 'success' and r.get('qerror')])
        },
        'stale_prior': {
            'total_time_s': sum([r['execution_time_ms'] for r in stale['results'] if r['status'] == 'success']) / 1000,
            'mean_qerror': sum([r['qerror']['mean'] for r in stale['results'] if r['status'] == 'success' and r.get('qerror')]) / len([r for r in stale['results'] if r['status'] == 'success' and r.get('qerror')])
        },
        'full_analyze': {
            'total_time_s': sum([r['execution_time_ms'] for r in full_analyze['results'] if r['status'] == 'success']) / 1000,
            'mean_qerror': sum([r['qerror']['mean'] for r in full_analyze['results'] if r['status'] == 'success' and r.get('qerror')]) / len([r for r in full_analyze['results'] if r['status'] == 'success' and r.get('qerror')])
        }
    }

    with open(args.output, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n✓ Comparison results saved to {args.output}")


if __name__ == '__main__':
    main()
