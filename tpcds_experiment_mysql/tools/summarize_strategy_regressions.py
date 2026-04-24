#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    return {row['query_id']: row for row in payload.get('results', [])}


def normalize_plan(plan_raw: Any) -> str:
    if plan_raw in (None, ''):
        return '<no plan>\n'
    if not isinstance(plan_raw, str):
        return json.dumps(plan_raw, indent=2, ensure_ascii=False, sort_keys=True) + '\n'
    try:
        parsed = json.loads(plan_raw)
        return json.dumps(parsed, indent=2, ensure_ascii=False, sort_keys=True) + '\n'
    except Exception:
        text = plan_raw.strip('\n')
        return text + ('\n' if text else '')


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare stale-prior results with another strategy.')
    parser.add_argument('--stale-results', required=True)
    parser.add_argument('--compare-results', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--threshold', type=float, default=0.20)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--diff-lines', type=int, default=60)
    args = parser.parse_args()

    stale = load_results(Path(args.stale_results))
    compare = load_results(Path(args.compare_results))
    output_dir = Path(args.output_dir)
    diff_dir = output_dir / 'plan_diffs' / args.label
    diff_dir.mkdir(parents=True, exist_ok=True)

    comparable = 0
    stale_faster: list[tuple[str, int, int, float]] = []
    compare_faster: list[tuple[str, int, int, float]] = []
    major_regressions: list[tuple[str, int, int, float, Path, str]] = []
    compare_failed: list[tuple[str, Any, Any, Any]] = []

    for query_id in sorted(set(stale) & set(compare)):
        stale_row = stale[query_id]
        compare_row = compare[query_id]
        stale_ok = stale_row.get('status') == 'success' and stale_row.get('execution_time_ms') is not None
        compare_ok = compare_row.get('status') == 'success' and compare_row.get('execution_time_ms') is not None

        if stale_ok and compare_ok:
            comparable += 1
            stale_ms = stale_row['execution_time_ms']
            compare_ms = compare_row['execution_time_ms']
            if stale_ms < compare_ms:
                ratio = ((compare_ms - stale_ms) / stale_ms) if stale_ms > 0 else float('inf')
                stale_faster.append((query_id, stale_ms, compare_ms, ratio))
                if ratio >= args.threshold:
                    stale_plan = normalize_plan(stale_row.get('plan_raw'))
                    compare_plan = normalize_plan(compare_row.get('plan_raw'))
                    diff_text = ''.join(
                        difflib.unified_diff(
                            stale_plan.splitlines(keepends=True),
                            compare_plan.splitlines(keepends=True),
                            fromfile=f'{query_id}:stale_prior',
                            tofile=f'{query_id}:{args.label}',
                        )
                    )
                    diff_path = diff_dir / f'{query_id}.diff'
                    diff_path.write_text(diff_text or '(plans are identical after normalization)\n')
                    major_regressions.append((query_id, stale_ms, compare_ms, ratio, diff_path, diff_text))
            elif compare_ms < stale_ms:
                ratio = ((stale_ms - compare_ms) / stale_ms) if stale_ms > 0 else 0.0
                compare_faster.append((query_id, stale_ms, compare_ms, ratio))
        elif stale_ok and not compare_ok:
            compare_failed.append((query_id, stale_row.get('execution_time_ms'), compare_row.get('status'), compare_row.get('error')))

    print('')
    print(f'Comparison vs stale_prior: {args.label}')
    print('-' * 70)
    print(f'Comparable successful queries: {comparable}')
    print(f'Stale prior faster queries:    {len(stale_faster)}')
    print(f'{args.label} faster queries:      {len(compare_faster)}')
    print(f'{args.label} failed while stale succeeded: {len(compare_failed)}')
    print(f'Stale faster by >= {int(args.threshold * 100)}%: {len(major_regressions)}')

    if stale_faster:
        top = ', '.join(
            f"{query_id} ({stale_ms}ms -> {compare_ms}ms, +{ratio*100:.1f}%)"
            for query_id, stale_ms, compare_ms, ratio in stale_faster[:10]
        )
        print(f'Top stale-faster queries: {top}')

    if compare_failed:
        failed_preview = ', '.join(
            f"{query_id} ({status})" for query_id, _, status, _ in compare_failed[:10]
        )
        print(f'Compare-side failures: {failed_preview}')

    for query_id, stale_ms, compare_ms, ratio, diff_path, diff_text in major_regressions:
        print('')
        print(f'[Plan Diff] {query_id}: stale {stale_ms}ms vs {args.label} {compare_ms}ms (+{ratio*100:.1f}%)')
        print(f'Full diff saved to: {diff_path}')
        lines = diff_text.splitlines()
        preview = lines[:args.diff_lines] if lines else ['(no diff)']
        for line in preview:
            print(line)
        if len(lines) > args.diff_lines:
            print(f'... ({len(lines) - args.diff_lines} more lines)')

    summary = {
        'label': args.label,
        'comparable_successes': comparable,
        'stale_faster_count': len(stale_faster),
        'compare_faster_count': len(compare_faster),
        'compare_failed_while_stale_succeeded': len(compare_failed),
        'major_regressions': [
            {
                'query_id': query_id,
                'stale_ms': stale_ms,
                'compare_ms': compare_ms,
                'slowdown_ratio': ratio,
                'diff_path': str(diff_path),
            }
            for query_id, stale_ms, compare_ms, ratio, diff_path, _ in major_regressions
        ],
    }
    (output_dir / f'{args.label}_vs_stale_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )


if __name__ == '__main__':
    main()
