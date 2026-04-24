#!/usr/bin/env python3
"""
TPC-DS Per-Query Analysis Script

Analyzes per-query execution time differences between Stale/OASIS/Full ANALYZE.
Produces:
  - Per-query time breakdown table
  - Plan-change analysis
  - Q-error correlation analysis
"""
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_results(results_dir: Path) -> Dict[str, Dict]:
    """Load all strategy results from a directory."""
    strategies = {}
    for pattern in ["stale_prior", "histogram_only", "oasis", "full_analyze"]:
        json_file = results_dir / f"{pattern}_results.json"
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)
            strategies[pattern] = data
    return strategies


def compute_per_query_analysis(strategies: Dict[str, Dict]) -> List[Dict]:
    """Compute per-query analysis across strategies."""
    if "stale_prior" not in strategies:
        print("Warning: stale_prior results not found")
        return []

    stale_results = {r["query_id"]: r for r in strategies["stale_prior"]["results"]}

    rows = []
    for query_id in sorted(stale_results.keys()):
        stale = stale_results[query_id]
        row = {
            "query_id": query_id,
            "stale_time_ms": stale.get("execution_time_ms", 0),
            "stale_status": stale.get("status", "unknown"),
        }

        for strategy_name in ["histogram_only", "oasis", "full_analyze"]:
            if strategy_name in strategies:
                strat_results = {r["query_id"]: r for r in strategies[strategy_name]["results"]}
                if query_id in strat_results:
                    r = strat_results[query_id]
                    row[f"{strategy_name}_time_ms"] = r.get("execution_time_ms", 0)
                    row[f"{strategy_name}_status"] = r.get("status", "unknown")

                    # Compute improvement vs stale
                    stale_t = row["stale_time_ms"]
                    strat_t = r.get("execution_time_ms", 0)
                    if stale_t > 0:
                        row[f"{strategy_name}_improvement_pct"] = (stale_t - strat_t) / stale_t * 100

                    # Q-error comparison
                    if stale.get("qerror") and r.get("qerror"):
                        stale_qe = stale["qerror"].get("geometric_mean", 1.0)
                        strat_qe = r["qerror"].get("geometric_mean", 1.0)
                        row[f"{strategy_name}_qerror"] = strat_qe
                        row[f"{strategy_name}_qerror_improvement"] = (stale_qe - strat_qe) / max(stale_qe, 1e-6) * 100

        rows.append(row)

    return rows


def print_per_query_table(rows: List[Dict]) -> None:
    """Print formatted per-query analysis table."""
    # Header
    print(f"\n{'Query':<10} {'Stale(ms)':>10} {'OASIS(ms)':>10} {'Analyze(ms)':>11} "
          f"{'OASIS Δ%':>9} {'Anal. Δ%':>9} {'OASIS QE':>9}")
    print("-" * 78)

    total_stale = 0
    total_oasis = 0
    total_analyze = 0
    oasis_wins = 0
    analyze_wins = 0
    total_queries = 0

    for row in rows:
        stale_t = row.get("stale_time_ms", 0)
        oasis_t = row.get("histogram_only_time_ms") or row.get("oasis_time_ms", 0)
        analyze_t = row.get("full_analyze_time_ms", 0)

        if row["stale_status"] != "success":
            continue

        total_queries += 1
        total_stale += stale_t
        total_oasis += oasis_t
        total_analyze += analyze_t

        oasis_imp = row.get("histogram_only_improvement_pct") or row.get("oasis_improvement_pct", 0) or 0
        analyze_imp = row.get("full_analyze_improvement_pct", 0) or 0
        oasis_qe = row.get("histogram_only_qerror") or row.get("oasis_qerror", 1.0) or 1.0

        if oasis_imp and oasis_imp > 0:
            oasis_wins += 1
        if analyze_imp and analyze_imp > 0:
            analyze_wins += 1

        print(f"{row['query_id']:<10} {stale_t:>10.0f} {oasis_t:>10.0f} {analyze_t:>11.0f} "
              f"{oasis_imp:>+8.1f}% {analyze_imp:>+8.1f}% {oasis_qe:>9.2f}")

    print("-" * 78)
    if total_queries > 0:
        print(f"{'TOTAL':<10} {total_stale:>10.0f} {total_oasis:>10.0f} {total_analyze:>11.0f} "
              f"{(total_stale - total_oasis)/total_stale*100:>+8.1f}% "
              f"{(total_stale - total_analyze)/total_stale*100:>+8.1f}%")
        print(f"\nQueries improved by OASIS: {oasis_wins}/{total_queries} ({oasis_wins/total_queries*100:.0f}%)")
        print(f"Queries improved by ANALYZE: {analyze_wins}/{total_queries} ({analyze_wins/total_queries*100:.0f}%)")

        # Recovery fraction
        oasis_delta = total_stale - total_oasis
        analyze_delta = total_stale - total_analyze
        if abs(analyze_delta) > 1e-6:
            recovery = oasis_delta / analyze_delta * 100
            print(f"OASIS recovery of ANALYZE improvement: {recovery:.1f}%")


def classify_queries(rows: List[Dict]) -> Dict:
    """Classify queries into improvement categories."""
    categories = {
        "big_win_oasis": [],      # OASIS improves > 10%
        "small_win_oasis": [],    # OASIS improves 0-10%
        "neutral_oasis": [],      # OASIS within +/- 2%
        "regression_oasis": [],   # OASIS worsens > 2%
        "big_win_analyze": [],
        "regression_analyze": [],
    }

    for row in rows:
        if row["stale_status"] != "success":
            continue

        oasis_imp = row.get("histogram_only_improvement_pct") or row.get("oasis_improvement_pct", 0) or 0
        analyze_imp = row.get("full_analyze_improvement_pct", 0) or 0

        if oasis_imp > 10:
            categories["big_win_oasis"].append(row["query_id"])
        elif oasis_imp > 2:
            categories["small_win_oasis"].append(row["query_id"])
        elif oasis_imp > -2:
            categories["neutral_oasis"].append(row["query_id"])
        else:
            categories["regression_oasis"].append(row["query_id"])

        if analyze_imp > 10:
            categories["big_win_analyze"].append(row["query_id"])
        elif analyze_imp < -2:
            categories["regression_analyze"].append(row["query_id"])

    return categories


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TPC-DS Per-Query Analysis")
    parser.add_argument("--results-dir", type=Path, required=True,
                       help="Directory containing *_results.json files")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output JSON file for per-query analysis")
    args = parser.parse_args()

    strategies = load_results(args.results_dir)
    print(f"Loaded strategies: {list(strategies.keys())}")

    rows = compute_per_query_analysis(strategies)
    print_per_query_table(rows)

    categories = classify_queries(rows)
    print(f"\n=== Query Classification ===")
    print(f"OASIS big wins (>10%): {len(categories['big_win_oasis'])} — {categories['big_win_oasis'][:10]}")
    print(f"OASIS small wins (2-10%): {len(categories['small_win_oasis'])} — {categories['small_win_oasis'][:10]}")
    print(f"OASIS neutral (±2%): {len(categories['neutral_oasis'])}")
    print(f"OASIS regressions (<-2%): {len(categories['regression_oasis'])} — {categories['regression_oasis'][:10]}")
    print(f"Full ANALYZE big wins: {len(categories['big_win_analyze'])}")
    print(f"Full ANALYZE regressions: {len(categories['regression_analyze'])}")

    if args.output:
        output_data = {
            "per_query": rows,
            "categories": categories,
            "summary": {
                "total_queries": len(rows),
                "oasis_wins": len(categories["big_win_oasis"]) + len(categories["small_win_oasis"]),
                "oasis_regressions": len(categories["regression_oasis"]),
            }
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
