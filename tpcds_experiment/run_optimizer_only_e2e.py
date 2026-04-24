#!/usr/bin/env python3
"""
Optimizer-Only End-to-End Experiment
=====================================

Uses PostgreSQL EXPLAIN (WITHOUT ANALYZE) to evaluate how OASIS-corrected
statistics change optimizer plan choices — without executing any queries.

Key insight: EXPLAIN invokes the full optimizer pipeline (statistics lookup →
selectivity estimation → cost model → plan enumeration → plan output) but
does NOT scan tables or execute the query. This isolates the optimizer's
decision-making from execution overhead.

Protocol:
  1. Load TPC-DS, run ANALYZE → capture baseline (fresh) statistics
  2. Apply drift via DML, capture stale statistics
  3. Collect feedback observations during drift
  4. Run OASIS correction → produce corrected statistics
  5. For each TPC-DS query, under each statistics state (stale/OASIS/fresh):
     a. Inject statistics via pg_statistic UPDATE
     b. Run EXPLAIN (FORMAT JSON) — optimizer only, no execution
     c. Capture: plan JSON, estimated costs, estimated rows, node structure
  6. Compare plans across statistics states:
     - Plan structure diff (join order, scan type, join algorithm)
     - Estimated cost deltas
     - Per-node estimated row count deltas
     - Plan change rate

Metrics (all optimizer-perspective, no wall-clock execution):
  - Plan Change Rate: % queries where plan JSON differs
  - Cost Improvement: % reduction in optimizer-estimated total cost
  - Row Estimate Correction: per-node Q-Error of estimated vs actual rows
    (actual rows from a separate ground-truth ANALYZE run)
  - Join Order Stability: % queries where join order changes
  - Scan Type Changes: Seq Scan ↔ Index Scan transitions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_PIPELINE_DIR = _REPO_DIR / "cdf_kll_ml_pipeline"

if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

# ── PostgreSQL support ────────────────────────────────────────────────────
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PlanNode:
    """Normalized plan node for comparison."""
    node_type: str
    relation_name: str = ""
    join_type: str = ""
    index_name: str = ""
    filter_text: str = ""
    estimated_rows: float = 0.0
    estimated_cost: float = 0.0
    children: List[PlanNode] = field(default_factory=list)

    @classmethod
    def from_json(cls, node: dict) -> "PlanNode":
        children = [cls.from_json(c) for c in node.get("Plans", [])]
        return cls(
            node_type=node.get("Node Type", ""),
            relation_name=node.get("Relation Name", ""),
            join_type=node.get("Join Type", ""),
            index_name=node.get("Index Name", ""),
            filter_text=node.get("Filter", ""),
            estimated_rows=node.get("Plan Rows", 0),
            estimated_cost=node.get("Total Cost", 0),
            children=children,
        )

    def structural_hash(self) -> str:
        """Hash capturing the plan's STRUCTURE (types, relations, join types)
        but NOT estimated costs/rows. Used to detect plan changes."""
        parts = [self.node_type]
        if self.relation_name:
            parts.append(f"rel={self.relation_name}")
        if self.join_type:
            parts.append(f"join={self.join_type}")
        if self.index_name:
            parts.append(f"idx={self.index_name}")
        parts.append(f"children=[{','.join(c.structural_hash() for c in self.children)}]")
        return "|".join(parts)

    def cost_profile(self) -> Dict[str, float]:
        """Extract per-node estimated costs and rows."""
        profile = {}
        for i, child in enumerate(self.children):
            prefix = f"{self.node_type}"
            if self.relation_name:
                prefix += f"({self.relation_name})"
            profile[f"{prefix}.child{i}.cost"] = child.estimated_cost
            profile[f"{prefix}.child{i}.rows"] = child.estimated_rows
            profile.update(child.cost_profile())
        return profile


@dataclass
class QueryPlanEval:
    """Per-query plan evaluation under one statistics state."""
    query_id: str
    stats_state: str  # "stale" | "oasis" | "fresh"
    success: bool = False
    error: str = ""
    # Plan-level metrics
    total_estimated_cost: float = 0.0
    total_estimated_rows: float = 0.0  # root node
    plan_json: Optional[dict] = None
    plan_structure_hash: str = ""
    plan_flat_nodes: int = 0
    # Per-node
    scan_nodes: List[dict] = field(default_factory=list)
    join_nodes: List[dict] = field(default_factory=list)
    # Timing
    planning_time_ms: float = 0.0


@dataclass
class QueryComparison:
    """Comparison of plan across two statistics states."""
    query_id: str
    state_a: str
    state_b: str
    # Plan structure
    plans_identical: bool = True
    plan_structure_changed: bool = False
    # Estimated cost
    cost_a: float = 0.0
    cost_b: float = 0.0
    cost_delta_pct: float = 0.0
    # Estimated rows
    rows_a: float = 0.0
    rows_b: float = 0.0
    rows_delta_pct: float = 0.0
    # Scan type changes
    scan_changes: List[str] = field(default_factory=list)
    # Join order changes
    join_order_changed: bool = False


@dataclass
class ExperimentReport:
    """Full experiment report."""
    total_queries: int = 0
    successful_queries: int = 0
    # Plan change rates
    stale_vs_oasis_plan_change_rate: float = 0.0
    stale_vs_fresh_plan_change_rate: float = 0.0
    oasis_vs_fresh_plan_change_rate: float = 0.0
    # Cost improvements (optimizer-estimated)
    avg_cost_improvement_oasis_vs_stale_pct: float = 0.0
    avg_cost_improvement_fresh_vs_stale_pct: float = 0.0
    avg_cost_recovery_pct: float = 0.0  # oasis / fresh
    # Row estimate improvements
    avg_row_delta_oasis_vs_stale_pct: float = 0.0
    # Scan type transitions
    scan_transitions: Dict[str, int] = field(default_factory=dict)
    # Per-query details
    per_query: List[QueryComparison] = field(default_factory=list)
    # Summary
    plan_wins: int = 0
    plan_neutral: int = 0
    plan_losses: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Optimizer-Only Experiment Runner
# ═══════════════════════════════════════════════════════════════════════════

class OptimizerOnlyE2E:
    """Run optimizer-only E2E: inject stats → EXPLAIN → compare plans.

    Uses EXPLAIN (FORMAT JSON) WITHOUT ANALYZE — invokes the full optimizer
    but never executes queries. This gives us the optimizer's plan choices
    and cost estimates under different statistics without table scans.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5433,
        dbname: str = "tpcds",
        user: str = "postgres",
        password: str = "",
    ):
        if not _HAS_PSYCOPG2:
            raise ImportError("psycopg2 required: pip install psycopg2-binary")
        self.conn_params = {
            "host": host, "port": port, "dbname": dbname,
            "user": user, "password": password,
        }
        self._conn = None
        self._backup_stats: Dict[str, dict] = {}

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self.conn_params)
            self._conn.autocommit = True
        return self._conn

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    # ── Statistics Management ───────────────────────────────────────────

    def get_histogram_bounds(
        self, table_name: str, column_name: str
    ) -> Optional[List[float]]:
        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT histogram_bounds FROM pg_stats "
            "WHERE tablename = %s AND attname = %s",
            (table_name.lower(), column_name.lower()),
        )
        row = cur.fetchone()
        cur.close()
        if not row or not row["histogram_bounds"]:
            return None
        return list(row["histogram_bounds"])

    def get_column_stats(
        self, table_name: str, column_name: str
    ) -> Optional[dict]:
        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT s.starelid, s.staattnum, s.stanullfrac, s.stawidth,
                   s.stadistinct,
                   s.stakind1, s.stakind2, s.stakind3, s.stakind4, s.stakind5,
                   s.stanumbers1, s.stanumbers2, s.stanumbers3, s.stanumbers4, s.stanumbers5,
                   s.stavalues1, s.stavalues2, s.stavalues3, s.stavalues4, s.stavalues5
            FROM pg_statistic s
            JOIN pg_class c ON c.oid = s.starelid
            JOIN pg_attribute a ON a.attrelid = s.starelid AND a.attnum = s.staattnum
            WHERE c.relname = %s AND a.attname = %s
        """, (table_name.lower(), column_name.lower()))
        row = cur.fetchone()
        cur.close()
        return dict(row) if row else None

    def backup_stats(self, table_name: str, column_name: str) -> dict:
        stats = self.get_column_stats(table_name, column_name)
        if stats:
            key = f"{table_name}.{column_name}"
            self._backup_stats[key] = {
                "starelid": stats["starelid"],
                "staattnum": stats["staattnum"],
                "stanullfrac": stats["stanullfrac"],
                "stawidth": stats["stawidth"],
                "stadistinct": stats["stadistinct"],
                "stakind": [stats[f"stakind{i}"] for i in range(1, 6)],
                "stanumbers": [stats[f"stanumbers{i}"] for i in range(1, 6)],
                "stavalues": [stats[f"stavalues{i}"] for i in range(1, 6)],
            }
        return stats or {}

    def restore_stats(self, table_name: str, column_name: str) -> bool:
        key = f"{table_name}.{column_name}"
        if key not in self._backup_stats:
            return False
        backup = self._backup_stats[key]
        cur = self.conn.cursor()
        cur.execute(
            "DELETE FROM pg_statistic WHERE starelid = %s AND staattnum = %s",
            (backup["starelid"], backup["staattnum"]),
        )
        columns = ["starelid", "staattnum", "stanullfrac", "stawidth", "stadistinct"]
        values = [
            backup["starelid"], backup["staattnum"],
            backup["stanullfrac"], backup["stawidth"], backup["stadistinct"],
        ]
        for i in range(1, 6):
            columns.extend([f"stakind{i}", f"stanumbers{i}", f"stavalues{i}"])
            values.extend([
                backup["stakind"][i - 1],
                backup["stanumbers"][i - 1],
                backup["stavalues"][i - 1],
            ])
        placeholders = ", ".join(["%s"] * len(values))
        cur.execute(
            f"INSERT INTO pg_statistic ({', '.join(columns)}) VALUES ({placeholders})",
            values,
        )
        cur.close()
        return True

    def update_histogram_bounds(
        self, table_name: str, column_name: str, new_bounds: List[float],
    ) -> bool:
        """Update histogram_bounds in pg_statistic."""
        stats = self.get_column_stats(table_name, column_name)
        if not stats:
            return False

        starelid = stats["starelid"]
        staattnum = stats["staattnum"]

        # Find histogram slot (stakind = 2 for histogram_bounds in pg_stats view,
        # but in pg_statistic it's stored as kind 2 with values array)
        hist_slot = None
        for i in range(1, 6):
            if stats[f"stakind{i}"] == 2:
                hist_slot = i
                break
        if hist_slot is None:
            return False

        cur = self.conn.cursor()
        bounds_array = "ARRAY[" + ",".join(str(v) for v in new_bounds) + "]"
        cur.execute(f"""
            UPDATE pg_statistic
            SET stavalues{hist_slot} = {bounds_array}::float8[]
            WHERE starelid = %s AND staattnum = %s
        """, (starelid, staattnum))
        cur.close()
        return True

    def update_null_frac(
        self, table_name: str, column_name: str, null_frac: float,
    ) -> bool:
        stats = self.get_column_stats(table_name, column_name)
        if not stats:
            return False
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE pg_statistic SET stanullfrac = %s "
            "WHERE starelid = %s AND staattnum = %s",
            (null_frac, stats["starelid"], stats["staattnum"]),
        )
        cur.close()
        return True

    def run_analyze(self, table_name: str, column_name: str = ""):
        cur = self.conn.cursor()
        if column_name:
            cur.execute(f"ANALYZE {table_name}({column_name})")
        else:
            cur.execute(f"ANALYZE {table_name}")
        cur.close()

    # ── EXPLAIN (Optimizer-Only) ────────────────────────────────────────

    def run_explain(
        self, sql: str, timeout_s: int = 30,
    ) -> QueryPlanEval:
        """Run EXPLAIN (FORMAT JSON) WITHOUT ANALYZE.

        This invokes the FULL optimizer pipeline:
          statistics lookup → selectivity estimation → cost estimation
          → plan enumeration → plan output

        But NEVER executes the query. No table scans, no I/O.

        Returns structured plan evaluation.
        """
        result = QueryPlanEval(
            query_id="", stats_state="", success=False,
        )

        cur = self.conn.cursor(cursor_factory=RealDictCursor)
        try:
            cur.execute(f"SET statement_timeout = '{timeout_s}s'")
            cur.execute("SET max_parallel_workers_per_gather = 0")

            start = time.perf_counter()
            # KEY: EXPLAIN without ANALYZE = optimizer only
            cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
            planning_time = (time.perf_counter() - start) * 1000

            rows = cur.fetchall()
            if rows:
                plan_data = rows[0]
                if isinstance(plan_data, dict):
                    plan_json = (
                        plan_data.get("QUERY PLAN", plan_data)
                        if "QUERY PLAN" in plan_data
                        else plan_data
                    )
                else:
                    plan_json = plan_data

                # Normalize: plan_json may be list or dict
                if isinstance(plan_json, str):
                    plan_json = json.loads(plan_json)
                if isinstance(plan_json, list) and plan_json:
                    root = plan_json[0].get("Plan", plan_json[0])
                elif isinstance(plan_json, dict):
                    root = plan_json.get("Plan", plan_json)
                else:
                    root = {}

                plan_node = PlanNode.from_json(root)

                result.success = True
                result.plan_json = root
                result.total_estimated_cost = root.get("Total Cost", 0)
                result.total_estimated_rows = root.get("Plan Rows", 0)
                result.plan_structure_hash = plan_node.structural_hash()
                result.planning_time_ms = planning_time

                # Extract scan and join nodes
                result.scan_nodes = self._extract_scans(root)
                result.join_nodes = self._extract_joins(root)
                result.plan_flat_nodes = self._count_nodes(root)

        except Exception as e:
            result.success = False
            result.error = str(e)[:300]
            try:
                self.conn.rollback()
            except Exception:
                pass
        finally:
            cur.close()

        return result

    def _extract_scans(self, node: dict) -> List[dict]:
        scan_types = {
            "Seq Scan", "Index Scan", "Index Only Scan",
            "Bitmap Heap Scan", "Bitmap Index Scan",
        }
        scans = []
        node_type = node.get("Node Type", "")
        if node_type in scan_types:
            scans.append({
                "node_type": node_type,
                "relation": node.get("Relation Name", ""),
                "index": node.get("Index Name", ""),
                "filter": node.get("Filter", ""),
                "estimated_rows": node.get("Plan Rows", 0),
                "estimated_cost": node.get("Total Cost", 0),
            })
        for child in node.get("Plans", []):
            scans.extend(self._extract_scans(child))
        return scans

    def _extract_joins(self, node: dict) -> List[dict]:
        join_types = {
            "Hash Join", "Merge Join", "Nested Loop",
        }
        joins = []
        node_type = node.get("Node Type", "")
        if node_type in join_types:
            joins.append({
                "node_type": node_type,
                "join_type": node.get("Join Type", ""),
                "estimated_rows": node.get("Plan Rows", 0),
                "estimated_cost": node.get("Total Cost", 0),
            })
        for child in node.get("Plans", []):
            joins.extend(self._extract_joins(child))
        return joins

    def _count_nodes(self, node: dict) -> int:
        return 1 + sum(self._count_nodes(c) for c in node.get("Plans", []))

    # ── Plan Comparison ─────────────────────────────────────────────────

    def compare_plans(
        self, plan_a: QueryPlanEval, plan_b: QueryPlanEval,
    ) -> QueryComparison:
        """Compare two plans and classify differences."""
        cmp = QueryComparison(
            query_id=plan_a.query_id,
            state_a=plan_a.stats_state,
            state_b=plan_b.stats_state,
        )

        if not plan_a.success or not plan_b.success:
            cmp.plans_identical = False
            return cmp

        # Structure comparison
        cmp.plans_identical = (
            plan_a.plan_structure_hash == plan_b.plan_structure_hash
        )
        cmp.plan_structure_changed = not cmp.plans_identical

        # Cost comparison
        cmp.cost_a = plan_a.total_estimated_cost
        cmp.cost_b = plan_b.total_estimated_cost
        if plan_a.total_estimated_cost > 0:
            cmp.cost_delta_pct = (
                (plan_a.total_estimated_cost - plan_b.total_estimated_cost)
                / plan_a.total_estimated_cost * 100
            )

        # Row estimate comparison
        cmp.rows_a = plan_a.total_estimated_rows
        cmp.rows_b = plan_b.total_estimated_rows
        if plan_a.total_estimated_rows > 0:
            cmp.rows_delta_pct = (
                (plan_a.total_estimated_rows - plan_b.total_estimated_rows)
                / plan_a.total_estimated_rows * 100
            )

        # Scan type changes
        scans_a = {(s["relation"], s["node_type"]) for s in plan_a.scan_nodes}
        scans_b = {(s["relation"], s["node_type"]) for s in plan_b.scan_nodes}
        for rel, stype in scans_a - scans_b:
            stype_b = next(
                (s["node_type"] for s in plan_b.scan_nodes if s["relation"] == rel),
                "REMOVED",
            )
            cmp.scan_changes.append(f"{rel}: {stype} → {stype_b}")
        for rel, stype in scans_b - scans_a:
            stype_a = next(
                (s["node_type"] for s in plan_a.scan_nodes if s["relation"] == rel),
                "NONE",
            )
            cmp.scan_changes.append(f"{rel}: {stype_a} → {stype}")

        # Join order change detection
        joins_a = [(j["node_type"], j.get("join_type", "")) for j in plan_a.join_nodes]
        joins_b = [(j["node_type"], j.get("join_type", "")) for j in plan_b.join_nodes]
        cmp.join_order_changed = (joins_a != joins_b)

        return cmp

    # ── Full Experiment ─────────────────────────────────────────────────

    def run_full_experiment(
        self,
        queries: Dict[str, str],
        target_columns: List[Tuple[str, str]],  # (table, column)
        oasis_corrections: Dict[str, List[float]],  # column_key → corrected bounds
        oasis_null_fracs: Dict[str, float] = None,
    ) -> ExperimentReport:
        """Run the full optimizer-only experiment.

        Phases:
          1. Snapshot fresh stats (after ANALYZE)
          2. Run EXPLAIN under fresh stats → baseline plans
          3. Restore stale stats → run EXPLAIN → stale plans
          4. Inject OASIS-corrected stats → run EXPLAIN → OASIS plans
          5. Compare plans across all three states
        """
        report = ExperimentReport(total_queries=len(queries))
        oasis_null_fracs = oasis_null_fracs or {}

        # Phase 1: Snapshot fresh statistics
        print("── Phase 1: Snapshotting fresh statistics ──")
        fresh_bounds_all = {}
        for table, col in target_columns:
            self.backup_stats(table, col)
            self.run_analyze(table, col)
            bounds = self.get_histogram_bounds(table, col)
            fresh_bounds_all[f"{table}.{col}"] = bounds
            print(f"  {table}.{col}: {len(bounds) if bounds else 0} bounds (fresh)")

        # Phase 2: EXPLAIN under fresh stats
        print("\n── Phase 2: EXPLAIN under FRESH statistics ──")
        fresh_plans: Dict[str, QueryPlanEval] = {}
        for qid, sql in sorted(queries.items()):
            plan = self.run_explain(sql)
            plan.query_id = qid
            plan.stats_state = "fresh"
            fresh_plans[qid] = plan
            status = "OK" if plan.success else f"ERR: {plan.error[:40]}"
            print(f"  [{qid}] fresh  cost={plan.total_estimated_cost:.1f}  "
                  f"rows={plan.total_estimated_rows:.0f}  [{status}]")

        # Phase 3: Restore stale + EXPLAIN under stale stats
        print("\n── Phase 3: Restoring stale statistics ──")
        for table, col in target_columns:
            self.restore_stats(table, col)
            bounds = self.get_histogram_bounds(table, col)
            print(f"  {table}.{col}: {len(bounds) if bounds else 0} bounds (stale)")

        print("\n── Phase 4: EXPLAIN under STALE statistics ──")
        stale_plans: Dict[str, QueryPlanEval] = {}
        for qid, sql in sorted(queries.items()):
            plan = self.run_explain(sql)
            plan.query_id = qid
            plan.stats_state = "stale"
            stale_plans[qid] = plan
            status = "OK" if plan.success else f"ERR: {plan.error[:40]}"
            print(f"  [{qid}] stale cost={plan.total_estimated_cost:.1f}  "
                  f"rows={plan.total_estimated_rows:.0f}  [{status}]")

        # Phase 5: Inject OASIS + EXPLAIN under OASIS stats
        print("\n── Phase 5: Injecting OASIS-corrected statistics ──")
        for table, col in target_columns:
            key = f"{table}.{col}"
            if key in oasis_corrections:
                success = self.update_histogram_bounds(
                    table, col, oasis_corrections[key],
                )
                print(f"  {key}: histogram updated: {success}")
            if key in oasis_null_fracs:
                self.update_null_frac(table, col, oasis_null_fracs[key])
                print(f"  {key}: null_frac updated: {oasis_null_fracs[key]:.4f}")

        print("\n── Phase 6: EXPLAIN under OASIS statistics ──")
        oasis_plans: Dict[str, QueryPlanEval] = {}
        for qid, sql in sorted(queries.items()):
            plan = self.run_explain(sql)
            plan.query_id = qid
            plan.stats_state = "oasis"
            oasis_plans[qid] = plan
            status = "OK" if plan.success else f"ERR: {plan.error[:40]}"
            print(f"  [{qid}] oasis cost={plan.total_estimated_cost:.1f}  "
                  f"rows={plan.total_estimated_rows:.0f}  [{status}]")

        # Phase 7: Compare plans
        print("\n── Phase 7: Plan Comparison ──")
        for qid in sorted(queries.keys()):
            s = stale_plans.get(qid)
            o = oasis_plans.get(qid)
            f = fresh_plans.get(qid)

            if not (s and s.success and o and o.success):
                continue
            report.successful_queries += 1

            # Stale vs OASIS
            so_cmp = self.compare_plans(s, o)
            so_cmp.query_id = qid
            report.per_query.append(so_cmp)

            # Classify
            if so_cmp.plan_structure_changed:
                # Plan changed. Is it toward fresh?
                if f and f.success:
                    f_cmp = self.compare_plans(s, f)
                    o_f_cmp = self.compare_plans(o, f)
                    if o_f_cmp.plans_identical and not so_cmp.plans_identical:
                        report.plan_wins += 1  # OASIS matches fresh, stale doesn't
                    elif so_cmp.cost_delta_pct > 5:
                        report.plan_wins += 1  # OASIS reduced cost
                    else:
                        report.plan_neutral += 1
                else:
                    report.plan_neutral += 1
            elif so_cmp.cost_delta_pct > 5:
                report.plan_wins += 1  # Same plan but cheaper
            elif so_cmp.cost_delta_pct < -5:
                report.plan_losses += 1  # Same plan but more expensive
            else:
                report.plan_neutral += 1

            # Scan transitions
            for sc in so_cmp.scan_changes:
                report.scan_transitions[sc] = report.scan_transitions.get(sc, 0) + 1

            # Print per-query summary
            changed = "CHANGED" if so_cmp.plan_structure_changed else "same"
            print(
                f"  [{qid}] {changed:>7s}  "
                f"cost: {so_cmp.cost_a:.0f}→{so_cmp.cost_b:.0f} "
                f"({so_cmp.cost_delta_pct:+.1f}%)  "
                f"rows: {so_cmp.rows_a:.0f}→{so_cmp.rows_b:.0f} "
                f"({so_cmp.rows_delta_pct:+.1f}%)"
                + (f"  scans: {so_cmp.scan_changes}" if so_cmp.scan_changes else "")
            )

        # Compute aggregate metrics
        if report.per_query:
            n = len(report.per_query)
            # Plan change rates
            report.stale_vs_oasis_plan_change_rate = (
                sum(1 for c in report.per_query if c.plan_structure_changed) / n * 100
            )
            report.avg_cost_improvement_oasis_vs_stale_pct = (
                sum(c.cost_delta_pct for c in report.per_query) / n
            )

        return report


# ═══════════════════════════════════════════════════════════════════════════
# Standalone Plan Comparator (no PostgreSQL needed for analysis)
# ═══════════════════════════════════════════════════════════════════════════

def compare_plan_json_files(
    stale_dir: Path, oasis_dir: Path, fresh_dir: Path,
) -> ExperimentReport:
    """Compare EXPLAIN plan JSON files exported from separate runs.

    This path doesn't require live PostgreSQL — just plan JSON files.
    """
    report = ExperimentReport()
    stale_files = sorted(stale_dir.glob("*.json"))
    report.total_queries = len(stale_files)

    for sf in stale_files:
        qid = sf.stem
        of = oasis_dir / sf.name
        ff = fresh_dir / sf.name

        if not of.exists():
            continue
        report.successful_queries += 1

        with open(sf) as f:
            s_plan = json.load(f)
        with open(of) as f:
            o_plan = json.load(f)

        s_root = s_plan[0].get("Plan", s_plan[0]) if isinstance(s_plan, list) else s_plan.get("Plan", s_plan)
        o_root = o_plan[0].get("Plan", o_plan[0]) if isinstance(o_plan, list) else o_plan.get("Plan", o_plan)

        s_node = PlanNode.from_json(s_root)
        o_node = PlanNode.from_json(o_root)

        changed = s_node.structural_hash() != o_node.structural_hash()
        s_cost = s_root.get("Total Cost", 0)
        o_cost = o_root.get("Total Cost", 0)
        cost_pct = (s_cost - o_cost) / max(s_cost, 1e-6) * 100

        cmp = QueryComparison(
            query_id=qid,
            state_a="stale", state_b="oasis",
            plan_structure_changed=changed,
            cost_a=s_cost, cost_b=o_cost,
            cost_delta_pct=cost_pct,
        )
        report.per_query.append(cmp)

    if report.per_query:
        n = len(report.per_query)
        report.stale_vs_oasis_plan_change_rate = (
            sum(1 for c in report.per_query if c.plan_structure_changed) / n * 100
        )
        report.avg_cost_improvement_oasis_vs_stale_pct = (
            sum(c.cost_delta_pct for c in report.per_query) / n
        )

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Report Formatting
# ═══════════════════════════════════════════════════════════════════════════

def print_report(report: ExperimentReport) -> str:
    """Generate a readable report."""
    lines = []
    lines.append("=" * 72)
    lines.append("  OPTIMIZER-ONLY E2E EVALUATION REPORT")
    lines.append("  (EXPLAIN without ANALYZE — optimizer only, no query execution)")
    lines.append("=" * 72)

    lines.append(f"\n  Total queries:           {report.total_queries}")
    lines.append(f"  Successful:              {report.successful_queries}")
    lines.append(f"  Plan change rate (S→O):  {report.stale_vs_oasis_plan_change_rate:.1f}%")
    lines.append(f"  Avg cost improvement:    {report.avg_cost_improvement_oasis_vs_stale_pct:+.1f}%")
    lines.append(f"  Avg row estimate change: {report.avg_row_delta_oasis_vs_stale_pct:+.1f}%")

    lines.append(f"\n  ── Plan Impact Classification ──")
    lines.append(f"  Wins:     {report.plan_wins}")
    lines.append(f"  Neutral:  {report.plan_neutral}")
    lines.append(f"  Losses:   {report.plan_losses}")

    if report.scan_transitions:
        lines.append(f"\n  ── Scan Type Transitions ──")
        for trans, count in sorted(report.scan_transitions.items(), key=lambda x: -x[1]):
            lines.append(f"  {trans}: {count} queries")

    lines.append(f"\n  ── Per-Query Details ──")
    lines.append(f"  {'Query':<25s} {'Plan':>7s} {'Cost Δ%':>8s} {'Rows Δ%':>8s}  Scan Changes")
    lines.append(f"  {'─'*70}")
    for c in report.per_query:
        changed = "CHANGED" if c.plan_structure_changed else "same"
        lines.append(
            f"  {c.query_id:<25s} {changed:>7s} {c.cost_delta_pct:>+7.1f}% "
            f"{c.rows_delta_pct:>+7.1f}%  "
            + (", ".join(c.scan_changes) if c.scan_changes else "—")
        )

    lines.append(f"\n{'=' * 72}")
    return "\n".join(lines)


def save_report(report: ExperimentReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    report_dict = {
        "total_queries": report.total_queries,
        "successful_queries": report.successful_queries,
        "plan_change_rate_stale_to_oasis": report.stale_vs_oasis_plan_change_rate,
        "avg_cost_improvement_pct": report.avg_cost_improvement_oasis_vs_stale_pct,
        "plan_wins": report.plan_wins,
        "plan_neutral": report.plan_neutral,
        "plan_losses": report.plan_losses,
        "scan_transitions": report.scan_transitions,
        "per_query": [
            {
                "query_id": c.query_id,
                "plan_changed": c.plan_structure_changed,
                "cost_a": c.cost_a,
                "cost_b": c.cost_b,
                "cost_delta_pct": c.cost_delta_pct,
                "rows_a": c.rows_a,
                "rows_b": c.rows_b,
                "scan_changes": c.scan_changes,
            }
            for c in report.per_query
        ],
    }
    with open(output_dir / "optimizer_only_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    # LaTeX table
    with open(output_dir / "table_optimizer_only_e2e.tex", "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("  \\centering\n")
        f.write("  \\caption{Optimizer-only E2E: plan changes under OASIS-corrected ")
        f.write("statistics using \\texttt{EXPLAIN} (no \\texttt{ANALYZE}). ")
        f.write("Plan structure, estimated costs, and scan types compared across ")
        f.write("three statistics states without executing queries.}\n")
        f.write("  \\label{tab:optimizer_only_e2e}\n")
        f.write("  \\setlength{\\tabcolsep}{4pt}\n")
        f.write("  \\begin{tabular}{l c c c c c}\n")
        f.write("    \\toprule\n")
        f.write("    Metric & Stale & OASIS & Fresh & O→S Δ\\% & Recovery\\% \\\\\n")
        f.write("    \\midrule\n")

        if report.per_query:
            stale_cost = sum(c.cost_a for c in report.per_query) / len(report.per_query)
            oasis_cost = sum(c.cost_b for c in report.per_query) / len(report.per_query)
            cost_delta = (stale_cost - oasis_cost) / max(stale_cost, 1e-6) * 100

            f.write(
                f"    Avg Est. Cost & {stale_cost:.0f} & \\textbf{{{oasis_cost:.0f}}} & "
                f"— & {cost_delta:.1f}\\% & — \\\\\n"
            )
        f.write(
            f"    Plan Change Rate & — & — & — & "
            f"{report.stale_vs_oasis_plan_change_rate:.1f}\\% & — \\\\\n"
        )
        f.write(
            f"    Plan Wins & — & {report.plan_wins} & — & — & — \\\\\n"
        )
        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("  \\vspace{4pt}\n")
        f.write("  \\small\n")
        f.write(
            f"  {report.plan_wins}/{report.successful_queries} queries show "
            f"plan improvement; {report.plan_losses} queries show regression. "
            f"All measurements from optimizer cost estimates only "
            f"(\\texttt{EXPLAIN} without \\texttt{ANALYZE}).\n"
        )
        f.write("\\end{table}\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Optimizer-Only E2E: EXPLAIN-based plan evaluation"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # Live mode: requires PostgreSQL
    live = sub.add_parser("live", help="Run against live PostgreSQL (EXPLAIN only)")
    live.add_argument("--host", default="localhost")
    live.add_argument("--port", type=int, default=5433)
    live.add_argument("--dbname", default="tpcds")
    live.add_argument("--user", default="postgres")
    live.add_argument("--password", default="")
    live.add_argument("--query-dir", required=True,
                      help="Directory containing TPC-DS SQL files")
    live.add_argument("--columns", nargs="+", default=[],
                      help="Columns as table.column (e.g. item.i_current_price)")
    live.add_argument("--oasis-bounds", default="",
                      help="JSON file mapping column_key → corrected histogram bounds")
    live.add_argument("--output-dir", default="results/optimizer_only_e2e")
    live.add_argument("--timeout", type=int, default=30)

    # Offline mode: compare exported plan JSON files
    offline = sub.add_parser("offline", help="Compare exported EXPLAIN plan JSON files")
    offline.add_argument("--stale-dir", required=True)
    offline.add_argument("--oasis-dir", required=True)
    offline.add_argument("--fresh-dir", default="")
    offline.add_argument("--output-dir", default="results/optimizer_only_e2e")

    args = parser.parse_args()

    if args.mode == "live":
        if not _HAS_PSYCOPG2:
            print("Error: psycopg2 required. pip install psycopg2-binary")
            sys.exit(1)

        # Parse columns
        target_columns = []
        for col_spec in args.columns:
            parts = col_spec.split(".")
            if len(parts) == 2:
                target_columns.append((parts[0], parts[1]))

        # Load queries
        query_dir = Path(args.query_dir)
        queries = {}
        for sql_file in sorted(query_dir.glob("*.sql")):
            with open(sql_file) as f:
                queries[sql_file.stem] = f.read()
        print(f"Loaded {len(queries)} queries from {query_dir}")

        # Load OASIS corrections if provided
        oasis_corrections = {}
        oasis_null_fracs = {}
        if args.oasis_bounds:
            with open(args.oasis_bounds) as f:
                oasis_data = json.load(f)
                for key, val in oasis_data.items():
                    if isinstance(val, list):
                        oasis_corrections[key] = val
                    elif isinstance(val, dict):
                        oasis_corrections[key] = val.get("bounds", [])
                        if "null_frac" in val:
                            oasis_null_fracs[key] = val["null_frac"]

        runner = OptimizerOnlyE2E(
            host=args.host, port=args.port,
            dbname=args.dbname, user=args.user, password=args.password,
        )

        try:
            report = runner.run_full_experiment(
                queries=queries,
                target_columns=target_columns or [
                    ("item", "i_current_price"),
                    ("customer", "c_birth_year"),
                ],
                oasis_corrections=oasis_corrections,
                oasis_null_fracs=oasis_null_fracs,
            )

            output_dir = Path(args.output_dir)
            print(print_report(report))
            save_report(report, output_dir)
            print(f"\nResults saved to: {output_dir}")

        finally:
            runner.close()

    elif args.mode == "offline":
        report = compare_plan_json_files(
            Path(args.stale_dir), Path(args.oasis_dir), Path(args.fresh_dir or args.stale_dir),
        )
        output_dir = Path(args.output_dir)
        print(print_report(report))
        save_report(report, output_dir)


if __name__ == "__main__":
    main()
