#!/usr/bin/env python3
"""Generate the Stage-2 calibration LaTeX tables from cached result directories.

This is presentation-only: it reads existing experiment outputs (no reruns) and
emits the four ``.tex`` tables that the main paper and supplementary ``\\input``:

  - table_stage2_calibration_main.tex   (main paper, tab:stage2_calibration)
  - table_stage2_calibration_single.tex (supplementary, tab:stage2-app-single)
  - table_stage2_calibration_safety.tex (supplementary, tab:stage2-app-safety)
  - table_stage2_calibration_pg.tex     (supplementary, tab:stage2-app-pg)

Each Stage-2 soft variant (full-window / recent-window / conflict-aware) is the
``oasis_soft_projection`` method of a *different* run, so the run directory for
each variant is passed explicitly. Missing sources degrade to ``---`` rather
than crashing the paper build.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
_RESULTS = _REPO / "experiments" / "results"

MISSING = "---"


# ── readers ──────────────────────────────────────────────────────────────────

def _smoke_summary(run_dir: Path) -> Dict[str, dict]:
    """oasis_accuracy_smoke summary.csv -> {method: row}."""
    path = run_dir / "summary.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {row["method"]: row for row in csv.DictReader(handle)}


def _optimizer_agg(run_dir: Path) -> Dict[str, dict]:
    """optimizer_decision_proxy summary.json aggregate rows -> {method: row}."""
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("summary", [])
    agg = {}
    for row in rows:
        if str(row.get("q_mods")) in {"None", "all", "All"}:
            agg[row["method"]] = row
    if agg:
        return agg
    # Fallback: geometric mean across per-q rows.
    by_method: Dict[str, List[dict]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    out = {}
    for method, mrows in by_method.items():
        def gm(key: str) -> float:
            vals = [float(r[key]) for r in mrows if r.get(key) not in (None, "")]
            return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")
        out[method] = {"selectivity_qerr_gm": gm("selectivity_qerr_gm"),
                       "join_regret_gm": gm("join_regret_gm")}
    return out


def _per_group(run_dir: Path, group_col: str) -> Dict[str, Dict[str, float]]:
    """ood/trace summary.csv -> {group: {method: qerror_gm}}."""
    path = run_dir / "summary.csv"
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grp = row.get(group_col)
            if grp is None:
                continue
            out.setdefault(grp, {})[row["method"]] = float(row["qerror_gm"])
    return out


def _pg_all(run_dir: Path) -> Dict[str, dict]:
    """postgres batch_summary.json family=='all' rows -> {method: row}."""
    path = run_dir / "batch_summary.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = [rows]
    return {r["method"]: r for r in rows if str(r.get("family")) in {"all", "None"}}


# ── formatting helpers ───────────────────────────────────────────────────────

def fnum(value, fmt: str = "{:.3f}") -> str:
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return MISSING


def fpct(value, fmt: str = "{:.1f}\\%") -> str:
    try:
        return fmt.format(float(value) * 100.0)
    except (TypeError, ValueError):
        return MISSING


def write(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


# ── table builders ───────────────────────────────────────────────────────────

def build_single_column_rows(args) -> List[dict]:
    """Common per-variant single-column metrics used by the main and supp tables."""
    cal = _smoke_summary(args.calibrated_dir)        # hard, conflict-soft, calibrated, Hybrid
    full = _smoke_summary(args.soft_full_dir)        # full-window soft
    rec = _smoke_summary(args.soft_recent8_dir)      # recent-window soft

    def sc(row):
        if not row:
            return dict.fromkeys(("qerr", "resid", "qmae", "joinreg"))
        return {
            "qerr": row.get("selectivity_qerr_gm"),
            "resid": row.get("feedback_residual_mean"),
            "qmae": row.get("quantile_mae_mean"),
            "joinreg": row.get("join_regret_gm"),
        }

    return [
        {"name": "Hard projection (OASIS-Proj)", **sc(cal.get("oasis_full"))},
        {"name": "Full-window soft", **sc(full.get("oasis_soft_projection"))},
        {"name": "Recent-window soft ($k{=}8$)", **sc(rec.get("oasis_soft_projection"))},
        {"name": "Conflict-aware soft", **sc(cal.get("oasis_soft_projection"))},
        {"name": "Hybrid (no soft)", **sc(cal.get("residual_hybrid"))},
        {"name": "\\textbf{Calibrated router}", **sc(cal.get("calibrated_hybrid"))},
    ]


def build_optimizer_map(args) -> Dict[str, dict]:
    cal = _optimizer_agg(args.opt_calibrated_dir)
    full = _optimizer_agg(args.opt_soft_full_dir)
    rec = _optimizer_agg(args.opt_soft_recent8_dir)

    def op(row):
        if not row:
            return {"qerr": None, "joinreg": None}
        return {"qerr": row.get("selectivity_qerr_gm"), "joinreg": row.get("join_regret_gm")}

    return {
        "Hard projection (OASIS-Proj)": op(cal.get("oasis_projected")),
        "Full-window soft": op(full.get("oasis_soft_projection")),
        "Recent-window soft ($k{=}8$)": op(rec.get("oasis_soft_projection")),
        "Conflict-aware soft": op(cal.get("oasis_soft_projection")),
        "Hybrid (no soft)": op(cal.get("hybrid")),
        "\\textbf{Calibrated router}": op(cal.get("calibrated_hybrid")),
    }


def emit_main_table(args, out_dir: Path) -> None:
    rows = build_single_column_rows(args)
    body = []
    for r in rows:
        body.append(f"    {r['name']} & {fnum(r['qerr'])} & {fnum(r['resid'], '{:.4f}')} & "
                    f"{fnum(r['qmae'], '{:.4f}')} & {fnum(r['joinreg'], '{:.4f}')} \\\\")
    lines = [
        "% Auto-generated by experiments/make_stage2_calibration_tables.py",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Stage-2 calibration variants on held-out future predicates (cached",
        "  drift, fixed OASIS checkpoint). Lower is better. Full-window soft has the best",
        "  raw single-column metrics but is unsafe under sequential drift",
        "  (Section~\\ref{sec:ood-drift-realism}); the calibrated router is the best",
        "  \\emph{deployable} variant, recovering most of the single-column gain over hard",
        "  projection while remaining safe to route.}",
        "  \\label{tab:stage2_calibration}",
        "  \\setlength{\\tabcolsep}{5pt}",
        "  \\begin{tabular}{lrrrr}",
        "    \\toprule",
        "    Stage-2 variant & Sel. QErr & Feedback resid. & Quantile MAE & Join regret \\\\",
        "    \\midrule",
        *body,
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    write(out_dir / "table_stage2_calibration_main.tex", lines)


def emit_single_table(args, out_dir: Path) -> None:
    rows = build_single_column_rows(args)
    opt = build_optimizer_map(args)
    body = []
    for r in rows:
        o = opt[r["name"]]
        body.append(f"{r['name']} & {fnum(r['qerr'])} & {fnum(r['resid'], '{:.4f}')} & "
                    f"{fnum(r['qmae'], '{:.4f}')} & {fnum(r['joinreg'], '{:.4f}')} & "
                    f"{fnum(o['qerr'])} & {fnum(o['joinreg'], '{:.4f}')} \\\\")
    lines = [
        "% Auto-generated by experiments/make_stage2_calibration_tables.py",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Stage-2 calibration variants on held-out future predicates (expanded",
        "cached drift, fixed checkpoint) and on the optimizer-decision proxy. Lower is",
        "better.}",
        "\\label{tab:stage2-app-single}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        " & \\multicolumn{4}{c}{Single-column} & \\multicolumn{2}{c}{Optimizer proxy} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}",
        "Stage-2 variant & Sel.\\ QErr & Resid. & Q.\\ MAE & JoinReg & Sel.\\ QErr & JoinReg \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    write(out_dir / "table_stage2_calibration_single.tex", lines)


def emit_safety_table(args, out_dir: Path) -> None:
    ood_c = _per_group(args.ood_conflict_dir, "pattern")
    ood_r = _per_group(args.ood_recent8_dir, "pattern")
    ood_f = _per_group(args.ood_full_dir, "pattern")
    tr_c = _per_group(args.trace_conflict_dir, "trace")
    tr_f = _per_group(args.trace_full_dir, "trace")

    ood_rows = [
        ("OOD batch load", "batch_load"),
        ("OOD range shift", "range_shift"),
        ("OOD skew evolution", "skew_evol"),
        ("OOD outlier burst", "outlier"),
        ("OOD multimodal", "multimodal"),
        ("OOD seasonal/mixed", "seasonal"),
    ]
    tr_rows = [
        ("Trace sales append", "tpcds_sales_append"),
        ("Trace returns/cancel", "returns_cancellation"),
    ]

    def row_ood(label, key):
        hard = ood_c.get(key, {}).get("oasis_projected")
        rec = ood_r.get(key, {}).get("oasis_soft_projection")
        con = ood_c.get(key, {}).get("oasis_soft_projection")
        full = ood_f.get(key, {}).get("oasis_soft_projection")
        return f"{label} & {fnum(hard)} & {fnum(rec)} & {fnum(con)} & {fnum(full)} \\\\"

    def row_trace(label, key):
        hard = tr_c.get(key, {}).get("oasis_projected")
        con = tr_c.get(key, {}).get("oasis_soft_projection")
        full = tr_f.get(key, {}).get("oasis_soft_projection")
        return f"{label} & {fnum(hard)} & {MISSING} & {fnum(con)} & {fnum(full)} \\\\"

    lines = [
        "% Auto-generated by experiments/make_stage2_calibration_tables.py",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Future selectivity Q-error by OOD drift family and DML trace. Soft",
        "columns use no recency decay; conflict-aware uses $\\tau{=}0.03$. Full-window soft",
        "fails on the strong batch-load and range-shift families and on append/returns",
        "traces; the recent-window and conflict-aware variants remove those failures.}",
        "\\label{tab:stage2-app-safety}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Family / trace & Hard (Proj) & Recent8 soft & Conflict soft & Full-window soft \\\\",
        "\\midrule",
        *[row_ood(label, key) for label, key in ood_rows],
        "\\midrule",
        *[row_trace(label, key) for label, key in tr_rows],
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    write(out_dir / "table_stage2_calibration_safety.tex", lines)


def emit_pg_table(args, out_dir: Path) -> None:
    pg = _pg_all(args.pg_dir)

    def row(label, method):
        r = pg.get(method)
        if not r:
            return f"{label} & {MISSING} & {MISSING} & {MISSING} \\\\"
        return (f"{label} & {fnum(r.get('row_qerr_gm'))} & "
                f"{fpct(r.get('fresh_plan_match_frac'))} & "
                f"{fpct(r.get('new_plan_deviation_frac'), '{:.2f}\\%')} \\\\")

    lines = [
        "% Auto-generated by experiments/make_stage2_calibration_tables.py",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{PostgreSQL planner-only six-configuration subset (left/right/bimodal",
        "shift $\\times$ two seeds, 504 queries). The calibrated router is identical to the",
        "Hybrid because the residual gate routes to ISOMER and never selects soft.}",
        "\\label{tab:stage2-app-pg}",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Row QErr & Fresh-plan match & New plan deviations \\\\",
        "\\midrule",
        row("Hard projection (OASIS-Proj)", "oasis_projected"),
        row("Conflict-aware soft", "oasis_soft_projection"),
        row("Hybrid (no soft)", "hybrid"),
        row("Calibrated router", "calibrated_hybrid"),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    write(out_dir / "table_stage2_calibration_pg.tex", lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Stage-2 calibration LaTeX tables")
    p.add_argument("--out-dir", type=Path, default=_RESULTS / "stage2_calibration_20260531")
    p.add_argument("--calibrated-dir", type=Path, default=_RESULTS / "oasis_calibrated_hybrid_expanded_20260531")
    p.add_argument("--soft-full-dir", type=Path, default=_RESULTS / "oasis_soft_projection_expanded_20260531")
    p.add_argument("--soft-recent8-dir", type=Path, default=_RESULTS / "oasis_soft_recent8_expanded_20260531")
    p.add_argument("--opt-calibrated-dir", type=Path, default=_RESULTS / "optimizer_calibrated_full_20260531")
    p.add_argument("--opt-soft-full-dir", type=Path, default=_RESULTS / "optimizer_soft_full_20260531")
    p.add_argument("--opt-soft-recent8-dir", type=Path, default=_RESULTS / "optimizer_soft_recent8_full_20260531")
    p.add_argument("--ood-conflict-dir", type=Path, default=_RESULTS / "ood_drift_realism_soft_conflict_full_20260531")
    p.add_argument("--ood-recent8-dir", type=Path, default=_RESULTS / "ood_drift_realism_soft_recent8_full_20260531")
    p.add_argument("--ood-full-dir", type=Path, default=_RESULTS / "ood_drift_realism_soft_full_20260531")
    p.add_argument("--trace-conflict-dir", type=Path, default=_RESULTS / "trace_grounded_drift_soft_conflict_full_20260531")
    p.add_argument("--trace-full-dir", type=Path, default=_RESULTS / "trace_grounded_drift_soft_full_20260531")
    p.add_argument("--pg-dir", type=Path, default=_RESULTS / "postgres_calibrated_batch_subset_20260531")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    emit_main_table(args, out_dir)
    emit_single_table(args, out_dir)
    emit_safety_table(args, out_dir)
    emit_pg_table(args, out_dir)


if __name__ == "__main__":
    main()
