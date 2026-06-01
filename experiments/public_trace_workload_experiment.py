#!/usr/bin/env python3
"""Public production-trace case study for OASIS.

The project does not have private DBMS workload logs. This experiment therefore
uses a public production telemetry trace: the NASA Kennedy Space Center HTTP
access log from the Internet Traffic Archive. Each request is treated as an
append to an analytics/event table. We build stale single-column statistics
from an early prefix, append later real requests while collecting predicate
feedback, and evaluate future predicate constants drawn from the same public
trace.

This is intentionally a trace-grounded external-validity case study, not a
claim that we observed a private database production workload.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_PIPELINE_DIR = _REPO_DIR / "cdf_kll_ml_pipeline"
for _path in (_SCRIPT_DIR, _PIPELINE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from extended_drift_generators import ExtendedMemoryTable
from histogram_math import clamp01
from histogram_types import DEFAULT_QUANTILE_LEVELS, FeedbackObservation, KllFeedbackSample, KllPrior
from mlp_histogram_model_v2 import MlpHistogramModelV2
from optimizer_decision_proxy_experiment import build_method_boundaries, estimate_selectivity, feedback_residual
from ood_drift_realism_experiment import geomean, mean, q_error, quantile_boundaries, quantile_mae, selectivity_mae


NASA_JUL95_URL = "https://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz"
LOG_RE = re.compile(
    r'^(?P<host>\S+) \S+ \S+ \[(?P<ts>[^\]]+)\] "(?P<request>[^"]*)" '
    r"(?P<status>\d{3}) (?P<bytes>\S+)"
)

COLUMN_ORDER = ["reply_bytes", "time_of_day", "path_length"]
COLUMN_LABEL = {
    "reply_bytes": "Reply bytes",
    "time_of_day": "Time of day",
    "path_length": "Path length",
}
COLUMN_SIGNAL = {
    "reply_bytes": "log reply size",
    "time_of_day": "request timestamp",
    "path_length": "request path text",
}
METHOD_ORDER = [
    "stale",
    "isomer",
    "oasis",
    "oasis_projected",
    "oasis_soft_projection",
    "hybrid",
    "calibrated_hybrid",
    "fresh",
]
METHOD_LABEL = {
    "stale": "Stale",
    "isomer": "ISOMER",
    "oasis": "OASIS-noProj",
    "oasis_projected": "OASIS",
    "oasis_soft_projection": "Soft",
    "hybrid": "Hybrid",
    "calibrated_hybrid": "Router",
    "fresh": "Fresh",
}


@dataclass
class PublicTraceCase:
    sample: KllFeedbackSample
    column: str
    case_id: int
    start_event: int
    initial_events: int
    appended_events: int
    future_events: int
    stale_boundaries: List[float]
    fresh_boundaries: List[float]
    metric_points: List[float]


@dataclass
class PublicTraceRow:
    column: str
    case_id: int
    method: str
    qerror: float
    selectivity_mae: float
    quantile_mae: float
    feedback_residual: float
    beats_stale_qerr: bool


class ProgressBar:
    def __init__(self, total: int, *, label: str, enabled: bool = True) -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.enabled = enabled
        self.count = 0
        self.start = time.time()
        self.last_draw = 0.0

    def update(self, step: int = 1, status: str = "") -> None:
        self.count = min(self.total, self.count + step)
        if not self.enabled:
            return
        now = time.time()
        if self.count < self.total and now - self.last_draw < 0.20:
            return
        self.last_draw = now
        width = 30
        filled = int(width * self.count / self.total)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = max(now - self.start, 1e-9)
        rate = self.count / elapsed
        eta = (self.total - self.count) / max(rate, 1e-9)
        msg = (
            f"\r{self.label} [{bar}] {self.count}/{self.total} "
            f"({self.count / self.total * 100:5.1f}%) "
            f"elapsed {elapsed:6.1f}s eta {eta:6.1f}s"
        )
        if status:
            msg += f" | {status[:54]:<54}"
        sys.stderr.write(msg)
        sys.stderr.flush()

    def close(self) -> None:
        if self.enabled:
            self.update(0)
            sys.stderr.write("\n")
            sys.stderr.flush()


def download_with_progress(url: str, dest: Path, *, no_progress: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get("Content-Length", "0") or 0)
        progress = ProgressBar(total or 1, label="download-public-trace", enabled=not no_progress)
        with tmp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                progress.update(len(chunk), status=dest.name)
        progress.close()
    tmp.replace(dest)


def request_path(request: str) -> str:
    parts = request.split()
    if len(parts) >= 2:
        return parts[1]
    return request or "/"


def parse_timestamp(ts: str) -> Optional[datetime]:
    try:
        return datetime.strptime(ts, "%d/%b/%Y:%H:%M:%S %z")
    except ValueError:
        return None


def normalize_log(values: Sequence[float], percentile: float = 99.5) -> Tuple[List[float], float]:
    logs = np.log1p(np.asarray(values, dtype=float))
    scale = float(np.percentile(logs, percentile)) if len(logs) else 1.0
    scale = max(scale, 1e-9)
    return [clamp01(float(value / scale)) for value in logs], scale


def parse_public_trace(path: Path, max_events: int, *, no_progress: bool) -> Tuple[Dict[str, List[float]], dict]:
    bytes_raw: List[float] = []
    seconds_raw: List[float] = []
    path_len_raw: List[float] = []
    status_counts: Counter = Counter()
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None
    skipped = 0

    progress = ProgressBar(max_events, label="parse-public-trace", enabled=not no_progress)
    with gzip.open(path, "rt", encoding="latin-1", errors="replace") as handle:
        for line in handle:
            if len(bytes_raw) >= max_events:
                break
            match = LOG_RE.match(line)
            if not match:
                skipped += 1
                continue
            ts = parse_timestamp(match.group("ts"))
            if ts is None:
                skipped += 1
                continue
            if first_ts is None:
                first_ts = ts.isoformat()
            last_ts = ts.isoformat()
            raw_bytes = match.group("bytes")
            try:
                byte_count = float(raw_bytes) if raw_bytes != "-" else 0.0
            except ValueError:
                byte_count = 0.0
            path_text = request_path(match.group("request"))
            seconds = ts.hour * 3600 + ts.minute * 60 + ts.second
            bytes_raw.append(max(byte_count, 0.0))
            seconds_raw.append(seconds / 86400.0)
            path_len_raw.append(float(len(path_text)))
            status_counts[match.group("status")] += 1
            progress.update(status=f"events={len(bytes_raw)} skipped={skipped}")
    progress.close()

    reply_bytes, reply_scale = normalize_log(bytes_raw)
    path_length, path_scale = normalize_log(path_len_raw)
    columns = {
        "reply_bytes": reply_bytes,
        "time_of_day": [clamp01(value) for value in seconds_raw],
        "path_length": path_length,
    }
    metadata = {
        "source_url": NASA_JUL95_URL,
        "local_path": str(path),
        "parsed_events": len(bytes_raw),
        "skipped_lines": skipped,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "reply_bytes_log_p99_5": reply_scale,
        "path_length_log_p99_5": path_scale,
        "status_counts": dict(status_counts.most_common(12)),
    }
    return columns, metadata


def draw_observation(
    rng: random.Random,
    table: ExtendedMemoryTable,
    stale_boundaries: Sequence[float],
    timestamp: datetime,
) -> FeedbackObservation:
    pred_type = rng.choices(["<=", ">=", "BETWEEN", "="], weights=[0.32, 0.32, 0.30, 0.06], k=1)[0]
    value_upper: Optional[float] = None
    if table.data:
        sorted_data = sorted(table.data)
        if pred_type == "BETWEEN":
            lo_idx = rng.randint(0, max(0, len(sorted_data) - 2))
            hi_idx = rng.randint(lo_idx + 1, len(sorted_data) - 1)
            value = sorted_data[lo_idx]
            value_upper = sorted_data[hi_idx]
        else:
            value = sorted_data[rng.randint(0, len(sorted_data) - 1)]
    else:
        value = rng.random()
        if pred_type == "BETWEEN":
            value_upper = min(1.0, value + rng.uniform(0.02, 0.25))
    predicate = {"predicate_type": pred_type, "value": value, "value_upper": value_upper}
    return FeedbackObservation(
        predicate_type=pred_type,
        value=float(value),
        value_upper=value_upper,
        actual_selectivity=table.query_conditional_sel(pred_type, float(value), value_upper),
        estimated_selectivity=estimate_selectivity(stale_boundaries, predicate),
        timestamp=timestamp,
    )


def observation_dicts(observations: Sequence[FeedbackObservation]) -> List[dict]:
    return [
        {
            "predicate_type": obs.predicate_type,
            "value": obs.value,
            "value_upper": obs.value_upper,
            "actual_sel": obs.actual_selectivity,
            "estimated_sel": obs.estimated_selectivity,
        }
        for obs in observations
    ]


def future_metric_points(values: Sequence[float], rng: random.Random, count: int) -> List[float]:
    points: List[float] = []
    if values:
        sample_count = min(len(values), max(count // 2, 1))
        points.extend(float(values[idx]) for idx in rng.sample(range(len(values)), sample_count))
    while len(points) < count:
        points.append(rng.uniform(0.005, 0.995))
    return sorted(clamp01(point) for point in points[:count])


def case_starts(n_events: int, window: int, cases: int, seed: int) -> List[int]:
    max_start = max(n_events - window - 1, 0)
    if cases <= 1:
        return [0]
    rng = random.Random(seed)
    starts = []
    stride = max_start / cases
    jitter = max(int(stride * 0.20), 1)
    for idx in range(cases):
        base = int(idx * stride)
        starts.append(min(max_start, max(0, base + rng.randint(-jitter, jitter))))
    return sorted(starts)


def build_case(column: str, case_id: int, values: Sequence[float], start: int, args: argparse.Namespace) -> PublicTraceCase:
    rng = random.Random(args.seed + case_id * 7919 + COLUMN_ORDER.index(column) * 1_000_003)
    init_end = start + args.initial_events
    table = ExtendedMemoryTable(list(values[start:init_end]), 0)
    stale_boundaries = quantile_boundaries(table, args.num_buckets)
    prior_null = table.get_null_fraction()
    observations: List[FeedbackObservation] = []
    pos = init_end
    base_time = datetime(1995, 7, 1, tzinfo=timezone.utc) + timedelta(seconds=start)
    for obs_idx in range(args.observations):
        chunk = values[pos:pos + args.events_per_observation]
        table.data.extend(clamp01(float(value)) for value in chunk)
        pos += len(chunk)
        observations.append(draw_observation(rng, table, stale_boundaries, base_time + timedelta(minutes=obs_idx)))
    fresh_boundaries = quantile_boundaries(table, args.num_buckets)
    future_values = list(values[pos:pos + args.future_events])
    metric_points = future_metric_points(future_values, rng, args.metric_points)
    sample = KllFeedbackSample(
        prior=KllPrior(
            min_value=0.0,
            max_value=1.0,
            null_fraction=prior_null,
            quantile_levels=list(DEFAULT_QUANTILE_LEVELS),
            quantile_values=stale_boundaries[1:-1],
            value_type="double",
        ),
        observations=observations,
        corrected_quantile_values=fresh_boundaries[1:-1],
        source_path=f"public_trace:nasa_jul95:{column}:{case_id}",
    )
    return PublicTraceCase(
        sample=sample,
        column=column,
        case_id=case_id,
        start_event=start,
        initial_events=args.initial_events,
        appended_events=pos - init_end,
        future_events=len(future_values),
        stale_boundaries=stale_boundaries,
        fresh_boundaries=fresh_boundaries,
        metric_points=metric_points,
    )


def summarize_cases(cases: Sequence[PublicTraceCase]) -> List[dict]:
    by_column: Dict[str, List[PublicTraceCase]] = defaultdict(list)
    for case in cases:
        by_column[case.column].append(case)
    rows = []
    for column in COLUMN_ORDER:
        items = by_column[column]
        if not items:
            continue
        rows.append({
            "column": column,
            "label": COLUMN_LABEL[column],
            "signal": COLUMN_SIGNAL[column],
            "n_cases": len(items),
            "initial_events_mean": mean([case.initial_events for case in items]),
            "appended_events_mean": mean([case.appended_events for case in items]),
            "growth_mean": mean([case.appended_events / max(case.initial_events, 1) for case in items]),
            "future_events_mean": mean([case.future_events for case in items]),
            "stale_quantile_mae_mean": mean([quantile_mae(case.stale_boundaries, case.fresh_boundaries) for case in items]),
        })
    return rows


def aggregate(rows: Sequence[PublicTraceRow], router_choices: Dict[str, Counter]) -> List[dict]:
    grouped: Dict[Tuple[str, str], List[PublicTraceRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.column, row.method)].append(row)
        grouped[("all", row.method)].append(row)

    result = []
    for column in COLUMN_ORDER + ["all"]:
        stale_rows = grouped[(column, "stale")]
        if not stale_rows:
            continue
        stale_qerr = geomean([row.qerror for row in stale_rows])
        total_choices = sum(router_choices[column].values())
        for method in METHOD_ORDER:
            method_rows = grouped[(column, method)]
            if not method_rows:
                continue
            qerr = geomean([row.qerror for row in method_rows])
            result.append({
                "column": column,
                "label": "Aggregate" if column == "all" else COLUMN_LABEL[column],
                "signal": "all public trace columns" if column == "all" else COLUMN_SIGNAL[column],
                "method": method,
                "n_cases": len(method_rows),
                "qerror_gm": qerr,
                "qerror_improvement_pct": (stale_qerr - qerr) / max(stale_qerr, 1e-12) * 100.0,
                "selectivity_mae_mean": mean([row.selectivity_mae for row in method_rows]),
                "quantile_mae_mean": mean([row.quantile_mae for row in method_rows]),
                "feedback_residual_mean": mean([row.feedback_residual for row in method_rows if math.isfinite(row.feedback_residual)]),
                "beats_stale_frac": mean([1.0 if row.beats_stale_qerr else 0.0 for row in method_rows]),
                "router_isomer_frac": router_choices[column]["isomer"] / max(total_choices, 1),
                "router_hard_frac": router_choices[column]["oasis_projected"] / max(total_choices, 1),
                "router_soft_frac": router_choices[column]["oasis_soft_projection"] / max(total_choices, 1),
                "router_oasis_frac": router_choices[column]["oasis"] / max(total_choices, 1),
                "router_stale_frac": router_choices[column]["stale"] / max(total_choices, 1),
            })
    return result


def write_table(output_dir: Path, summary: Sequence[dict], case_summary: Sequence[dict]) -> None:
    by_key = {(row["column"], row["method"]): row for row in summary}
    by_col = {row["column"]: row for row in case_summary}
    with (output_dir / "table_public_trace_workload.tex").open("w") as handle:
        handle.write("\\begin{table*}[t]\n")
        handle.write("  \\centering\n")
        handle.write("  \\small\n")
        handle.write("  \\caption{Public production-trace case study using the NASA Kennedy Space Center HTTP request trace. Each request appends one event to an analytics table; stale statistics are built from an early prefix, feedback is collected after later real requests, and future predicate probes include constants drawn from held-out later requests. Values are geometric mean selectivity Q-error. This is a public telemetry trace, not a private DBMS query log.}\n")
        handle.write("  \\label{tab:public_trace_workload}\n")
        handle.write("  \\setlength{\\tabcolsep}{4pt}\n")
        handle.write("  \\resizebox{\\textwidth}{!}{%\n")
        handle.write("  \\begin{tabular}{llrrrrrrrrr}\n")
        handle.write("    \\toprule\n")
        handle.write("    Trace column & Signal & Growth & Stale & ISOMER & OASIS-noProj & OASIS & Soft & Hybrid & Router & Fresh \\\\\n")
        handle.write("    \\midrule\n")
        for column in COLUMN_ORDER + ["all"]:
            row = by_col.get(column, {"growth_mean": 0.0, "signal": "all public trace columns"})
            values = {method: by_key[(column, method)]["qerror_gm"] for method in METHOD_ORDER}
            candidates = {method: values[method] for method in METHOD_ORDER if method not in {"stale", "fresh"}}
            best = min(candidates, key=candidates.get)

            def fmt(method: str) -> str:
                text = f"{values[method]:.3f}"
                return f"\\textbf{{{text}}}" if method == best else text

            label = "Aggregate" if column == "all" else COLUMN_LABEL[column]
            signal = "all" if column == "all" else row["signal"]
            growth = "--" if column == "all" else f"{row['growth_mean'] * 100:.0f}\\%"
            handle.write(
                f"    {label} & {signal} & {growth} & {values['stale']:.3f} & "
                f"{fmt('isomer')} & {fmt('oasis')} & {fmt('oasis_projected')} & "
                f"{fmt('oasis_soft_projection')} & {fmt('hybrid')} & "
                f"{fmt('calibrated_hybrid')} & {values['fresh']:.3f} \\\\\n"
            )
        handle.write("    \\bottomrule\n")
        handle.write("  \\end{tabular}%\n")
        handle.write("  }\n")
        handle.write("\\end{table*}\n")


def write_outputs(
    output_dir: Path,
    rows: Sequence[PublicTraceRow],
    summary: Sequence[dict],
    case_summary: Sequence[dict],
    metadata: dict,
    router_choices: Dict[str, Counter],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(list(summary), handle, indent=2)
    with (output_dir / "case_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_summary[0].keys()))
        writer.writeheader()
        writer.writerows(case_summary)
    with (output_dir / "case_summary.json").open("w") as handle:
        json.dump(list(case_summary), handle, indent=2)
    with (output_dir / "router_choices.json").open("w") as handle:
        json.dump({key: dict(value) for key, value in router_choices.items()}, handle, indent=2)
    with (output_dir / "trace_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    with (output_dir / "run_config.json").open("w") as handle:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, handle, indent=2)
    if args.write_rows:
        with (output_dir / "case_rows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
    write_table(output_dir, summary, case_summary)
    write_summary_text(output_dir, summary, case_summary, router_choices)


def write_summary_text(
    output_dir: Path,
    summary: Sequence[dict],
    case_summary: Sequence[dict],
    router_choices: Dict[str, Counter],
) -> None:
    by_key = {(row["column"], row["method"]): row for row in summary}
    by_col = {row["column"]: row for row in case_summary}
    lines = [
        "Public production-trace case study (NASA HTTP)",
        "=" * 56,
        "The trace is a public HTTP request log; no private DBMS production query log is used.",
        "",
        "Column          Growth  Stale  ISOMER  OASIS  Proj   Soft  Hybrid  Router  Fresh",
        "-" * 90,
    ]
    for column in COLUMN_ORDER + ["all"]:
        growth = "--" if column == "all" else f"{by_col[column]['growth_mean'] * 100:5.1f}%"
        label = "Aggregate" if column == "all" else COLUMN_LABEL[column]
        row = lambda method: by_key[(column, method)]["qerror_gm"]
        lines.append(
            f"{label:<15s} {growth:>7s}  {row('stale'):5.3f}  {row('isomer'):6.3f}  "
            f"{row('oasis'):5.3f}  {row('oasis_projected'):5.3f}  "
            f"{row('oasis_soft_projection'):5.3f}  {row('hybrid'):6.3f}  "
            f"{row('calibrated_hybrid'):6.3f}  {row('fresh'):5.3f}"
        )
        total = sum(router_choices[column].values())
        if total:
            mix = ", ".join(
                f"{method}={count / total * 100:.1f}%"
                for method, count in sorted(router_choices[column].items())
            )
            lines.append(f"  Router choices: {mix}")
    text = "\n".join(lines)
    (output_dir / "summary.txt").write_text(text + "\n")
    print(text)


def run(args: argparse.Namespace) -> None:
    if not args.trace_path.exists():
        if args.no_download:
            raise FileNotFoundError(f"Trace file missing and --no-download was set: {args.trace_path}")
        download_with_progress(args.trace_url, args.trace_path, no_progress=args.no_progress)

    columns, metadata = parse_public_trace(args.trace_path, args.max_events, no_progress=args.no_progress)
    window = args.initial_events + args.observations * args.events_per_observation + args.future_events
    for column, values in columns.items():
        if len(values) <= window:
            raise ValueError(f"Not enough events for {column}: parsed={len(values)}, required window={window}")

    model = MlpHistogramModelV2.load(str(args.model_path))
    model_window = int(getattr(model, "max_observations", 16))
    rows: List[PublicTraceRow] = []
    cases: List[PublicTraceCase] = []
    router_choices: Dict[str, Counter] = {column: Counter() for column in COLUMN_ORDER + ["all"]}
    total_cases = args.cases_per_column * len(COLUMN_ORDER)
    progress = ProgressBar(total_cases, label="public-trace-cases", enabled=not args.no_progress)

    for column in COLUMN_ORDER:
        values = columns[column]
        starts = case_starts(len(values), window, args.cases_per_column, args.seed + COLUMN_ORDER.index(column) * 1009)
        for case_id, start in enumerate(starts):
            case = build_case(column, case_id, values, start, args)
            cases.append(case)
            boundaries, _ = build_method_boundaries(
                case.sample,
                model=model,
                num_buckets=args.num_buckets,
                max_observations=model_window,
                soft_projection_strength=args.soft_projection_strength,
                soft_projection_recency_decay=args.soft_projection_recency_decay,
                soft_projection_target_blend=args.soft_projection_target_blend,
                soft_projection_window=args.soft_projection_window,
                soft_projection_iters=args.soft_projection_iters,
                soft_projection_lr=args.soft_projection_lr,
                soft_projection_tol=args.soft_projection_tol,
                soft_projection_active_set=False,
                soft_projection_conflict_aware=not args.disable_conflict_aware_soft,
                soft_projection_conflict_ref_window=args.soft_projection_conflict_ref_window,
                soft_projection_conflict_tau=args.soft_projection_conflict_tau,
                soft_projection_conflict_floor=args.soft_projection_conflict_floor,
            )
            observations = observation_dicts(case.sample.observations)
            router_choice = min(
                ["stale", "isomer", "oasis", "oasis_projected", "oasis_soft_projection"],
                key=lambda method: feedback_residual(boundaries[method], observations),
            )
            router_choices[column][router_choice] += 1
            router_choices["all"][router_choice] += 1
            stale_qerr = q_error(boundaries["stale"], case.fresh_boundaries, case.metric_points)
            for method in METHOD_ORDER:
                method_qerr = q_error(boundaries[method], case.fresh_boundaries, case.metric_points)
                rows.append(PublicTraceRow(
                    column=column,
                    case_id=case_id,
                    method=method,
                    qerror=method_qerr,
                    selectivity_mae=selectivity_mae(boundaries[method], case.fresh_boundaries, case.metric_points),
                    quantile_mae=quantile_mae(boundaries[method], case.fresh_boundaries),
                    feedback_residual=feedback_residual(boundaries[method], observations),
                    beats_stale_qerr=method_qerr < stale_qerr,
                ))
            progress.update(status=f"{column} case={case_id + 1}/{len(starts)}")
    progress.close()

    case_summary = summarize_cases(cases)
    summary = aggregate(rows, router_choices)
    metadata.update({
        "columns": COLUMN_ORDER,
        "column_signal": COLUMN_SIGNAL,
        "cases_per_column": args.cases_per_column,
        "initial_events": args.initial_events,
        "events_per_observation": args.events_per_observation,
        "observations": args.observations,
        "future_events": args.future_events,
    })
    write_outputs(args.output_dir, rows, summary, case_summary, metadata, router_choices, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public production-trace case study using NASA HTTP logs")
    parser.add_argument("--trace-url", default=NASA_JUL95_URL)
    parser.add_argument("--trace-path", type=Path,
                        default=_REPO_DIR / "experiments" / "data" / "public_traces" / "NASA_access_log_Jul95.gz")
    parser.add_argument("--model-path", type=Path,
                        default=_REPO_DIR / "experiments" / "results" / "synthetic_paper_suite_rerun_20260529" / "models" / "oasis_k16.json")
    parser.add_argument("--output-dir", type=Path,
                        default=_REPO_DIR / "experiments" / "results" / "public_trace_workload_20260601")
    parser.add_argument("--max-events", type=int, default=750_000)
    parser.add_argument("--cases-per-column", type=int, default=96)
    parser.add_argument("--initial-events", type=int, default=2_000)
    parser.add_argument("--events-per-observation", type=int, default=1_000)
    parser.add_argument("--observations", type=int, default=16)
    parser.add_argument("--future-events", type=int, default=4_000)
    parser.add_argument("--num-buckets", type=int, default=10)
    parser.add_argument("--metric-points", type=int, default=64)
    parser.add_argument("--soft-projection-strength", type=float, default=30.0)
    parser.add_argument("--soft-projection-recency-decay", type=float, default=1.0)
    parser.add_argument("--soft-projection-target-blend", type=float, default=1.0)
    parser.add_argument("--soft-projection-window", type=int, default=0)
    parser.add_argument("--soft-projection-iters", type=int, default=500)
    parser.add_argument("--soft-projection-lr", type=float, default=0.05)
    parser.add_argument("--soft-projection-tol", type=float, default=1e-9)
    parser.add_argument("--disable-conflict-aware-soft", action="store_true")
    parser.add_argument("--soft-projection-conflict-ref-window", type=int, default=8)
    parser.add_argument("--soft-projection-conflict-tau", type=float, default=0.03)
    parser.add_argument("--soft-projection-conflict-floor", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--write-rows", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
