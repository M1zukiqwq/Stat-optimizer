#!/usr/bin/env python3
"""
MCV Adapter Validation Experiments

This script validates the three-phase MCV adapter pattern using a model that
matches PostgreSQL's actual single-column statistics layout:

1. MCV entries store explicit value frequencies.
2. The residual histogram stores only sorted boundary values (no per-bucket
   frequencies).  Each residual bin implicitly carries equal probability mass.
3. The adapter reconstructs a canonical full distribution from the MCV list
   and compressed histogram, then decomposes it back to PostgreSQL format.

The experiments report:
  1. Round-trip translation accuracy
  2. Predicate selectivity consistency
  3. Conversion overhead
  4. Sensitivity to the MCV extraction threshold

Results are written to experiments/results/mcv_validation_results.json.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DOMAIN_MIN = 0.0
DOMAIN_MAX = 1.0
EPS = 1e-12


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class MCVEntry:
    value: float
    frequency: float


@dataclass
class HistogramBucket:
    lower: float
    upper: float
    frequency: float


@dataclass
class SingletonBucket:
    value: float
    frequency: float


@dataclass
class FullHistogram:
    singletons: List[SingletonBucket]
    ranges: List[HistogramBucket]
    null_fraction: float = 0.0
    target_histogram_size: int = 0  # number of stored histogram boundary values
    n_distinct: int = 0

    def total_non_null_frequency(self) -> float:
        singleton_total = sum(s.frequency for s in self.singletons)
        range_total = sum(r.frequency for r in self.ranges)
        return singleton_total + range_total


@dataclass
class PostgreSQLStatistics:
    mcv_list: List[MCVEntry]
    histogram_bounds: List[float]  # M stored boundary values, defining M-1 bins
    n_distinct: int
    null_fraction: float = 0.0


# ============================================================================
# Helper utilities
# ============================================================================


@dataclass
class PiecewiseDistribution:
    points: List[float]
    atom_masses: List[float]
    interval_masses: List[float]
    densities: List[float]
    total_mass: float

    @classmethod
    def from_components(
        cls,
        singletons: Sequence[SingletonBucket],
        ranges: Sequence[HistogramBucket],
    ) -> "PiecewiseDistribution":
        atom_map: Dict[float, float] = {}
        points = set()

        normalized_ranges: List[HistogramBucket] = []
        for singleton in singletons:
            value = float(np.clip(singleton.value, DOMAIN_MIN, DOMAIN_MAX))
            atom_map[value] = atom_map.get(value, 0.0) + float(singleton.frequency)
            points.add(value)

        for bucket in ranges:
            lower = float(np.clip(bucket.lower, DOMAIN_MIN, DOMAIN_MAX))
            upper = float(np.clip(bucket.upper, DOMAIN_MIN, DOMAIN_MAX))
            freq = float(bucket.frequency)
            if freq <= EPS:
                continue
            if upper <= lower + EPS:
                atom_map[lower] = atom_map.get(lower, 0.0) + freq
                points.add(lower)
                continue
            normalized_ranges.append(HistogramBucket(lower, upper, freq))
            points.add(lower)
            points.add(upper)

        if not points:
            points = {DOMAIN_MIN, DOMAIN_MAX}
        sorted_points = sorted(points)
        atom_masses = [atom_map.get(point, 0.0) for point in sorted_points]
        interval_masses: List[float] = []
        densities: List[float] = []

        for left, right in zip(sorted_points[:-1], sorted_points[1:]):
            width = right - left
            if width <= EPS:
                interval_masses.append(0.0)
                densities.append(0.0)
                continue
            density = 0.0
            for bucket in normalized_ranges:
                if bucket.lower <= left + EPS and bucket.upper >= right - EPS:
                    density += bucket.frequency / max(bucket.upper - bucket.lower, EPS)
            interval_masses.append(density * width)
            densities.append(density)

        total_mass = sum(atom_masses) + sum(interval_masses)
        return cls(sorted_points, atom_masses, interval_masses, densities, total_mass)

    def cdf_less_than(self, value: float) -> float:
        if self.total_mass <= EPS:
            return 0.0
        value = float(np.clip(value, DOMAIN_MIN, DOMAIN_MAX))
        cumulative = 0.0

        for index, point in enumerate(self.points):
            if point >= value - EPS:
                if point >= value + EPS:
                    break
                # exact match -> do not include atom at this point for strict "<"
                break

            cumulative += self.atom_masses[index]
            if index >= len(self.interval_masses):
                continue

            left = point
            right = self.points[index + 1]
            if value >= right - EPS:
                cumulative += self.interval_masses[index]
            else:
                partial = max(value - left, 0.0)
                cumulative += self.densities[index] * partial
                break

        return max(0.0, min(cumulative / self.total_mass, 1.0))

    def cdf_less_equal(self, value: float) -> float:
        if self.total_mass <= EPS:
            return 0.0
        value = float(np.clip(value, DOMAIN_MIN, DOMAIN_MAX))
        cumulative = 0.0

        for index, point in enumerate(self.points):
            if point > value + EPS:
                break

            cumulative += self.atom_masses[index]
            if index >= len(self.interval_masses):
                continue

            left = point
            right = self.points[index + 1]
            if value >= right - EPS:
                cumulative += self.interval_masses[index]
            else:
                partial = max(value - left, 0.0)
                cumulative += self.densities[index] * partial
                break

        return max(0.0, min(cumulative / self.total_mass, 1.0))

    def quantile(self, probability: float) -> float:
        if self.total_mass <= EPS:
            return DOMAIN_MIN

        probability = float(np.clip(probability, 0.0, 1.0))
        if probability <= 0.0:
            return self.points[0]
        if probability >= 1.0:
            return self.points[-1]

        target_mass = probability * self.total_mass
        cumulative = 0.0

        for index, point in enumerate(self.points):
            atom_mass = self.atom_masses[index]
            if target_mass <= cumulative + atom_mass + EPS:
                return point
            cumulative += atom_mass

            if index >= len(self.interval_masses):
                continue

            interval_mass = self.interval_masses[index]
            if interval_mass <= EPS:
                continue
            if target_mass <= cumulative + interval_mass + EPS:
                density = self.densities[index]
                if density <= EPS:
                    return point
                return min(
                    self.points[index + 1],
                    point + (target_mass - cumulative) / density,
                )
            cumulative += interval_mass

        return self.points[-1]


@dataclass
class Predicate:
    kind: str
    value: float
    value_upper: Optional[float] = None


@dataclass
class SelectivitySummary:
    mae: float
    p95: float
    p99: float
    maximum: float


def _summarize(values: Sequence[float]) -> Dict[str, float]:
    values_arr = np.asarray(list(values), dtype=float)
    return {
        "mean": float(np.mean(values_arr)),
        "max": float(np.max(values_arr)),
        "std": float(np.std(values_arr)),
    }


def _clip_value(value: float) -> float:
    return float(np.clip(value, DOMAIN_MIN, DOMAIN_MAX))


def _histogram_positions(n_values: int, n_hist_values: int) -> List[int]:
    if n_hist_values <= 1:
        return [0]
    delta = (n_values - 1) // (n_hist_values - 1)
    deltafrac = (n_values - 1) % (n_hist_values - 1)
    pos = 0
    posfrac = 0
    positions: List[int] = []
    for _ in range(n_hist_values):
        positions.append(pos)
        pos += delta
        posfrac += deltafrac
        if posfrac >= (n_hist_values - 1):
            pos += 1
            posfrac -= (n_hist_values - 1)
    return positions


def _non_null_mass(pg_stats: PostgreSQLStatistics) -> float:
    mcv_mass = sum(entry.frequency for entry in pg_stats.mcv_list)
    residual_mass = max(0.0, 1.0 - pg_stats.null_fraction - mcv_mass)
    return mcv_mass + residual_mass


def _residual_mass(pg_stats: PostgreSQLStatistics) -> float:
    return max(0.0, 1.0 - pg_stats.null_fraction - sum(entry.frequency for entry in pg_stats.mcv_list))


def _find_bucket(bounds: Sequence[float], value: float) -> int:
    if len(bounds) < 2:
        return -1
    if value <= bounds[0]:
        return 0
    if value >= bounds[-1]:
        return len(bounds) - 2
    return max(0, min(int(np.searchsorted(bounds, value, side="right") - 1), len(bounds) - 2))


def _pg_histogram_cdf(bounds: Sequence[float], value: float) -> float:
    if len(bounds) < 2:
        return 0.0
    if value <= bounds[0]:
        return 0.0
    if value >= bounds[-1]:
        return 1.0

    bucket_index = _find_bucket(bounds, value)
    left = bounds[bucket_index]
    right = bounds[bucket_index + 1]
    if right <= left + EPS:
        bin_fraction = 1.0 if value >= right - EPS else 0.0
    else:
        bin_fraction = max(0.0, min((value - left) / (right - left), 1.0))

    bin_count = len(bounds) - 1
    return ((float(bucket_index) + bin_fraction) / float(bin_count))


def _pg_eq_selectivity(pg_stats: PostgreSQLStatistics, value: float) -> float:
    for entry in pg_stats.mcv_list:
        if abs(entry.value - value) <= 1e-9:
            return entry.frequency

    residual_mass = _residual_mass(pg_stats)
    other_distinct = max(pg_stats.n_distinct - len(pg_stats.mcv_list), 1)
    estimate = residual_mass / float(other_distinct)
    return max(0.0, estimate)


def _pg_lt_selectivity(pg_stats: PostgreSQLStatistics, value: float) -> float:
    mcv_mass = sum(entry.frequency for entry in pg_stats.mcv_list if entry.value < value)
    residual_mass = _residual_mass(pg_stats)
    residual_fraction = _pg_histogram_cdf(pg_stats.histogram_bounds, value)
    return max(0.0, min(mcv_mass + residual_mass * residual_fraction, 1.0))


def _pg_between_selectivity(pg_stats: PostgreSQLStatistics, lower: float, upper: float) -> float:
    if upper < lower:
        lower, upper = upper, lower
    mcv_mass = sum(entry.frequency for entry in pg_stats.mcv_list if lower <= entry.value <= upper)
    lower_cdf = _pg_histogram_cdf(pg_stats.histogram_bounds, lower)
    upper_cdf = _pg_histogram_cdf(pg_stats.histogram_bounds, upper)
    residual_mass = _residual_mass(pg_stats)
    residual_fraction = max(0.0, upper_cdf - lower_cdf)
    return max(0.0, min(mcv_mass + residual_mass * residual_fraction, 1.0))


def estimate_pg_selectivity(pg_stats: PostgreSQLStatistics, predicate: Predicate) -> float:
    if predicate.kind == "=":
        return _pg_eq_selectivity(pg_stats, predicate.value)
    if predicate.kind == "<":
        return _pg_lt_selectivity(pg_stats, predicate.value)
    if predicate.kind == "BETWEEN":
        assert predicate.value_upper is not None
        return _pg_between_selectivity(pg_stats, predicate.value, predicate.value_upper)
    raise ValueError(f"Unsupported predicate kind: {predicate.kind}")


def estimate_full_hist_selectivity(full_hist: FullHistogram, predicate: Predicate) -> float:
    dist = PiecewiseDistribution.from_components(full_hist.singletons, full_hist.ranges)
    if dist.total_mass <= EPS:
        return 0.0

    if predicate.kind == "=":
        target = _clip_value(predicate.value)
        mass = 0.0
        for singleton in full_hist.singletons:
            if abs(singleton.value - target) <= 1e-9:
                mass += singleton.frequency
        return max(0.0, min(mass / dist.total_mass, 1.0))

    if predicate.kind == "<":
        return dist.cdf_less_than(predicate.value)

    if predicate.kind == "BETWEEN":
        assert predicate.value_upper is not None
        lower = min(predicate.value, predicate.value_upper)
        upper = max(predicate.value, predicate.value_upper)
        return max(0.0, dist.cdf_less_equal(upper) - dist.cdf_less_than(lower))

    raise ValueError(f"Unsupported predicate kind: {predicate.kind}")


def build_predicates(
    full_hist: FullHistogram,
    rng: np.random.Generator,
    n_total: int = 100,
    equality_ratio: float = 0.2,
    between_ratio: float = 0.4,
) -> List[Predicate]:
    singletons = [singleton.value for singleton in full_hist.singletons]
    predicates: List[Predicate] = []

    n_eq = int(round(n_total * equality_ratio))
    n_between = int(round(n_total * between_ratio))
    n_lt = max(n_total - n_eq - n_between, 0)

    for index in range(n_eq):
        if singletons and index < max(1, n_eq // 2):
            value = float(rng.choice(singletons))
        else:
            value = float(rng.uniform(DOMAIN_MIN, DOMAIN_MAX))
        predicates.append(Predicate("=", value))

    for _ in range(n_lt):
        predicates.append(Predicate("<", float(rng.uniform(DOMAIN_MIN, DOMAIN_MAX))))

    for _ in range(n_between):
        low = float(rng.uniform(DOMAIN_MIN, DOMAIN_MAX))
        high = float(rng.uniform(DOMAIN_MIN, DOMAIN_MAX))
        predicates.append(Predicate("BETWEEN", min(low, high), max(low, high)))

    rng.shuffle(predicates)
    return predicates


def summarize_selectivity_errors(errors: Sequence[float]) -> SelectivitySummary:
    values = np.asarray(list(errors), dtype=float)
    return SelectivitySummary(
        mae=float(np.mean(values)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
        maximum=float(np.max(values)),
    )


# ============================================================================
# Phase 1: Reconstruction
# ============================================================================


def reconstruct_full_histogram(pg_stats: PostgreSQLStatistics) -> FullHistogram:
    residual_mass = _residual_mass(pg_stats)
    singletons = [SingletonBucket(float(entry.value), float(entry.frequency)) for entry in pg_stats.mcv_list]
    ranges: List[HistogramBucket] = []

    bounds = [float(bound) for bound in pg_stats.histogram_bounds]
    if len(bounds) >= 2 and residual_mass > EPS:
        bin_count = len(bounds) - 1
        per_bin_mass = residual_mass / float(bin_count)
        for left, right in zip(bounds[:-1], bounds[1:]):
            ranges.append(HistogramBucket(float(left), float(right), per_bin_mass))

    return FullHistogram(
        singletons=singletons,
        ranges=ranges,
        null_fraction=float(pg_stats.null_fraction),
        target_histogram_size=len(bounds),
        n_distinct=int(pg_stats.n_distinct),
    )


# ============================================================================
# Phase 3: Decomposition
# ============================================================================


def _estimate_residual_support_size(
    residual_singletons: Sequence[SingletonBucket],
    residual_ranges: Sequence[HistogramBucket],
    fallback_histogram_size: int,
) -> int:
    if residual_ranges:
        return max(fallback_histogram_size, 2)
    unique_singletons = len({round(singleton.value, 12) for singleton in residual_singletons})
    return max(0, unique_singletons)


def _construct_histogram_bounds(
    residual_singletons: Sequence[SingletonBucket],
    residual_ranges: Sequence[HistogramBucket],
    histogram_size: int,
) -> List[float]:
    if histogram_size < 2:
        return []

    support_size = _estimate_residual_support_size(residual_singletons, residual_ranges, histogram_size)
    if support_size < 2:
        return []

    distribution = PiecewiseDistribution.from_components(residual_singletons, residual_ranges)
    if distribution.total_mass <= EPS:
        return []

    return [
        _clip_value(distribution.quantile(level / float(histogram_size - 1)))
        for level in range(histogram_size)
    ]


def decompose_to_postgres_format(
    full_hist: FullHistogram,
    mcv_threshold: float = 0.01,
    mcv_limit: int = 100,
) -> PostgreSQLStatistics:
    mcv_candidates = [(singleton.value, singleton.frequency) for singleton in full_hist.singletons if singleton.frequency >= mcv_threshold]
    mcv_candidates.sort(key=lambda item: (-item[1], item[0]))
    mcv_candidates = mcv_candidates[:mcv_limit]

    selected_values = {value for value, _ in mcv_candidates}
    mcv_list = [MCVEntry(float(value), float(freq)) for value, freq in mcv_candidates]

    residual_singletons = [
        SingletonBucket(float(singleton.value), float(singleton.frequency))
        for singleton in full_hist.singletons
        if singleton.value not in selected_values
    ]
    residual_ranges = [HistogramBucket(float(bucket.lower), float(bucket.upper), float(bucket.frequency)) for bucket in full_hist.ranges]

    histogram_size = full_hist.target_histogram_size
    histogram_bounds = _construct_histogram_bounds(residual_singletons, residual_ranges, histogram_size)

    return PostgreSQLStatistics(
        mcv_list=mcv_list,
        histogram_bounds=histogram_bounds,
        n_distinct=max(int(full_hist.n_distinct), len(mcv_list)),
        null_fraction=float(full_hist.null_fraction),
    )


# ============================================================================
# Synthetic data generation
# ============================================================================


def generate_test_distribution(
    dist_type: str,
    n_values: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(42)

    if dist_type == "uniform":
        return rng.uniform(DOMAIN_MIN, DOMAIN_MAX, n_values)
    if dist_type == "normal":
        return np.clip(rng.normal(0.5, 0.15, n_values), DOMAIN_MIN, DOMAIN_MAX)
    if dist_type == "zipf":
        ranks = np.arange(1, 101)
        probs = 1.0 / ranks
        probs /= probs.sum()
        return rng.choice(ranks / 100.0, size=n_values, p=probs)
    if dist_type == "bimodal":
        mode1 = rng.normal(0.3, 0.05, n_values // 2)
        mode2 = rng.normal(0.7, 0.05, n_values // 2)
        return np.clip(np.concatenate([mode1, mode2]), DOMAIN_MIN, DOMAIN_MAX)

    raise ValueError(f"Unknown distribution type: {dist_type}")


def create_postgres_stats_from_data(
    data: np.ndarray,
    n_mcv: int = 10,
    n_buckets: int = 10,
) -> PostgreSQLStatistics:
    total_count = len(data)
    if total_count == 0:
        return PostgreSQLStatistics([], [], 0, 0.0)

    non_null_data = np.asarray(data, dtype=float)
    null_fraction = 0.0
    unique_values, counts = np.unique(non_null_data, return_counts=True)
    n_distinct = int(len(unique_values))

    mcv_candidates = [index for index, count in enumerate(counts) if count > 1]
    mcv_candidates.sort(key=lambda index: (-counts[index], float(unique_values[index])))
    selected_indices = mcv_candidates[: min(n_mcv, len(mcv_candidates))]

    mcv_values = set(float(unique_values[index]) for index in selected_indices)
    mcv_list = [
        MCVEntry(float(unique_values[index]), float(counts[index] / total_count))
        for index in selected_indices
    ]

    residual_mask = ~np.isin(non_null_data, list(mcv_values))
    residual_data = np.sort(non_null_data[residual_mask])
    residual_unique = np.unique(residual_data)

    histogram_bounds: List[float] = []
    if len(residual_unique) >= 2:
        histogram_size = min(len(residual_unique), n_buckets + 1)
        positions = _histogram_positions(len(residual_data), histogram_size)
        histogram_bounds = [float(residual_data[pos]) for pos in positions]

    return PostgreSQLStatistics(
        mcv_list=mcv_list,
        histogram_bounds=histogram_bounds,
        n_distinct=n_distinct,
        null_fraction=null_fraction,
    )


def apply_simulated_correction(
    full_hist: FullHistogram,
    rng: np.random.Generator,
    max_relative_shift: float = 0.08,
) -> FullHistogram:
    components: List[Tuple[str, float, float, float]] = []
    for singleton in full_hist.singletons:
        components.append(("singleton", float(singleton.value), 0.0, float(singleton.frequency)))
    for bucket in full_hist.ranges:
        components.append(("range", float(bucket.lower), float(bucket.upper), float(bucket.frequency)))

    if not components:
        return FullHistogram([], [], full_hist.null_fraction, full_hist.target_histogram_size, full_hist.n_distinct)

    masses = np.array([component[3] for component in components], dtype=float)
    shifts = rng.uniform(-max_relative_shift, max_relative_shift, size=len(masses))
    updated = np.clip(masses * (1.0 + shifts), 1e-9, None)
    updated *= full_hist.total_non_null_frequency() / float(updated.sum())

    singletons: List[SingletonBucket] = []
    ranges: List[HistogramBucket] = []
    for (kind, first, second, _), freq in zip(components, updated):
        if kind == "singleton":
            singletons.append(SingletonBucket(first, float(freq)))
        else:
            ranges.append(HistogramBucket(first, second, float(freq)))

    return FullHistogram(
        singletons=singletons,
        ranges=ranges,
        null_fraction=full_hist.null_fraction,
        target_histogram_size=full_hist.target_histogram_size,
        n_distinct=full_hist.n_distinct,
    )


# ============================================================================
# Experiment 1: Round-trip accuracy
# ============================================================================


def _sample_residual_quantile_mae(original: PostgreSQLStatistics, reconstructed: PostgreSQLStatistics) -> float:
    if len(original.histogram_bounds) < 2 and len(reconstructed.histogram_bounds) < 2:
        return 0.0

    original_full = reconstruct_full_histogram(original)
    reconstructed_full = reconstruct_full_histogram(reconstructed)
    original_dist = PiecewiseDistribution.from_components([], original_full.ranges)
    reconstructed_dist = PiecewiseDistribution.from_components([], reconstructed_full.ranges)

    if original_dist.total_mass <= EPS and reconstructed_dist.total_mass <= EPS:
        return 0.0

    levels = np.linspace(0.0, 1.0, 101)
    errors = [
        abs(original_dist.quantile(float(level)) - reconstructed_dist.quantile(float(level)))
        for level in levels
    ]
    return float(np.mean(errors))


def measure_round_trip_error(pg_stats: PostgreSQLStatistics) -> Dict[str, float]:
    original_mcv_mass = sum(entry.frequency for entry in pg_stats.mcv_list)
    original_residual_mass = _residual_mass(pg_stats)
    original_total = original_mcv_mass + original_residual_mass

    full_hist = reconstruct_full_histogram(pg_stats)
    reconstructed = decompose_to_postgres_format(full_hist, mcv_threshold=0.0)

    reconstructed_mcv_mass = sum(entry.frequency for entry in reconstructed.mcv_list)
    reconstructed_residual_mass = _residual_mass(reconstructed)
    reconstructed_total = reconstructed_mcv_mass + reconstructed_residual_mass

    return {
        "total_freq_error": abs(original_total - reconstructed_total),
        "mcv_mass_error": abs(original_mcv_mass - reconstructed_mcv_mass),
        "residual_mass_error": abs(original_residual_mass - reconstructed_residual_mass),
        "residual_quantile_mae": _sample_residual_quantile_mae(pg_stats, reconstructed),
        "mcv_count_diff": abs(len(pg_stats.mcv_list) - len(reconstructed.mcv_list)),
        "hist_size_diff": abs(len(pg_stats.histogram_bounds) - len(reconstructed.histogram_bounds)),
    }


def experiment_round_trip_accuracy() -> Dict[str, object]:
    print("=" * 80)
    print("Experiment 1: Round-Trip Accuracy")
    print("=" * 80)

    distributions = ["uniform", "normal", "zipf", "bimodal"]
    trials_per_dist = 5

    per_distribution = []
    total_errors: List[float] = []
    mcv_errors: List[float] = []
    residual_errors: List[float] = []
    quantile_maes: List[float] = []

    for dist_idx, dist_type in enumerate(distributions):
        trial_results = []
        for trial in range(trials_per_dist):
            rng = np.random.default_rng(10_000 + dist_idx * 100 + trial)
            data = generate_test_distribution(dist_type, n_values=10000, rng=rng)
            pg_stats = create_postgres_stats_from_data(data, n_mcv=20, n_buckets=20)
            trial_results.append(measure_round_trip_error(pg_stats))

        row = {
            "distribution": dist_type,
            "total_freq_error_mean": float(np.mean([result["total_freq_error"] for result in trial_results])),
            "total_freq_error_max": float(np.max([result["total_freq_error"] for result in trial_results])),
            "mcv_mass_error_mean": float(np.mean([result["mcv_mass_error"] for result in trial_results])),
            "mcv_mass_error_max": float(np.max([result["mcv_mass_error"] for result in trial_results])),
            "residual_mass_error_mean": float(np.mean([result["residual_mass_error"] for result in trial_results])),
            "residual_mass_error_max": float(np.max([result["residual_mass_error"] for result in trial_results])),
            "residual_quantile_mae_mean": float(np.mean([result["residual_quantile_mae"] for result in trial_results])),
            "residual_quantile_mae_max": float(np.max([result["residual_quantile_mae"] for result in trial_results])),
            "mcv_count_diff_mean": float(np.mean([result["mcv_count_diff"] for result in trial_results])),
            "hist_size_diff_mean": float(np.mean([result["hist_size_diff"] for result in trial_results])),
        }
        per_distribution.append(row)

        total_errors.extend(result["total_freq_error"] for result in trial_results)
        mcv_errors.extend(result["mcv_mass_error"] for result in trial_results)
        residual_errors.extend(result["residual_mass_error"] for result in trial_results)
        quantile_maes.extend(result["residual_quantile_mae"] for result in trial_results)

        print(f"\nTesting {dist_type} distribution ({trials_per_dist} trials)...")
        print(f"  Total frequency error (mean/max):   {row['total_freq_error_mean']:.6e} / {row['total_freq_error_max']:.6e}")
        print(f"  MCV mass error (mean/max):          {row['mcv_mass_error_mean']:.6e} / {row['mcv_mass_error_max']:.6e}")
        print(f"  Residual quantile MAE (mean/max):   {row['residual_quantile_mae_mean']:.6e} / {row['residual_quantile_mae_max']:.6e}")

    summary = {
        "total_freq_error": _summarize(total_errors),
        "mcv_mass_error": _summarize(mcv_errors),
        "residual_mass_error": _summarize(residual_errors),
        "residual_quantile_mae": _summarize(quantile_maes),
    }

    return {
        "trials_per_distribution": trials_per_dist,
        "per_distribution": per_distribution,
        "summary": summary,
    }


# ============================================================================
# Experiment 2: Selectivity consistency
# ============================================================================


def experiment_selectivity_consistency() -> Dict[str, object]:
    print("\n" + "=" * 80)
    print("Experiment 2: Selectivity Estimation Consistency")
    print("=" * 80)

    distributions = ["uniform", "normal", "zipf", "bimodal"]
    trials_per_dist = 5
    predicates_per_trial = 100

    per_distribution = []
    all_errors: List[float] = []

    for dist_idx, dist_type in enumerate(distributions):
        trial_mae = []
        trial_p95 = []
        trial_p99 = []
        trial_max = []

        for trial in range(trials_per_dist):
            rng = np.random.default_rng(20_000 + dist_idx * 100 + trial)
            data = generate_test_distribution(dist_type, n_values=10000, rng=rng)
            pg_stats = create_postgres_stats_from_data(data, n_mcv=20, n_buckets=20)

            full_hist = reconstruct_full_histogram(pg_stats)
            reconstructed = decompose_to_postgres_format(full_hist, mcv_threshold=0.0)
            predicates = build_predicates(full_hist, rng, n_total=predicates_per_trial)

            errors = [
                abs(estimate_full_hist_selectivity(full_hist, predicate) - estimate_pg_selectivity(reconstructed, predicate))
                for predicate in predicates
            ]
            summary = summarize_selectivity_errors(errors)
            trial_mae.append(summary.mae)
            trial_p95.append(summary.p95)
            trial_p99.append(summary.p99)
            trial_max.append(summary.maximum)
            all_errors.extend(errors)

        row = {
            "distribution": dist_type,
            "mae_mean": float(np.mean(trial_mae)),
            "mae_max": float(np.max(trial_mae)),
            "p95_mean": float(np.mean(trial_p95)),
            "p99_mean": float(np.mean(trial_p99)),
            "max_mean": float(np.mean(trial_max)),
        }
        per_distribution.append(row)

        print(f"\n{dist_type}:")
        print(f"  MAE (mean/max over trials): {row['mae_mean']:.6e} / {row['mae_max']:.6e}")
        print(f"  95th percentile (mean):     {row['p95_mean']:.6e}")
        print(f"  99th percentile (mean):     {row['p99_mean']:.6e}")

    summary = {
        "mae": _summarize(all_errors),
        "p95_global": float(np.percentile(all_errors, 95)),
        "p99_global": float(np.percentile(all_errors, 99)),
    }

    print("\nSelectivity estimation error summary:")
    print(f"  Global MAE: {summary['mae']['mean']:.6e}")
    print(f"  Global max: {summary['mae']['max']:.6e}")
    print(f"  Global 95th percentile: {summary['p95_global']:.6e}")
    print(f"  Global 99th percentile: {summary['p99_global']:.6e}")

    return {
        "trials_per_distribution": trials_per_dist,
        "predicates_per_trial": predicates_per_trial,
        "per_distribution": per_distribution,
        "summary": summary,
    }


# ============================================================================
# Experiment 3: Performance overhead
# ============================================================================


def experiment_performance_overhead() -> Dict[str, object]:
    print("\n" + "=" * 80)
    print("Experiment 3: Performance Overhead")
    print("=" * 80)

    n_trials = 1000
    mcv_sizes = [10, 50, 100]
    bucket_counts = [10, 20, 50]
    configs = []

    for n_mcv in mcv_sizes:
        for n_buckets in bucket_counts:
            rng = np.random.default_rng(30_000 + n_mcv * 10 + n_buckets)
            data = generate_test_distribution("zipf", n_values=10000, rng=rng)
            pg_stats = create_postgres_stats_from_data(data, n_mcv=n_mcv, n_buckets=n_buckets)
            full_hist = reconstruct_full_histogram(pg_stats)

            recon_times = []
            for _ in range(n_trials):
                start = time.perf_counter()
                _ = reconstruct_full_histogram(pg_stats)
                recon_times.append((time.perf_counter() - start) * 1000.0)

            decomp_times = []
            for _ in range(n_trials):
                start = time.perf_counter()
                _ = decompose_to_postgres_format(full_hist)
                decomp_times.append((time.perf_counter() - start) * 1000.0)

            row = {
                "n_mcv": int(n_mcv),
                "n_buckets": int(n_buckets),
                "recon_mean_ms": float(np.mean(recon_times)),
                "recon_std_ms": float(np.std(recon_times)),
                "decomp_mean_ms": float(np.mean(decomp_times)),
                "decomp_std_ms": float(np.std(decomp_times)),
                "total_mean_ms": float(np.mean(recon_times) + np.mean(decomp_times)),
            }
            configs.append(row)

    print("\nPerformance Results (milliseconds):")
    print(f"{'Configuration':<30} {'Recon (ms)':<15} {'Decomp (ms)':<15} {'Total (ms)':<15}")
    print("-" * 75)
    for row in configs:
        label = f"MCV={row['n_mcv']}, Buckets={row['n_buckets']}"
        print(
            f"{label:<30} "
            f"{row['recon_mean_ms']:>6.4f}±{row['recon_std_ms']:<6.4f} "
            f"{row['decomp_mean_ms']:>6.4f}±{row['decomp_std_ms']:<6.4f} "
            f"{row['total_mean_ms']:>6.4f}"
        )

    total_times = [row["total_mean_ms"] for row in configs]
    summary = {
        "n_trials": n_trials,
        "total_time_ms": _summarize(total_times),
    }
    return {"configs": configs, "summary": summary}


# ============================================================================
# Experiment 4: Threshold sensitivity
# ============================================================================


def experiment_mcv_threshold_sensitivity() -> Dict[str, object]:
    print("\n" + "=" * 80)
    print("Experiment 4: MCV Threshold Sensitivity")
    print("=" * 80)

    base_rng = np.random.default_rng(40_000)
    data = generate_test_distribution("zipf", n_values=10000, rng=base_rng)
    pg_stats = create_postgres_stats_from_data(data, n_mcv=20, n_buckets=20)
    reconstructed = reconstruct_full_histogram(pg_stats)
    corrected_full = apply_simulated_correction(reconstructed, base_rng)

    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
    predicates = build_predicates(corrected_full, base_rng, n_total=120, equality_ratio=0.5, between_ratio=0.25)
    points = []

    print(f"\n{'Threshold':<12} {'MCV Count':<12} {'Freq Error':<14} {'Selectivity MAE':<18} {'P99 Error':<18}")
    print("-" * 80)

    for threshold in thresholds:
        decomp = decompose_to_postgres_format(corrected_full, mcv_threshold=threshold)
        estimated_total = sum(entry.frequency for entry in decomp.mcv_list) + _residual_mass(decomp)
        freq_error = abs(corrected_full.total_non_null_frequency() - estimated_total)

        errors = [
            abs(estimate_full_hist_selectivity(corrected_full, predicate) - estimate_pg_selectivity(decomp, predicate))
            for predicate in predicates
        ]
        summary = summarize_selectivity_errors(errors)

        row = {
            "threshold": float(threshold),
            "mcv_count": int(len(decomp.mcv_list)),
            "freq_error": float(freq_error),
            "selectivity_mae": summary.mae,
            "selectivity_p99": summary.p99,
        }
        points.append(row)

        print(
            f"{threshold:<12.4f} {row['mcv_count']:<12} {row['freq_error']:<14.6e} "
            f"{row['selectivity_mae']:<18.6e} {row['selectivity_p99']:<18.6e}"
        )

    return {"points": points}


# ============================================================================
# Result export / driver
# ============================================================================


def save_results(results: Dict[str, object]) -> Path:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mcv_validation_results.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    return out_path


def run_all_experiments() -> Dict[str, object]:
    print("\n" + "=" * 80)
    print("MCV ADAPTER VALIDATION EXPERIMENTS")
    print("=" * 80)
    print("\nValidating the PostgreSQL adapter pipeline:")
    print("  Phase 1: pg_statistic MCV + histogram bounds -> canonical full distribution")
    print("  Phase 2: correction stress test via simulated perturbation (threshold study only)")
    print("  Phase 3: canonical full distribution -> pg_statistic MCV + histogram bounds")
    print("=" * 80)

    exp1 = experiment_round_trip_accuracy()
    exp2 = experiment_selectivity_consistency()
    exp3 = experiment_performance_overhead()
    exp4 = experiment_mcv_threshold_sensitivity()

    results = {
        "round_trip_accuracy": exp1,
        "selectivity_consistency": exp2,
        "performance_overhead": exp3,
        "threshold_sensitivity": exp4,
    }
    out_path = save_results(results)

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"✓ Round-trip accuracy: max total mass error = {exp1['summary']['total_freq_error']['max']:.6e}")
    print(f"✓ Residual quantile fidelity: mean MAE = {exp1['summary']['residual_quantile_mae']['mean']:.6e}")
    print(f"✓ Selectivity consistency: global MAE = {exp2['summary']['mae']['mean']:.6e}")
    print(f"✓ Performance overhead: max mean round-trip time = {exp3['summary']['total_time_ms']['max']:.4f} ms")
    print(f"✓ Results written to: {out_path}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_all_experiments()
