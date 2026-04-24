from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Callable, List, Tuple

from histogram_math import (
    clamp,
    clamp01,
    evaluate_piecewise_cdf,
    inverse_piecewise_cdf,
    merge_duplicate_x,
    weighted_isotonic_regression,
)
from histogram_types import FeedbackObservation, KllFeedbackSample, KllPrior


def _now_from_sample(sample: KllFeedbackSample) -> datetime:
    if sample.observations:
        return max(obs.timestamp for obs in sample.observations)
    return datetime.now(tz=timezone.utc)


def build_prior_cdf(prior: KllPrior) -> Tuple[List[float], List[float]]:
    cdf_x = [prior.min_value] + list(prior.quantile_values) + [prior.max_value]
    cdf_p = [0.0] + list(prior.quantile_levels) + [1.0]

    paired = sorted(zip(cdf_x, cdf_p), key=lambda item: item[0])
    merged_x: List[float] = []
    merged_p: List[float] = []
    for value, prob in paired:
        if merged_x and value == merged_x[-1]:
            merged_p[-1] = max(merged_p[-1], clamp01(prob))
            continue
        merged_x.append(value)
        merged_p.append(clamp01(prob))

    for idx in range(1, len(merged_p)):
        if merged_p[idx] < merged_p[idx - 1]:
            merged_p[idx] = merged_p[idx - 1]
    return merged_x, merged_p


def _observation_weight(now: datetime, observation: FeedbackObservation, decay_lambda: float) -> float:
    age_seconds = max((now - observation.timestamp).total_seconds(), 0.0)
    return math.exp(-decay_lambda * age_seconds)


def _safe_non_null_fraction(prior: KllPrior) -> float:
    return max(1.0 - clamp01(prior.null_fraction), 1e-6)


def _effective_actual(observation: FeedbackObservation, prior: KllPrior) -> float:
    return clamp01(observation.actual_selectivity / _safe_non_null_fraction(prior))


def _observation_to_points(
    observation: FeedbackObservation,
    prior: KllPrior,
    prior_cdf: Callable[[float], float],
    now: datetime,
    decay_lambda: float,
    equality_weight_scale: float,
) -> List[Tuple[float, float, float]]:
    points: List[Tuple[float, float, float]] = []
    weight = _observation_weight(now, observation, decay_lambda)
    actual = _effective_actual(observation, prior)

    if observation.predicate_type in {"<", "<="}:
        points.append((observation.value, actual, weight))
        return points

    if observation.predicate_type in {">", ">="}:
        points.append((observation.value, 1.0 - actual, weight))
        return points

    if observation.predicate_type == "BETWEEN":
        if observation.value_upper is None:
            return points
        low = observation.value
        high = observation.value_upper
        prior_low = prior_cdf(low)
        prior_high = prior_cdf(high)
        prior_delta = clamp01(prior_high - prior_low)
        error = actual - prior_delta
        points.append((low, clamp01(prior_low - 0.5 * error), weight))
        points.append((high, clamp01(prior_high + 0.5 * error), weight))
        return points

    if observation.predicate_type == "=":
        eps = max(1e-9, 0.005 * prior.value_range)
        low = clamp(observation.value - eps, prior.min_value, prior.max_value)
        high = clamp(observation.value + eps, prior.min_value, prior.max_value)
        prior_low = prior_cdf(low)
        prior_high = prior_cdf(high)
        prior_delta = clamp01(prior_high - prior_low)
        error = actual - prior_delta
        weak_weight = weight * equality_weight_scale
        points.append((low, clamp01(prior_low - 0.5 * error), weak_weight))
        points.append((high, clamp01(prior_high + 0.5 * error), weak_weight))
    return points


def estimate_observation_coverage(sample: KllFeedbackSample) -> float:
    """Estimate what fraction of [min_value, max_value] has observational constraint coverage.

    Strategy: interval-union coverage.
    Each non-equality observation contributes a window of width ``window_frac`` around its
    value (and around ``value_upper`` for BETWEEN).  Equality predicates are excluded
    because they are point constraints with negligible CDF coverage.

    After collecting all windows, we merge overlapping intervals and sum their lengths
    as a fraction of ``value_range``.  This correctly distinguishes:
    - Two observations at 0.1 and 0.9  → coverage ≈ 2 * window  (low, ~0.2)
    - 10 uniformly spread observations → coverage ≈ 10 * window ≈ 1.0  (high)

    The old ``max-min`` metric would return 0.8 for both cases above.
    """
    if not sample.observations:
        return 0.0

    value_range = sample.prior.value_range
    if value_range < 1e-12:
        return 0.0

    # Each observation contributes a window of this half-width on each side of its value(s).
    half_window = 0.05 * value_range

    intervals: List[tuple] = []
    for obs in sample.observations:
        if obs.predicate_type == "=":
            # Point constraints give negligible CDF coverage; skip.
            continue
        low = max(sample.prior.min_value, obs.value - half_window)
        if obs.value_upper is not None:
            # BETWEEN: window around both endpoints plus the span between them
            high = min(sample.prior.max_value, obs.value_upper + half_window)
        else:
            high = min(sample.prior.max_value, obs.value + half_window)
        intervals.append((low, high))

    if not intervals:
        return 0.0

    # Merge overlapping intervals (union) and sum their lengths
    intervals.sort(key=lambda item: item[0])
    merged_low, merged_high = intervals[0]
    total_covered = 0.0
    for low, high in intervals[1:]:
        if low <= merged_high:
            # Overlapping or touching: extend the current merged interval
            merged_high = max(merged_high, high)
        else:
            total_covered += merged_high - merged_low
            merged_low, merged_high = low, high
    total_covered += merged_high - merged_low

    return clamp01(total_covered / value_range)


def correct_quantiles(
    sample: KllFeedbackSample,
    min_observations: int = 3,
    beta: float = 0.7,
    decay_lambda: float = 1.0 / (14.0 * 24.0 * 3600.0),
    equality_weight_scale: float = 0.3,
    coverage_soft_threshold: float = 0.2,
) -> List[float]:
    prior = sample.prior
    if len(sample.observations) < min_observations:
        return list(prior.quantile_values)

    now = _now_from_sample(sample)
    prior_x, prior_p = build_prior_cdf(prior)

    def prior_cdf_fn(value: float) -> float:
        return evaluate_piecewise_cdf(prior_x, prior_p, value)

    # prior_weight: 每个先验分位点的权重。
    # 目标：先验此 B-1 个点的总权重大致与观测总权重可比较。
    # 公式： prior_weight * (B-1) ≈ beta，观测平均每条贡献约 1~2 个点权重为 1.0
    num_prior_points = max(len(prior.quantile_levels), 1)
    prior_weight = beta / num_prior_points

    points: List[Tuple[float, float, float]] = []
    for value, prob in zip(prior.quantile_values, prior.quantile_levels):
        points.append((value, prob, prior_weight))

    for obs in sample.observations:
        points.extend(
            _observation_to_points(
                obs,
                prior,
                prior_cdf_fn,
                now=now,
                decay_lambda=decay_lambda,
                equality_weight_scale=equality_weight_scale,
            )
        )

    points.append((prior.min_value, 0.0, 1e6))
    points.append((prior.max_value, 1.0, 1e6))

    merged = merge_duplicate_x(points)
    x_values = [item[0] for item in merged]
    prob_values = [item[1] for item in merged]
    weights = [item[2] for item in merged]

    fitted_probs = weighted_isotonic_regression(prob_values, weights)
    corrected = [
        inverse_piecewise_cdf(x_values, fitted_probs, level) for level in prior.quantile_levels
    ]

    for idx in range(1, len(corrected)):
        if corrected[idx] < corrected[idx - 1]:
            corrected[idx] = corrected[idx - 1]

    corrected = [clamp(value, prior.min_value, prior.max_value) for value in corrected]

    coverage = estimate_observation_coverage(sample)
    if coverage < coverage_soft_threshold:
        alpha = coverage / max(coverage_soft_threshold, 1e-12)
        corrected = [
            (1.0 - alpha) * prior_value + alpha * corrected_value
            for prior_value, corrected_value in zip(prior.quantile_values, corrected)
        ]

    return corrected
