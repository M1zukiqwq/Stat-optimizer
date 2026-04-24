from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from cdf_teacher import correct_quantiles
from histogram_math import clamp01, project_monotonic
from histogram_types import FeedbackObservation, KllFeedbackSample

PREDICATE_ORDER = ["<", "<=", ">", ">=", "=", "BETWEEN"]
OBSERVATION_FEATURE_DIM = len(PREDICATE_ORDER) + 7        # 6-hot + value_norm, value_upper_norm, est_sel, act_sel, time_decay, has_upper, span_norm
OBSERVATION_FEATURE_DIM_NO_TS = len(PREDICATE_ORDER) + 6  # same as above but without time_decay
MAX_OBSERVATIONS_DEFAULT = 16


@dataclass
class TensorRecord:
    feature_tensor: List[float]
    observation_tensor: List[List[float]]
    mask_tensor: List[float]
    target_tensor: Optional[List[float]]
    metadata: Dict[str, object]



def _normalize_value(value: float, lower: float, upper: float) -> float:
    denominator = max(upper - lower, 1e-12)
    return clamp01((value - lower) / denominator)


def _now_from_sample(sample: KllFeedbackSample) -> datetime:
    if sample.observations:
        return max(obs.timestamp for obs in sample.observations)
    return datetime.now(tz=timezone.utc)


def _encode_observation(
    observation: FeedbackObservation,
    sample: KllFeedbackSample,
    now: datetime,
    decay_lambda: float,
    use_time_decay: bool = True,
) -> List[float]:
    one_hot = [1.0 if observation.predicate_type == predicate else 0.0 for predicate in PREDICATE_ORDER]

    lower = sample.prior.min_value
    upper = sample.prior.max_value
    value_norm = _normalize_value(observation.value, lower, upper)

    if observation.value_upper is None:
        value_upper_norm = 0.0
        has_upper = 0.0
        span_norm = 0.0
    else:
        value_upper_norm = _normalize_value(observation.value_upper, lower, upper)
        has_upper = 1.0
        span_norm = clamp01(value_upper_norm - value_norm)

    # Reconstruct the CBO "estimated_selectivity" using the prior CDF
    # This prevents the label leakage bug where the Java engine was 
    # mistakenly logging the overall runtime selectivity as estimated_sel.
    from cdf_teacher import evaluate_piecewise_cdf
    prior_x = [sample.prior.min_value] + list(sample.prior.quantile_values) + [sample.prior.max_value]
    prior_p = [0.0] + list(sample.prior.quantile_levels) + [1.0]

    if observation.predicate_type == "BETWEEN":
        val = observation.value
        val_upper = observation.value_upper
        est_cond = clamp01(
            evaluate_piecewise_cdf(prior_x, prior_p, val_upper)
            - evaluate_piecewise_cdf(prior_x, prior_p, val)
        )
    elif observation.predicate_type in {"<", "<="}:
        est_cond = clamp01(evaluate_piecewise_cdf(prior_x, prior_p, observation.value))
    elif observation.predicate_type in {">", ">="}:
        est_cond = clamp01(1.0 - evaluate_piecewise_cdf(prior_x, prior_p, observation.value))
    else:  # "="
        width = 0.01  # as modeled in simulate_memory_kll_dataset.py
        left = max(lower, observation.value - width)
        right = min(upper, observation.value + width)
        est_cond = clamp01(
            evaluate_piecewise_cdf(prior_x, prior_p, right)
            - evaluate_piecewise_cdf(prior_x, prior_p, left)
        )
    estimated_selectivity = clamp01(est_cond * max(1.0 - sample.prior.null_fraction, 1e-6))

    age_seconds = max((now - observation.timestamp).total_seconds(), 0.0)
    time_decay = math.exp(-decay_lambda * age_seconds)

    numerical = [
        value_norm,
        value_upper_norm,
        estimated_selectivity,
        observation.actual_selectivity,
    ]
    if use_time_decay:
        numerical.append(time_decay)  # only included when time-decay is enabled
    numerical += [
        has_upper,  # 0.0 或 1.0，标识是否存在 value_upper
        span_norm,  # BETWEEN 区间宽度（归一化到 [0,1]），否则 0.0
    ]
    return one_hot + numerical


def _prepare_target(
    sample: KllFeedbackSample,
    corrected_quantiles: List[float],
) -> List[float]:
    lower = sample.prior.min_value
    upper = sample.prior.max_value
    normalized = [_normalize_value(value, lower, upper) for value in corrected_quantiles]
    monotonic = project_monotonic(normalized)
    return monotonic


def tensorize_sample(
    sample: KllFeedbackSample,
    max_observations: int = MAX_OBSERVATIONS_DEFAULT,
    decay_lambda: float = 1.0 / (14.0 * 24.0 * 3600.0),
    teacher_fn: Optional[Callable[[KllFeedbackSample], List[float]]] = correct_quantiles,
    use_time_decay: bool = True,
) -> TensorRecord:
    """Tensorize a feedback sample.

    Args:
        use_time_decay: if False, the time-decay feature is dropped from each
            observation vector (dim shrinks by 1).  Use this for the timestamp
            ablation study.
    """
    now = _now_from_sample(sample)

    # Take the K most recent observations, then sort them chronologically
    # (oldest → newest within the window).
    # Rationale: sequence models (LSTM, Transformer) expect causal order;
    # putting the newest observation last avoids direction confusion.
    # The per-observation ``time_decay`` feature still encodes absolute recency.
    all_sorted = sorted(sample.observations, key=lambda obs: obs.timestamp)
    selected = all_sorted[-max_observations:]  # K most recent, now in time order

    obs_dim = OBSERVATION_FEATURE_DIM if use_time_decay else OBSERVATION_FEATURE_DIM_NO_TS
    observation_tensor: List[List[float]] = []
    mask_tensor: List[float] = []

    for obs in selected:
        observation_tensor.append(
            _encode_observation(obs, sample, now=now, decay_lambda=decay_lambda, use_time_decay=use_time_decay)
        )
        mask_tensor.append(1.0)

    while len(observation_tensor) < max_observations:
        observation_tensor.append([0.0] * obs_dim)
        mask_tensor.append(0.0)

    prior = sample.prior
    prior_norm = [_normalize_value(value, prior.min_value, prior.max_value) for value in prior.quantile_values]

    bucket_count = len(prior.quantile_levels) + 1  # B = (B-1 个内部分位点) + 1
    meta = [
        prior.null_fraction,
        min(len(sample.observations), max_observations) / float(max_observations),
        min(bucket_count / 64.0, 1.0),  # 使用真实桶数 B，与 JSON pipeline 统一
    ]

    flattened_observations = [value for row in observation_tensor for value in row]
    feature_tensor = prior_norm + meta + flattened_observations + mask_tensor

    target_tensor: Optional[List[float]] = None
    if sample.corrected_quantile_values is not None:
        target_tensor = _prepare_target(sample, sample.corrected_quantile_values)
    elif teacher_fn is not None:
        target_tensor = _prepare_target(sample, teacher_fn(sample))

    return TensorRecord(
        feature_tensor=feature_tensor,
        observation_tensor=observation_tensor,
        mask_tensor=mask_tensor,
        target_tensor=target_tensor,
        metadata={
            "source_path": sample.source_path,
            "min_value": prior.min_value,
            "max_value": prior.max_value,
            "quantile_levels": prior.quantile_levels,
            "value_type": prior.value_type,
            "sketch_k": prior.sketch_k,
            "max_observations": max_observations,
            "observation_feature_dim": obs_dim,
            "use_time_decay": use_time_decay,
        },
    )
