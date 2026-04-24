from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from kll_codec import clamp, normalize_quantiles

DEFAULT_KLL_INTERNAL_POINTS = 9
DEFAULT_QUANTILE_LEVELS = [
    index / (DEFAULT_KLL_INTERNAL_POINTS + 1)
    for index in range(1, DEFAULT_KLL_INTERNAL_POINTS + 1)
]
SUPPORTED_PREDICATES = {"<", "<=", ">", ">=", "=", "BETWEEN"}


def _clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


@dataclass
class KllPrior:
    min_value: float
    max_value: float
    null_fraction: float
    quantile_levels: List[float]
    quantile_values: List[float]
    value_type: str = "double"
    sketch_k: int = 1024
    sketch_bytes_base64: Optional[str] = None

    def __post_init__(self) -> None:
        tolerance = 1e-9
        if abs(self.min_value - 0.0) > tolerance or abs(self.max_value - 1.0) > tolerance:
            raise ValueError("normalized kll prior is required: min_value must be 0 and max_value must be 1")
        if self.min_value > self.max_value:
            raise ValueError("min_value must be <= max_value")

        self.null_fraction = _clamp01(self.null_fraction)

        if len(self.quantile_levels) != len(self.quantile_values):
            raise ValueError("quantile_levels and quantile_values must have same length")
        if not self.quantile_levels:
            raise ValueError("quantile_levels is empty")

        paired = sorted(zip(self.quantile_levels, self.quantile_values), key=lambda item: item[0])
        self.quantile_levels = [float(level) for level, _ in paired]
        self.quantile_values = [float(value) for _, value in paired]

        # 使用 >= 严格检查（排序后相等说明有重复 level，属于非法输入）
        for left, right in zip(self.quantile_levels, self.quantile_levels[1:]):
            if left >= right:
                raise ValueError("quantile_levels must be strictly increasing (duplicate levels are not allowed)")
        if any(level < 0.0 or level > 1.0 for level in self.quantile_levels):
            raise ValueError("quantile_levels must be within [0, 1]")

        self.quantile_values = normalize_quantiles(self.quantile_values, self.min_value, self.max_value)

    @property
    def value_range(self) -> float:
        return max(self.max_value - self.min_value, 1e-12)


@dataclass
class FeedbackObservation:
    predicate_type: str
    value: float
    value_upper: Optional[float]
    actual_selectivity: float
    timestamp: datetime
    estimated_selectivity: Optional[float] = None

    def __post_init__(self) -> None:
        self.predicate_type = self.predicate_type.upper()
        if self.predicate_type not in SUPPORTED_PREDICATES:
            raise ValueError(f"unsupported predicate_type: {self.predicate_type}")

        self.value = _clamp01(self.value)
        if self.predicate_type == "BETWEEN" and self.value_upper is None:
            raise ValueError("BETWEEN requires value_upper")

        if self.value_upper is not None:
            self.value_upper = _clamp01(self.value_upper)
            if self.value > self.value_upper:
                self.value, self.value_upper = self.value_upper, self.value

        if self.estimated_selectivity is not None:
            self.estimated_selectivity = _clamp01(self.estimated_selectivity)
        self.actual_selectivity = _clamp01(self.actual_selectivity)
        self.timestamp = ensure_utc(self.timestamp)


@dataclass
class KllFeedbackSample:
    prior: KllPrior
    observations: List[FeedbackObservation]
    corrected_quantile_values: Optional[List[float]] = None
    source_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.observations = sorted(self.observations, key=lambda obs: obs.timestamp)
        if self.corrected_quantile_values is None:
            return
        if len(self.corrected_quantile_values) != len(self.prior.quantile_levels):
            raise ValueError("corrected_quantile_values length must match quantile levels")
        self.corrected_quantile_values = normalize_quantiles(
            self.corrected_quantile_values,
            self.prior.min_value,
            self.prior.max_value,
        )
