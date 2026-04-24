from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple


def default_quantile_levels(internal_points: int) -> List[float]:
    if internal_points <= 0:
        raise ValueError("internal_points must be positive")
    denominator = internal_points + 1
    return [index / denominator for index in range(1, denominator)]


def clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def normalize_quantiles(values: Sequence[float], min_value: float, max_value: float) -> List[float]:
    if not values:
        return []
    lower = min(min_value, max_value)
    upper = max(min_value, max_value)
    cleaned = [clamp(float(value), lower, upper) for value in values]
    cleaned.sort()
    return cleaned


def quantiles_to_bucket_boundaries(min_value: float, max_value: float, quantile_values: Sequence[float]) -> List[float]:
    return [min_value] + list(quantile_values) + [max_value]


def encode_simulated_kll(
    quantile_levels: Sequence[float],
    quantile_values: Sequence[float],
    min_value: float,
    max_value: float,
    value_type: str = "double",
    sketch_k: int = 1024,
) -> str:
    payload = {
        "format": "presto-cdf-simulation-kll-v1",
        "type": value_type,
        "min": min_value,
        "max": max_value,
        "k": sketch_k,
        "quantile_levels": [float(level) for level in quantile_levels],
        "quantile_values": [float(value) for value in quantile_values],
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def decode_simulated_kll(sketch_bytes_base64: str) -> Optional[Dict[str, Any]]:
    if not sketch_bytes_base64:
        return None
    try:
        raw = base64.b64decode(sketch_bytes_base64)
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("format") != "presto-cdf-simulation-kll-v1":
        return None
    return payload


def parse_kll_quantiles_from_payload(payload: Dict[str, Any]) -> Optional[Tuple[List[float], List[float]]]:
    levels = payload.get("quantile_levels")
    values = payload.get("quantile_values")
    if not isinstance(levels, list) or not isinstance(values, list):
        return None
    if len(levels) != len(values):
        return None

    cleaned_levels = [float(level) for level in levels]
    cleaned_values = [float(value) for value in values]
    return cleaned_levels, cleaned_values
