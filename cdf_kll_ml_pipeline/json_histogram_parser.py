from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from histogram_types import (
    DEFAULT_QUANTILE_LEVELS,
    FeedbackObservation,
    KllFeedbackSample,
    KllPrior,
)
from kll_codec import decode_simulated_kll, parse_kll_quantiles_from_payload


class ParseError(ValueError):
    pass


def _to_datetime(raw: str) -> datetime:
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _get_float(item: Dict[str, Any], keys: List[str], default: Optional[float] = None) -> float:
    for key in keys:
        if key in item and item[key] is not None:
            return float(item[key])
    if default is None:
        raise ParseError(f"missing numeric field, expected one of: {keys}")
    return float(default)


def _get_optional_float(item: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        if key in item and item[key] is not None and item[key] != "":
            return float(item[key])
    return None


def _get_str(item: Dict[str, Any], keys: List[str], default: Optional[str] = None) -> str:
    for key in keys:
        if key in item and item[key] is not None:
            return str(item[key])
    if default is None:
        raise ParseError(f"missing string field, expected one of: {keys}")
    return default


def _from_boundaries(boundaries: List[float]) -> (List[float], List[float]):
    if len(boundaries) < 2:
        raise ParseError("boundaries must include at least two points")
    bucket_count = len(boundaries) - 1
    levels = [index / bucket_count for index in range(1, bucket_count)]
    values = boundaries[1:-1]
    return levels, values


def _parse_prior(payload: Dict[str, Any]) -> KllPrior:
    prior_kll = payload.get("prior_kll")
    prior_histogram = payload.get("prior_histogram")

    prior_source = None
    if isinstance(prior_kll, dict):
        prior_source = prior_kll
    elif isinstance(prior_histogram, dict):
        prior_source = prior_histogram

    if prior_source is None:
        raise ParseError("missing object field: prior_kll or prior_histogram")

    min_value = _get_float(prior_source, ["min", "min_value"], default=0.0)
    max_value = _get_float(prior_source, ["max", "max_value"], default=1.0)
    null_fraction = _get_float(prior_source, ["null_fraction"], default=0.0)
    value_type = _get_str(prior_source, ["type", "value_type"], default="double")
    sketch_k = int(_get_float(prior_source, ["k", "sketch_k"], default=1024.0))

    quantile_levels: Optional[List[float]] = None
    quantile_values: Optional[List[float]] = None

    direct_levels = prior_source.get("quantile_levels")
    direct_values = prior_source.get("quantile_values")
    if isinstance(direct_levels, list) and isinstance(direct_values, list):
        quantile_levels = [float(level) for level in direct_levels]
        quantile_values = [float(value) for value in direct_values]

    if quantile_levels is None or quantile_values is None:
        boundaries = prior_source.get("bucket_boundaries") or prior_source.get("boundaries")
        if isinstance(boundaries, list) and len(boundaries) >= 2:
            quantile_levels, quantile_values = _from_boundaries([float(value) for value in boundaries])

    sketch_bytes = prior_source.get("sketch_bytes_base64")
    if (quantile_levels is None or quantile_values is None) and isinstance(sketch_bytes, str):
        decoded = decode_simulated_kll(sketch_bytes)
        if decoded is not None:
            parsed = parse_kll_quantiles_from_payload(decoded)
            if parsed is not None:
                quantile_levels, quantile_values = parsed

    if quantile_levels is None or quantile_values is None:
        import sys
        print(
            "[json_histogram_parser] WARNING: unable to parse quantile info from prior_kll/prior_histogram; "
            "falling back to uniform distribution [0.1, ..., 0.9]. "
            "This sample's prior quality is degraded.",
            file=sys.stderr,
        )
        quantile_levels = list(DEFAULT_QUANTILE_LEVELS)
        quantile_values = list(DEFAULT_QUANTILE_LEVELS)

    if len(quantile_levels) != len(quantile_values):
        raise ParseError("prior quantile_levels and quantile_values length mismatch")

    return KllPrior(
        min_value=min_value,
        max_value=max_value,
        null_fraction=null_fraction,
        quantile_levels=quantile_levels,
        quantile_values=quantile_values,
        value_type=value_type,
        sketch_k=sketch_k,
        sketch_bytes_base64=sketch_bytes if isinstance(sketch_bytes, str) else None,
    )


def _parse_observations(payload: Dict[str, Any]) -> List[FeedbackObservation]:
    observations = payload.get("observations") or payload.get("feedbacks") or []
    if not isinstance(observations, list):
        raise ParseError("observations must be a list")

    result: List[FeedbackObservation] = []
    for item in observations:
        if not isinstance(item, dict):
            continue
        try:
            predicate = _get_str(item, ["predicate_type", "predicate"])
            value = _get_float(item, ["value", "predicate_value"])
            value_upper = _get_optional_float(item, ["value_upper", "upper", "predicate_value_upper"])
            estimated_sel = _get_float(item, ["estimated_sel", "estimated_selectivity"], default=0.0)
            actual_sel = _get_float(item, ["actual_sel", "actual_selectivity"])
            timestamp = _to_datetime(_get_str(item, ["timestamp", "query_timestamp"]))

            result.append(
                FeedbackObservation(
                    predicate_type=predicate,
                    value=value,
                    value_upper=value_upper,
                    estimated_selectivity=estimated_sel,
                    actual_selectivity=actual_sel,
                    timestamp=timestamp,
                )
            )
        except (ParseError, ValueError) as exc:
            import sys
            print(f"[json_histogram_parser] skipping invalid observation: {exc}; item={item}", file=sys.stderr)
    return result


def _parse_corrected_quantiles(payload: Dict[str, Any]) -> Optional[List[float]]:
    corrected_kll = payload.get("corrected_kll")
    if isinstance(corrected_kll, dict):
        values = corrected_kll.get("quantile_values")
        if isinstance(values, list):
            return [float(value) for value in values]

        boundaries = corrected_kll.get("bucket_boundaries")
        if isinstance(boundaries, list) and len(boundaries) >= 2:
            return [float(value) for value in boundaries[1:-1]]

    corrected_histogram = payload.get("corrected_histogram")
    if isinstance(corrected_histogram, dict):
        boundaries = corrected_histogram.get("bucket_boundaries") or corrected_histogram.get("boundaries")
        if isinstance(boundaries, list) and len(boundaries) >= 2:
            return [float(value) for value in boundaries[1:-1]]

    direct = payload.get("corrected_quantile_values")
    if isinstance(direct, list):
        return [float(value) for value in direct]

    return None


def load_feedback_sample(path: str) -> KllFeedbackSample:
    file_path = Path(path)
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ParseError("top-level json must be an object")

    prior = _parse_prior(payload)
    observations = _parse_observations(payload)
    corrected_quantiles = _parse_corrected_quantiles(payload)

    return KllFeedbackSample(
        prior=prior,
        observations=observations,
        corrected_quantile_values=corrected_quantiles,
        source_path=str(file_path.resolve()),
    )
