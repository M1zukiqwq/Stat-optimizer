from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def linear_interpolate(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    if x1 == x0:
        return y1
    ratio = (x - x0) / (x1 - x0)
    return y0 + ratio * (y1 - y0)


def evaluate_piecewise_cdf(cdf_x: Sequence[float], cdf_p: Sequence[float], value: float) -> float:
    if not cdf_x or len(cdf_x) != len(cdf_p):
        raise ValueError("cdf_x and cdf_p must be non-empty and have same length")

    if value <= cdf_x[0]:
        return clamp01(cdf_p[0])
    if value >= cdf_x[-1]:
        return clamp01(cdf_p[-1])

    left = 0
    right = len(cdf_x) - 1
    while left + 1 < right:
        mid = (left + right) // 2
        if cdf_x[mid] <= value:
            left = mid
        else:
            right = mid

    return clamp01(linear_interpolate(cdf_x[left], cdf_p[left], cdf_x[right], cdf_p[right], value))


def inverse_piecewise_cdf(cdf_x: Sequence[float], cdf_p: Sequence[float], prob: float) -> float:
    if not cdf_x or len(cdf_x) != len(cdf_p):
        raise ValueError("cdf_x and cdf_p must be non-empty and have same length")

    target = clamp01(prob)
    if target <= cdf_p[0]:
        return cdf_x[0]
    if target >= cdf_p[-1]:
        return cdf_x[-1]

    for idx in range(1, len(cdf_p)):
        if cdf_p[idx] >= target:
            prev_prob = cdf_p[idx - 1]
            prev_x = cdf_x[idx - 1]
            current_prob = cdf_p[idx]
            current_x = cdf_x[idx]
            if current_prob == prev_prob:
                return current_x
            return linear_interpolate(prev_prob, prev_x, current_prob, current_x, target)

    return cdf_x[-1]


def weighted_isotonic_regression(values: Sequence[float], weights: Sequence[float]) -> List[float]:
    if len(values) != len(weights):
        raise ValueError("values and weights must have same length")
    if not values:
        return []

    blocks: List[dict] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        w = max(weight, 1e-8)
        block = {
            "start": index,
            "end": index,
            "weighted_sum": value * w,
            "weight_sum": w,
        }
        blocks.append(block)

        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = left["weighted_sum"] / left["weight_sum"]
            right_mean = right["weighted_sum"] / right["weight_sum"]
            if left_mean <= right_mean:
                break
            merged = {
                "start": left["start"],
                "end": right["end"],
                "weighted_sum": left["weighted_sum"] + right["weighted_sum"],
                "weight_sum": left["weight_sum"] + right["weight_sum"],
            }
            blocks.pop()
            blocks.pop()
            blocks.append(merged)

    fitted = [0.0] * len(values)
    for block in blocks:
        mean = clamp01(block["weighted_sum"] / block["weight_sum"])
        for idx in range(block["start"], block["end"] + 1):
            fitted[idx] = mean
    return fitted


def project_monotonic(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    fitted = weighted_isotonic_regression([clamp01(v) for v in values], [1.0] * len(values))
    projected = [clamp01(v) for v in fitted]
    for idx in range(1, len(projected)):
        if projected[idx] < projected[idx - 1]:
            projected[idx] = projected[idx - 1]
    return projected


def merge_duplicate_x(points: Iterable[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    sorted_points = sorted(points, key=lambda item: item[0])
    if not sorted_points:
        return []

    merged: List[Tuple[float, float, float]] = []
    current_x, current_sum, current_w = sorted_points[0][0], sorted_points[0][1] * sorted_points[0][2], sorted_points[0][2]
    for x, prob, weight in sorted_points[1:]:
        if x == current_x:
            current_sum += prob * weight
            current_w += weight
            continue
        merged.append((current_x, clamp01(current_sum / max(current_w, 1e-8)), max(current_w, 1e-8)))
        current_x = x
        current_sum = prob * weight
        current_w = weight
    merged.append((current_x, clamp01(current_sum / max(current_w, 1e-8)), max(current_w, 1e-8)))
    return merged
