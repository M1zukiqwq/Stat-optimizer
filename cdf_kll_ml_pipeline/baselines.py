from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from histogram_math import evaluate_piecewise_cdf


@dataclass(frozen=True)
class HyperRect:
    lower: Tuple[float, ...]
    upper: Tuple[float, ...]

    def volume(self) -> float:
        volume = 1.0
        for lo, hi in zip(self.lower, self.upper):
            volume *= max(hi - lo, 0.0)
        return volume


@dataclass
class HoleNode:
    rect: HyperRect
    mass: float
    children: List["HoleNode"] = field(default_factory=list)

    @property
    def left(self) -> float:
        return self.rect.lower[0]

    @property
    def right(self) -> float:
        return self.rect.upper[0]

    def width(self) -> float:
        return max(self.right - self.left, 0.0)

    def is_leaf(self) -> bool:
        return not self.children


def _make_leaf(left: float, right: float, mass: float) -> HoleNode:
    return HoleNode(rect=HyperRect((left,), (right,)), mass=mass)


def _collect_leaves(node: HoleNode, out: List[HoleNode]) -> None:
    if node.is_leaf():
        out.append(node)
        return
    for child in node.children:
        _collect_leaves(child, out)


def _split_node_at_points(node: HoleNode, points: Sequence[float], eps: float = 1e-12) -> None:
    if node.is_leaf():
        internal_points = sorted({point for point in points if node.left + eps < point < node.right - eps})
        if not internal_points:
            return
        boundaries = [node.left] + internal_points + [node.right]
        width = max(node.width(), eps)
        node.children = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            seg_mass = node.mass * max(right - left, 0.0) / width
            node.children.append(_make_leaf(left, right, seg_mass))
        return

    for child in node.children:
        if any(child.left + eps < point < child.right - eps for point in points):
            _split_node_at_points(child, points, eps=eps)


def _find_covering_leaf(node: HoleNode, left: float, right: float, eps: float = 1e-12) -> Optional[HoleNode]:
    if right <= node.left + eps or left >= node.right - eps:
        return None
    if node.is_leaf():
        return node
    for child in node.children:
        if child.left - eps <= left and child.right + eps >= right:
            found = _find_covering_leaf(child, left, right, eps=eps)
            if found is not None:
                return found
    return None


def _initialize_hole_tree(prior_min: float, prior_max: float, prior_quantiles: Sequence[float]) -> HoleNode:
    boundaries = [prior_min] + sorted(prior_quantiles) + [prior_max]
    root = HoleNode(rect=HyperRect((prior_min,), (prior_max,)), mass=1.0)
    bucket_mass = 1.0 / max(len(boundaries) - 1, 1)
    root.children = [_make_leaf(left, right, bucket_mass) for left, right in zip(boundaries[:-1], boundaries[1:])]
    return root


def _query_interval(prior_min: float, prior_max: float, observation: dict) -> Optional[Tuple[float, float]]:
    value = observation["value"]
    value_upper = observation.get("value_upper", value)
    predicate = observation["predicate_type"]

    if predicate in {"<", "<="}:
        left, right = prior_min, value
    elif predicate in {">", ">="}:
        left, right = value, prior_max
    elif predicate == "BETWEEN":
        left, right = value, value_upper
    else:
        epsilon = (prior_max - prior_min) * 0.005
        left, right = value - epsilon, value + epsilon

    left = max(prior_min, left)
    right = min(prior_max, right)
    if right <= left:
        return None
    return left, right


def _align_query_boundaries(root: HoleNode, left: float, right: float) -> List[HoleNode]:
    pending = [(left, right)]
    while pending:
        next_pending: List[Tuple[float, float]] = []
        changed = False
        for seg_left, seg_right in pending:
            leaf = _find_covering_leaf(root, seg_left, seg_right)
            if leaf is None:
                continue
            split_points = [point for point in (seg_left, seg_right) if leaf.left < point < leaf.right]
            if split_points:
                _split_node_at_points(leaf, split_points)
                changed = True
            else:
                next_pending.append((seg_left, seg_right))
        if not changed:
            break
        pending = next_pending if next_pending else [(left, right)]

    leaves: List[HoleNode] = []
    _collect_leaves(root, leaves)
    return [leaf for leaf in leaves if leaf.left >= left - 1e-12 and leaf.right <= right + 1e-12]


def _scale_partition(leaves: Sequence[HoleNode], target_in_mass: float, in_query: Sequence[HoleNode], lr: float) -> None:
    in_nodes = list(in_query)
    out_nodes = [leaf for leaf in leaves if leaf not in in_nodes]
    current_in = sum(node.mass for node in in_nodes)
    current_out = sum(node.mass for node in out_nodes)

    target_in_mass = max(1e-6, min(1.0 - 1e-6, target_in_mass))
    blended_in = current_in + lr * (target_in_mass - current_in)
    blended_out = max(1.0 - blended_in, 1e-6)

    if in_nodes and current_in > 1e-12:
        scale_in = blended_in / current_in
        for node in in_nodes:
            node.mass *= scale_in
    if out_nodes and current_out > 1e-12:
        scale_out = blended_out / current_out
        for node in out_nodes:
            node.mass *= scale_out

    total = sum(node.mass for node in leaves)
    if total > 1e-12:
        for node in leaves:
            node.mass /= total


def _tree_to_quantiles(root: HoleNode, num_buckets: int) -> List[float]:
    leaves: List[HoleNode] = []
    _collect_leaves(root, leaves)
    leaves.sort(key=lambda node: node.left)

    if not leaves:
        return []

    cdf_x = [leaves[0].left]
    cdf_p = [0.0]
    cumulative = 0.0
    for leaf in leaves:
        cumulative += leaf.mass
        cdf_x.append(leaf.right)
        cdf_p.append(cumulative)
    cdf_p[-1] = 1.0
    target_levels = [index / num_buckets for index in range(1, num_buckets)]
    return [float(np.interp(level, cdf_p, cdf_x)) for level in target_levels]


def correct_stholes_flat(
    prior_min: float,
    prior_max: float,
    prior_quantiles: List[float],
    observations: List[dict],
    num_buckets: int = 10,
    lr: float = 0.5,
) -> List[float]:
    if prior_max <= prior_min:
        return prior_quantiles

    boundaries = [prior_min] + sorted(prior_quantiles) + [prior_max]
    buckets = []
    p_mass = 1.0 / (len(boundaries) - 1)
    for index in range(len(boundaries) - 1):
        buckets.append([boundaries[index], boundaries[index + 1], p_mass])

    for obs in observations:
        query = _query_interval(prior_min, prior_max, obs)
        if query is None:
            continue
        q_l, q_r = query
        act_sel = obs["actual_sel"]

        new_buckets = []
        for b_l, b_r, b_p in buckets:
            pts = sorted([b_l, b_r, max(b_l, min(b_r, q_l)), max(b_l, min(b_r, q_r))])
            unique_pts = []
            for point in pts:
                if not unique_pts or abs(point - unique_pts[-1]) > 1e-9:
                    unique_pts.append(point)
            if len(unique_pts) > 2:
                for left, right in zip(unique_pts[:-1], unique_pts[1:]):
                    seg_p = b_p * (right - left) / (b_r - b_l)
                    new_buckets.append([left, right, seg_p])
            else:
                new_buckets.append([b_l, b_r, b_p])
        buckets = new_buckets

        total_p_in_q = 0.0
        indices_in_q = []
        for index, (b_l, b_r, b_p) in enumerate(buckets):
            if b_l >= q_l - 1e-9 and b_r <= q_r + 1e-9:
                total_p_in_q += b_p
                indices_in_q.append(index)

        error = act_sel - total_p_in_q
        if indices_in_q:
            for index in indices_in_q:
                buckets[index][2] += lr * error * (buckets[index][2] / max(total_p_in_q, 1e-12))

        total = sum(bucket[2] for bucket in buckets)
        if total > 0:
            for bucket in buckets:
                bucket[2] /= total

        while len(buckets) > num_buckets:
            min_diff = 1e18
            merge_idx = -1
            for index in range(len(buckets) - 1):
                d1 = buckets[index][2] / max(buckets[index][1] - buckets[index][0], 1e-12)
                d2 = buckets[index + 1][2] / max(buckets[index + 1][1] - buckets[index + 1][0], 1e-12)
                diff = abs(d1 - d2)
                if diff < min_diff:
                    min_diff = diff
                    merge_idx = index
            first = buckets.pop(merge_idx)
            second = buckets.pop(merge_idx)
            buckets.insert(merge_idx, [first[0], second[1], first[2] + second[2]])

    buckets.sort(key=lambda item: item[0])
    cdf_x = [buckets[0][0]]
    cdf_p = [0.0]
    cumulative = 0.0
    for left, right, mass in buckets:
        cdf_x.append(right)
        cumulative += mass
        cdf_p.append(cumulative)
    cdf_p[-1] = 1.0

    target_levels = [index / num_buckets for index in range(1, num_buckets)]
    return [float(np.interp(level, cdf_p, cdf_x)) for level in target_levels]


def correct_stholes_tree(
    prior_min: float,
    prior_max: float,
    prior_quantiles: List[float],
    observations: List[dict],
    num_buckets: int = 10,
    lr: float = 0.5,
) -> List[float]:
    """
    Hierarchical hole-tree variant inspired by the original multidimensional
    STHoles data structure. In this repository we still evaluate single-column
    histograms, so the hyper-rectangle structure degenerates to nested 1D
    intervals, but updates are applied on a drilled hole tree rather than on a
    flat split/merge bucket list.
    """
    if prior_max <= prior_min:
        return prior_quantiles

    root = _initialize_hole_tree(prior_min, prior_max, prior_quantiles)

    for obs in observations:
        query = _query_interval(prior_min, prior_max, obs)
        if query is None:
            continue
        query_left, query_right = query
        in_nodes = _align_query_boundaries(root, query_left, query_right)
        if not in_nodes:
            continue
        all_leaves: List[HoleNode] = []
        _collect_leaves(root, all_leaves)
        _scale_partition(all_leaves, target_in_mass=float(obs["actual_sel"]), in_query=in_nodes, lr=lr)

    return _tree_to_quantiles(root, num_buckets=num_buckets)


def correct_stholes(
    prior_min: float,
    prior_max: float,
    prior_quantiles: List[float],
    observations: List[dict],
    num_buckets: int = 10,
    lr: float = 0.5,
) -> List[float]:
    return correct_stholes_flat(
        prior_min,
        prior_max,
        prior_quantiles,
        observations,
        num_buckets=num_buckets,
        lr=lr,
    )


def correct_qm(
    prior_min: float,
    prior_max: float,
    prior_quantiles: List[float],
    observations: List[dict],
    lr: float = 0.2,
) -> List[float]:
    qs = list(prior_quantiles)
    bucket_count = len(qs) + 1

    for obs in observations:
        predicate = obs["predicate_type"]
        value = obs["value"]
        act_sel = obs["actual_sel"]

        cdf_x = [prior_min] + qs + [prior_max]
        cdf_p = [index / bucket_count for index in range(bucket_count + 1)]
        est_sel = evaluate_piecewise_cdf(cdf_x, cdf_p, value)
        if predicate in {">", ">="}:
            est_sel = 1.0 - est_sel

        error = act_sel - est_sel
        for index in range(len(qs)):
            dist = 1.0 - abs(qs[index] - value)
            if dist > 0.7:
                shift = lr * error * dist
                qs[index] = max(prior_min, min(prior_max, qs[index] + shift))
        qs.sort()
    return qs
