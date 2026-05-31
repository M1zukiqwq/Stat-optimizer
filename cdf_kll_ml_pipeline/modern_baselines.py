"""
现代统计修正基线方法
====================

实现两个额外的统计修正基线，用于与OASIS对比：

1. QuickSel-H: 受QuickSel启发的混合模型方法
   - 从query feedback学习混合高斯模型拟合CDF
   - 然后从拟合的CDF中提取修正后的分位点
   - 参考: Park et al., "QuickSel: Quick Selectivity Learning with Mixture Models", SIGMOD 2020

2. ISOMER: 基于最大熵的直方图修正
   - 在满足观测约束的前提下，找到最大熵分布
   - 使用迭代比例拟合(IPF)算法
   - 参考: Markl et al., "Consistent Selectivity Estimation via Maximum Entropy", VLDB 2007
"""
import numpy as np
from typing import List, Dict, Optional, Sequence, Tuple
from histogram_math import evaluate_piecewise_cdf


def correct_quicksel_h(
    prior_min: float,
    prior_max: float,
    prior_quantiles: List[float],
    observations: List[dict],
    num_buckets: int = 10,
    n_components: int = 5,
    max_iter: int = 50,
) -> List[float]:
    """
    QuickSel-H: 受QuickSel启发的混合高斯模型直方图修正。

    QuickSel原始方法使用混合模型直接估计选择率，但不输出修正后的直方图。
    我们将其适配为直方图修正：
    1. 从prior直方图初始化混合高斯模型
    2. 使用EM算法结合query feedback约束进行拟合
    3. 从拟合的CDF中提取修正后的分位点

    参数:
        prior_min: 列最小值
        prior_max: 列最大值
        prior_quantiles: 先验分位点值列表
        observations: 观测列表，每个包含 value, predicate_type, actual_sel 等
        num_buckets: 目标桶数
        n_components: 混合高斯分量数
        max_iter: EM最大迭代次数
    """
    if prior_max <= prior_min or len(observations) < 2:
        return prior_quantiles

    vr = prior_max - prior_min

    # 1. 从prior直方图初始化混合高斯参数
    boundaries = [prior_min] + sorted(prior_quantiles) + [prior_max]
    n_bins = len(boundaries) - 1

    # 初始化: 每个分量对应一个桶的中心
    step = max(1, n_bins // n_components)
    means = []
    for i in range(n_components):
        idx = min(i * step, n_bins - 1)
        means.append((boundaries[idx] + boundaries[idx + 1]) / 2.0)
    means = np.array(means)
    stds = np.full(n_components, vr / (2 * n_components))
    weights = np.full(n_components, 1.0 / n_components)

    # 2. 收集约束点: (value, actual_cdf)
    constraints = []
    for obs in observations:
        v = obs["value"]
        pred = obs["predicate_type"]
        act_sel = obs["actual_sel"]

        if pred in {"<", "<="}:
            cdf_at_v = act_sel
        elif pred in {">", ">="}:
            cdf_at_v = 1.0 - act_sel
        elif pred == "BETWEEN":
            # BETWEEN给出的是区间选择率，转换为CDF约束
            v_upper = obs.get("value_upper", v)
            constraints.append((v, None))  # 标记为区间
            constraints.append((v_upper, None))
            continue
        elif pred == "=":
            continue  # 等值谓词不提供CDF信息
        else:
            continue

        cdf_at_v = max(0.001, min(0.999, cdf_at_v))
        constraints.append((v, cdf_at_v))

    if len(constraints) < 2:
        return prior_quantiles

    # 过滤有效约束
    valid_constraints = [(v, c) for v, c in constraints if c is not None]
    if len(valid_constraints) < 2:
        return prior_quantiles

    # 3. 简化EM: 通过最小化CDF约束误差来调整混合模型参数
    def mixture_cdf(x, means, stds, weights):
        """计算混合高斯CDF"""
        result = 0.0
        for m, s, w in zip(means, stds, weights):
            s = max(s, 1e-6)
            z = (x - m) / s
            # 用sigmoid近似正态CDF (更快)
            phi = 1.0 / (1.0 + np.exp(-1.7 * z))
            result += w * phi
        return result

    # 梯度下降优化
    lr = 0.01 * vr
    for iteration in range(max_iter):
        total_loss = 0.0
        grad_means = np.zeros(n_components)
        grad_stds = np.zeros(n_components)

        for v, target_cdf in valid_constraints:
            pred_cdf = mixture_cdf(v, means, stds, weights)
            error = pred_cdf - target_cdf
            total_loss += error ** 2

            # 计算梯度
            for k in range(n_components):
                s = max(stds[k], 1e-6)
                z = (v - means[k]) / s
                phi = 1.0 / (1.0 + np.exp(-1.7 * z))
                dphi = 1.7 * phi * (1.0 - phi)

                grad_means[k] += 2 * error * weights[k] * (-dphi / s)
                grad_stds[k] += 2 * error * weights[k] * (-dphi * z / s)

        # 更新参数
        means -= lr * grad_means
        stds -= lr * grad_stds
        stds = np.maximum(stds, vr * 0.01)  # 防止std过小

        # 衰减学习率
        if iteration > 0 and iteration % 15 == 0:
            lr *= 0.7

    # 4. 从拟合的混合CDF中提取分位点
    target_levels = [i / num_buckets for i in range(1, num_buckets)]
    corrected = []
    for level in target_levels:
        # 二分搜索找到CDF^{-1}(level)
        lo, hi = prior_min, prior_max
        for _ in range(50):
            mid = (lo + hi) / 2
            if mixture_cdf(mid, means, stds, weights) < level:
                lo = mid
            else:
                hi = mid
        corrected.append((lo + hi) / 2)

    return corrected


def _isomer_interval_from_observation(
    prior_min: float,
    prior_max: float,
    observation: dict,
) -> Optional[Tuple[float, float, float]]:
    predicate = observation["predicate_type"]
    value = float(observation["value"])
    value_upper = float(observation.get("value_upper", value))
    target = float(observation["actual_sel"])
    value_range = max(prior_max - prior_min, 1e-12)
    epsilon = value_range * 0.005

    if predicate in {"<", "<="}:
        left, right = prior_min, value
    elif predicate in {">", ">="}:
        left, right = value, prior_max
    elif predicate == "BETWEEN":
        left, right = sorted((value, value_upper))
    elif predicate == "=":
        left, right = value - epsilon, value + epsilon
    else:
        return None

    left = max(prior_min, left)
    right = min(prior_max, right)
    if right <= left:
        return None

    target = max(1e-6, min(1.0 - 1e-6, target))
    return left, right, target


def _isomer_unique_sorted(values: Sequence[float], eps: float = 1e-12) -> List[float]:
    result: List[float] = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > eps:
            result.append(float(value))
    return result


def _isomer_prior_cell_probs(
    prior_min: float,
    prior_max: float,
    prior_quantiles: Sequence[float],
    cell_boundaries: Sequence[float],
) -> np.ndarray:
    prior_boundaries = [prior_min] + sorted(prior_quantiles) + [prior_max]
    bucket_mass = 1.0 / max(len(prior_boundaries) - 1, 1)

    probs = np.zeros(len(cell_boundaries) - 1, dtype=float)
    bucket_index = 0
    for cell_index, (left, right) in enumerate(zip(cell_boundaries[:-1], cell_boundaries[1:])):
        while bucket_index + 1 < len(prior_boundaries) and prior_boundaries[bucket_index + 1] <= left + 1e-12:
            bucket_index += 1
        safe_index = min(bucket_index, len(prior_boundaries) - 2)
        bucket_left = prior_boundaries[safe_index]
        bucket_right = prior_boundaries[safe_index + 1]
        bucket_width = max(bucket_right - bucket_left, 1e-12)
        probs[cell_index] = bucket_mass * max(right - left, 0.0) / bucket_width

    total = probs.sum()
    if total <= 1e-12:
        return np.full(len(probs), 1.0 / max(len(probs), 1), dtype=float)
    return probs / total


def _isomer_build_partition(
    prior_min: float,
    prior_max: float,
    prior_quantiles: Sequence[float],
    intervals: Sequence[Tuple[float, float, float]],
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    boundaries = [prior_min, prior_max]
    boundaries.extend(prior_quantiles)
    for left, right, _ in intervals:
        boundaries.extend((left, right))
    cell_boundaries = np.array(_isomer_unique_sorted(boundaries), dtype=float)
    prior_probs = _isomer_prior_cell_probs(prior_min, prior_max, prior_quantiles, cell_boundaries)

    masks: List[np.ndarray] = []
    targets = []
    for left, right, target in intervals:
        mask = np.logical_and(cell_boundaries[:-1] >= left - 1e-12, cell_boundaries[1:] <= right + 1e-12)
        if np.any(mask):
            masks.append(mask)
            targets.append(target)
    return cell_boundaries, prior_probs, masks, np.array(targets, dtype=float)


def _isomer_project_single_constraint(probs: np.ndarray, mask: np.ndarray, target: float) -> np.ndarray:
    inside = mask
    outside = ~mask
    current = float(probs[inside].sum())

    if not np.any(inside):
        return probs
    if current <= 1e-12 or current >= 1.0 - 1e-12:
        return probs

    next_probs = probs.copy()
    next_probs[inside] *= target / current
    outside_mass = max(1.0 - current, 1e-12)
    next_probs[outside] *= (1.0 - target) / outside_mass

    total = next_probs.sum()
    if total > 1e-12:
        next_probs /= total
    return next_probs


def _isomer_fit_active_set(
    prior_probs: np.ndarray,
    masks: Sequence[np.ndarray],
    targets: np.ndarray,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, float, bool]:
    probs = prior_probs.copy()
    if not masks:
        return probs, 0.0, True

    best_error = float("inf")
    for _ in range(max_iter):
        for mask, target in zip(masks, targets):
            probs = _isomer_project_single_constraint(probs, mask, float(target))

        residuals = [abs(float(probs[mask].sum()) - float(target)) for mask, target in zip(masks, targets)]
        max_error = max(residuals) if residuals else 0.0
        best_error = min(best_error, max_error)
        if max_error <= tol:
            return probs, max_error, True

    return probs, best_error, False


def _isomer_latest_feasible_suffix(
    prior_min: float,
    prior_max: float,
    prior_quantiles: Sequence[float],
    intervals: Sequence[Tuple[float, float, float]],
    max_iter: int,
    tol: float,
) -> List[Tuple[float, float, float]]:
    active_intervals: List[Tuple[float, float, float]] = []
    latest_feasible: List[Tuple[float, float, float]] = []

    for interval in intervals:
        active_intervals.append(interval)
        while active_intervals:
            _, prior_probs, masks, targets = _isomer_build_partition(
                prior_min,
                prior_max,
                prior_quantiles,
                active_intervals,
            )
            _, _, converged = _isomer_fit_active_set(
                prior_probs,
                masks,
                targets,
                max_iter=max_iter,
                tol=tol,
            )
            if converged:
                latest_feasible = list(active_intervals)
                break
            active_intervals.pop(0)
        if not active_intervals:
            latest_feasible = []

    return latest_feasible


def _isomer_quantiles_from_cells(
    cell_boundaries: np.ndarray,
    probs: np.ndarray,
    num_buckets: int,
) -> List[float]:
    cumulative = np.cumsum(probs)
    cumulative[-1] = 1.0
    target_levels = [index / num_buckets for index in range(1, num_buckets)]
    corrected: List[float] = []

    for level in target_levels:
        cell_index = int(np.searchsorted(cumulative, level, side="left"))
        cell_index = min(cell_index, len(probs) - 1)
        prev_cdf = cumulative[cell_index - 1] if cell_index > 0 else 0.0
        cell_mass = max(cumulative[cell_index] - prev_cdf, 1e-12)
        fraction = (level - prev_cdf) / cell_mass
        left = float(cell_boundaries[cell_index])
        right = float(cell_boundaries[cell_index + 1])
        corrected.append(float(left + fraction * (right - left)))
    return corrected


def correct_isomer(
    prior_min: float,
    prior_max: float,
    prior_quantiles: List[float],
    observations: List[dict],
    num_buckets: int = 10,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> List[float]:
    """
    A closer single-column adaptation of ISOMER.

    The implementation follows the original idea more closely than the previous
    smoothed fine-grid variant: it builds an exact interval partition induced by
    the active feedback predicates, initializes cell masses from the stale prior
    histogram, and performs cyclic I-projections onto each query-feedback range
    constraint. When sequential drift makes the active set inconsistent, older
    constraints are dropped first, mirroring ISOMER's invalid-QFR handling for
    updates.
    """
    if prior_max <= prior_min or not observations:
        return prior_quantiles

    parsed_intervals = []
    for observation in observations:
        interval = _isomer_interval_from_observation(prior_min, prior_max, observation)
        if interval is not None:
            parsed_intervals.append(interval)

    if not parsed_intervals:
        return prior_quantiles

    active_intervals: List[Tuple[float, float, float]] = []
    fitted_probs: Optional[np.ndarray] = None
    fitted_boundaries: Optional[np.ndarray] = None

    for interval in parsed_intervals:
        active_intervals.append(interval)
        while active_intervals:
            boundaries, prior_probs, masks, targets = _isomer_build_partition(
                prior_min,
                prior_max,
                prior_quantiles,
                active_intervals,
            )
            probs, max_error, converged = _isomer_fit_active_set(
                prior_probs,
                masks,
                targets,
                max_iter=max_iter,
                tol=tol,
            )
            if converged:
                fitted_probs = probs
                fitted_boundaries = boundaries
                break
            active_intervals.pop(0)
        if not active_intervals:
            fitted_probs = None
            fitted_boundaries = None

    if fitted_probs is None or fitted_boundaries is None:
        boundaries, prior_probs, _, _ = _isomer_build_partition(
            prior_min,
            prior_max,
            prior_quantiles,
            [],
        )
        fitted_probs = prior_probs
        fitted_boundaries = boundaries

    return _isomer_quantiles_from_cells(fitted_boundaries, fitted_probs, num_buckets=num_buckets)


def correct_soft_isomer(
    prior_min: float,
    prior_max: float,
    prior_quantiles: List[float],
    observations: List[dict],
    num_buckets: int = 10,
    constraint_strength: float = 30.0,
    recency_decay: float = 0.8,
    target_blend: float = 1.0,
    max_iter: int = 500,
    learning_rate: float = 0.05,
    tol: float = 1e-9,
    active_set: bool = False,
    active_set_max_iter: int = 50,
    active_set_tol: float = 1e-4,
    conflict_aware: bool = False,
    conflict_ref_window: int = 8,
    conflict_tau: float = 0.05,
    conflict_floor: float = 0.0,
    conflict_ref_max_iter: int = 50,
    conflict_ref_tol: float = 1e-4,
) -> List[float]:
    """
    Soft feedback-consistency projection.

    Unlike :func:`correct_isomer`, this operator does not force the final
    histogram to exactly satisfy every feedback predicate. It optimizes a soft
    calibration objective over the same interval partition:

        KL(p || prior) + lambda * sum_j w_j (A_j p - y_j)^2

    where ``prior`` is the learned or stale marginal supplied by the caller,
    ``A_j`` is a feedback interval mask, and ``w_j`` optionally emphasizes more
    recent observations. This gives OASIS a Stage-2 calibration alternative that
    can back off from noisy or inconsistent feedback while still penalizing
    residuals directly. When ``active_set`` is enabled, the soft objective is
    applied only to the latest suffix of feedback constraints that hard ISOMER
    can satisfy, preserving ISOMER's protection against stale contradictory
    feedback while keeping the KL-to-learned-prior soft calibration.

    When ``conflict_aware`` is enabled, each feedback constraint is additionally
    reweighted by how consistent it is with the most recent ``conflict_ref_window``
    observations, which are treated as the trusted reference for the current data
    state. A hard reference distribution is fit to that recent suffix, and every
    constraint's residual against the reference distribution becomes a conflict
    score. Constraints that the recent feedback contradicts are down-weighted by
    ``exp(-(conflict / conflict_tau)^2)`` (clipped to a minimum of
    ``conflict_floor``). This is a smooth, deployment-visible generalization of a
    fixed recent-window: consistent old observations keep full weight (preserving
    single-column accuracy) while contradicted old observations are suppressed
    (preserving OOD/trace/planner safety), without ever using fresh labels.
    """
    if prior_max <= prior_min or not observations:
        return prior_quantiles

    parsed_intervals = []
    for observation in observations:
        interval = _isomer_interval_from_observation(prior_min, prior_max, observation)
        if interval is not None:
            parsed_intervals.append(interval)

    if not parsed_intervals:
        return prior_quantiles

    if active_set:
        parsed_intervals = _isomer_latest_feasible_suffix(
            prior_min,
            prior_max,
            prior_quantiles,
            parsed_intervals,
            max_iter=max(1, int(active_set_max_iter)),
            tol=max(float(active_set_tol), 0.0),
        )
        if not parsed_intervals:
            return prior_quantiles

    cell_boundaries, prior_probs, masks, targets = _isomer_build_partition(
        prior_min,
        prior_max,
        prior_quantiles,
        parsed_intervals,
    )
    if not masks:
        return prior_quantiles

    prior = np.maximum(prior_probs.astype(float), 1e-12)
    prior /= max(float(prior.sum()), 1e-12)
    probs = prior.copy()

    mask_matrix = np.vstack([mask.astype(float) for mask in masks])
    targets = np.asarray(targets, dtype=float)
    target_blend = max(0.0, min(1.0, float(target_blend)))
    if target_blend < 1.0:
        prior_targets = mask_matrix @ prior
        targets = target_blend * targets + (1.0 - target_blend) * prior_targets
    targets = np.clip(targets, 1e-6, 1.0 - 1e-6)

    decay = max(1e-6, min(1.0, float(recency_decay)))
    if decay < 1.0 and len(targets) > 1:
        weights = np.array([decay ** (len(targets) - idx - 1) for idx in range(len(targets))], dtype=float)
    else:
        weights = np.ones(len(targets), dtype=float)

    if conflict_aware and len(targets) > 1:
        # Fit a hard reference distribution to the most recent observations,
        # then down-weight every constraint contradicted by that reference.
        ref_window = max(1, int(conflict_ref_window))
        ref_count = min(ref_window, len(masks))
        ref_masks = list(masks[-ref_count:])
        ref_targets = targets[-ref_count:]
        ref_probs, _, _ = _isomer_fit_active_set(
            prior,
            ref_masks,
            ref_targets,
            max_iter=max(1, int(conflict_ref_max_iter)),
            tol=max(float(conflict_ref_tol), 0.0),
        )
        ref_estimates = mask_matrix @ ref_probs
        conflict = np.abs(ref_estimates - targets)
        tau = max(float(conflict_tau), 1e-6)
        conflict_weights = np.exp(-(conflict / tau) ** 2)
        conflict_weights = np.clip(conflict_weights, max(0.0, float(conflict_floor)), 1.0)
        # The reference observations are trusted by construction.
        if ref_count > 0:
            conflict_weights[-ref_count:] = 1.0
        weights = weights * conflict_weights

    weights *= len(weights) / max(float(weights.sum()), 1e-12)

    strength = max(0.0, float(constraint_strength))
    if strength <= 1e-12 or max_iter <= 0:
        return _isomer_quantiles_from_cells(cell_boundaries, prior, num_buckets=num_buckets)

    best_probs = probs.copy()
    best_objective = float("inf")
    previous_objective: Optional[float] = None
    base_step = float(learning_rate) / max(1.0, np.sqrt(strength / 30.0))

    for iteration in range(max_iter):
        estimates = mask_matrix @ probs
        residuals = estimates - targets
        safe_probs = np.maximum(probs, 1e-12)
        kl_term = float(np.sum(safe_probs * np.log(safe_probs / prior)))
        residual_term = float(np.sum(weights * residuals * residuals))
        objective = kl_term + strength * residual_term

        if objective < best_objective:
            best_objective = objective
            best_probs = probs.copy()

        if (
            previous_objective is not None
            and iteration > 30
            and abs(previous_objective - objective) <= tol
        ):
            break
        previous_objective = objective

        gradient = (
            np.log(safe_probs / prior)
            + 1.0
            + 2.0 * strength * (mask_matrix.T @ (weights * residuals))
        )
        step = base_step / np.sqrt(1.0 + iteration / 50.0)
        update = np.clip(-step * gradient, -30.0, 30.0)
        probs = safe_probs * np.exp(update)
        total = float(probs.sum())
        if total <= 1e-12 or not np.isfinite(total):
            probs = prior.copy()
        else:
            probs = np.maximum(probs / total, 1e-12)
            probs /= max(float(probs.sum()), 1e-12)

    return _isomer_quantiles_from_cells(cell_boundaries, best_probs, num_buckets=num_buckets)


def _isomer_band_widths(targets: np.ndarray, band_kappa: float, band_floor: float) -> np.ndarray:
    """Per-constraint feedback confidence half-width.

    The width follows a binomial-shape proxy ``kappa * sqrt(y (1 - y))`` (widest
    near y = 0.5, where an observed selectivity carries the most sampling
    uncertainty) plus a constant floor. ``kappa = 0`` and ``band_floor = 0``
    reproduce exact equality constraints, i.e. hard projection.
    """
    t = np.clip(np.asarray(targets, dtype=float), 1e-6, 1.0 - 1e-6)
    return float(band_floor) + float(band_kappa) * np.sqrt(t * (1.0 - t))


def _isomer_project_banded(probs: np.ndarray, mask: np.ndarray, target: float, band: float) -> np.ndarray:
    """I-projection onto a feedback confidence band [target-band, target+band].

    If the current interval mass already lies inside the band the distribution
    is left untouched (it stays as close to the prior as the band allows);
    otherwise it is multiplicatively projected onto the nearer band edge.
    """
    current = float(probs[mask].sum())
    lo = max(1e-6, target - band)
    hi = min(1.0 - 1e-6, target + band)
    if lo <= current <= hi:
        return probs
    edge = lo if current < lo else hi
    return _isomer_project_single_constraint(probs, mask, edge)


def _isomer_fit_active_set_banded(
    prior_probs: np.ndarray,
    masks: Sequence[np.ndarray],
    targets: np.ndarray,
    bands: np.ndarray,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, float, bool]:
    probs = prior_probs.copy()
    if not masks:
        return probs, 0.0, True

    best_error = float("inf")
    for _ in range(max_iter):
        for mask, target, band in zip(masks, targets, bands):
            probs = _isomer_project_banded(probs, mask, float(target), float(band))

        violations = [
            max(0.0, abs(float(probs[mask].sum()) - float(target)) - float(band))
            for mask, target, band in zip(masks, targets, bands)
        ]
        max_error = max(violations) if violations else 0.0
        best_error = min(best_error, max_error)
        if max_error <= tol:
            return probs, max_error, True

    return probs, best_error, False


def correct_band_isomer(
    prior_min: float,
    prior_max: float,
    prior_quantiles: List[float],
    observations: List[dict],
    num_buckets: int = 10,
    band_kappa: float = 0.04,
    band_floor: float = 0.0,
    max_iter: int = 200,
    tol: float = 1e-4,
    conflict_aware: bool = False,
    conflict_ref_window: int = 8,
    conflict_tau: float = 0.05,
    conflict_drop: float = 0.10,
    conflict_ref_max_iter: int = 50,
    conflict_ref_tol: float = 1e-4,
) -> List[float]:
    """
    Banded (uncertainty-aware) feedback-consistency projection.

    This generalizes :func:`correct_isomer`: instead of forcing each active
    feedback predicate to be matched exactly, it I-projects the supplied prior
    onto the polytope ``y_j - delta_j <= A_j p <= y_j + delta_j``, where
    ``delta_j`` is a deployment-visible confidence band (see
    :func:`_isomer_band_widths`). It keeps ISOMER's incremental active set with
    drop-oldest-on-infeasibility behaviour (the implicit stale-constraint filter
    that protects against abrupt drift), so for ``band_kappa = band_floor = 0``
    it reduces exactly to hard projection. For positive band widths the solution
    is allowed to stay closer to the learned prior whenever the feedback is
    already satisfied to tolerance, which improves future single-column
    generalization while bounding the feedback residual by the band.

    When ``conflict_aware`` is set, constraints whose residual against the most
    recent ``conflict_ref_window`` observations exceeds ``conflict_drop`` are
    discarded before fitting (a smooth-cutoff analogue of ISOMER's age drop that
    keeps old-but-consistent feedback).
    """
    if prior_max <= prior_min or not observations:
        return prior_quantiles

    parsed_intervals = []
    for observation in observations:
        interval = _isomer_interval_from_observation(prior_min, prior_max, observation)
        if interval is not None:
            parsed_intervals.append(interval)

    if not parsed_intervals:
        return prior_quantiles

    if conflict_aware and len(parsed_intervals) > 1:
        cell_boundaries, prior_probs, masks, targets = _isomer_build_partition(
            prior_min, prior_max, prior_quantiles, parsed_intervals
        )
        if masks:
            prior = np.maximum(prior_probs.astype(float), 1e-12)
            prior /= max(float(prior.sum()), 1e-12)
            mask_matrix = np.vstack([m.astype(float) for m in masks])
            targets_arr = np.asarray(targets, dtype=float)
            ref_count = min(max(1, int(conflict_ref_window)), len(masks))
            ref_probs, _, _ = _isomer_fit_active_set(
                prior, list(masks[-ref_count:]), targets_arr[-ref_count:],
                max_iter=max(1, int(conflict_ref_max_iter)),
                tol=max(float(conflict_ref_tol), 0.0),
            )
            conflict = np.abs(mask_matrix @ ref_probs - targets_arr)
            # Map kept-mask conflicts back to parsed intervals: rebuild keep flags.
            kept_intervals = []
            mask_idx = 0
            for interval in parsed_intervals:
                left, right, _ = interval
                m = np.logical_and(cell_boundaries[:-1] >= left - 1e-12, cell_boundaries[1:] <= right + 1e-12)
                if np.any(m):
                    if mask_idx >= len(masks) - ref_count or conflict[mask_idx] <= float(conflict_drop):
                        kept_intervals.append(interval)
                    mask_idx += 1
            if kept_intervals:
                parsed_intervals = kept_intervals

    active_intervals: List[Tuple[float, float, float]] = []
    fitted_probs: Optional[np.ndarray] = None
    fitted_boundaries: Optional[np.ndarray] = None

    for interval in parsed_intervals:
        active_intervals.append(interval)
        while active_intervals:
            boundaries, prior_probs, masks, targets = _isomer_build_partition(
                prior_min, prior_max, prior_quantiles, active_intervals
            )
            bands = _isomer_band_widths(targets, band_kappa, band_floor)
            probs, max_error, converged = _isomer_fit_active_set_banded(
                prior_probs, masks, targets, bands, max_iter=max_iter, tol=tol
            )
            if converged:
                fitted_probs = probs
                fitted_boundaries = boundaries
                break
            active_intervals.pop(0)
        if not active_intervals:
            fitted_probs = None
            fitted_boundaries = None

    if fitted_probs is None or fitted_boundaries is None:
        boundaries, prior_probs, _, _ = _isomer_build_partition(
            prior_min, prior_max, prior_quantiles, []
        )
        fitted_probs = prior_probs
        fitted_boundaries = boundaries

    return _isomer_quantiles_from_cells(fitted_boundaries, fitted_probs, num_buckets=num_buckets)
