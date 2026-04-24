"""
Drift Gate: 自动选择统计校正策略
================================

根据观测窗口特征自动选择最优校正策略：
- Prior: 无漂移，不校正
- Teacher (Analytical Baseline): 轻度漂移，无训练方法
- OASIS: 中重度漂移，学习型方法

用法：
    from drift_gate import DriftGate, CorrectionStrategy

    gate = DriftGate(low_threshold=0.05, high_threshold=0.15)
    strategy = gate.select_strategy(sample)

    if strategy == CorrectionStrategy.PRIOR:
        corrected = sample.prior.quantile_values
    elif strategy == CorrectionStrategy.TEACHER:
        corrected = correct_quantiles(sample)
    elif strategy == CorrectionStrategy.OASIS:
        corrected = oasis_model.predict(sample)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from histogram_types import KllFeedbackSample, Observation


class CorrectionStrategy(Enum):
    """校正策略枚举"""
    PRIOR = "prior"          # 不校正
    TEACHER = "teacher"      # 解析基线（无训练）
    OASIS_A = "oasis_a"      # 高漂移专用模型 (q=10,20)
    OASIS_B = "oasis_b"      # 低漂移专用模型 (q=1,3,5)
    OASIS_C = "oasis_c"      # 全范围模型 (q=1,3,5,10,15,20)


@dataclass
class DriftMetrics:
    """漂移强度指标"""
    mean_selectivity_error: float  # 选择率误差均值
    max_selectivity_error: float   # 选择率误差最大值
    observation_coverage: float    # 观测值覆盖范围 [0,1]
    observation_count: int         # 观测数量
    drift_score: float             # 综合漂移评分 [0,1]


class DriftGate:
    """
    漂移门控：根据观测窗口特征自动选择校正策略

    参数：
        low_threshold: 低漂移阈值（默认0.05），低于此值使用Prior
        high_threshold: 高漂移阈值（默认0.15），高于此值使用OASIS
        min_observations: 最少观测数（默认3），少于此值使用Prior
        coverage_weight: 覆盖范围权重（默认0.3）
        error_weight: 误差权重（默认0.7）
    """

    def __init__(
        self,
        low_threshold: float = 0.05,
        high_threshold: float = 0.15,
        min_observations: int = 3,
        coverage_weight: float = 0.3,
        error_weight: float = 0.7,
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.min_observations = min_observations
        self.coverage_weight = coverage_weight
        self.error_weight = error_weight

    def compute_drift_metrics(self, sample: KllFeedbackSample) -> DriftMetrics:
        """计算漂移强度指标"""
        obs = sample.observations

        if len(obs) < self.min_observations:
            return DriftMetrics(
                mean_selectivity_error=0.0,
                max_selectivity_error=0.0,
                observation_coverage=0.0,
                observation_count=len(obs),
                drift_score=0.0,
            )

        # 1. 选择率误差
        errors = [
            abs(o.estimated_selectivity - o.actual_selectivity)
            for o in obs
            if o.estimated_selectivity is not None and o.actual_selectivity is not None
        ]

        if not errors:
            mean_error = 0.0
            max_error = 0.0
        else:
            mean_error = sum(errors) / len(errors)
            max_error = max(errors)

        # 2. 观测覆盖范围
        values = [self._normalize_value(o, sample) for o in obs]
        values = [v for v in values if v is not None]

        if len(values) < 2:
            coverage = 0.0
        else:
            coverage = max(values) - min(values)

        # 3. 综合漂移评分
        # 高误差 + 低覆盖 = 高漂移（数据集中变化）
        # 高误差 + 高覆盖 = 高漂移（全局变化）
        drift_score = self.error_weight * mean_error + self.coverage_weight * (1 - coverage)
        drift_score = min(max(drift_score, 0.0), 1.0)

        return DriftMetrics(
            mean_selectivity_error=mean_error,
            max_selectivity_error=max_error,
            observation_coverage=coverage,
            observation_count=len(obs),
            drift_score=drift_score,
        )

    def select_strategy(
        self,
        sample: KllFeedbackSample,
        available_models: Optional[List[CorrectionStrategy]] = None,
    ) -> CorrectionStrategy:
        """
        选择最优校正策略

        参数：
            sample: 反馈样本
            available_models: 可用的OASIS模型列表（默认仅OASIS_C）

        返回：
            推荐的校正策略
        """
        if available_models is None:
            available_models = [CorrectionStrategy.OASIS_C]

        metrics = self.compute_drift_metrics(sample)

        # 规则1：观测数不足，使用Prior
        if metrics.observation_count < self.min_observations:
            return CorrectionStrategy.PRIOR

        # 规则2：漂移极低，使用Prior
        if metrics.drift_score < self.low_threshold:
            return CorrectionStrategy.PRIOR

        # 规则3：轻度漂移，使用Teacher（安全回退）
        if metrics.drift_score < self.high_threshold:
            return CorrectionStrategy.TEACHER

        # 规则4：中重度漂移，选择最合适的OASIS模型
        return self._select_oasis_variant(metrics, available_models)

    def _select_oasis_variant(
        self,
        metrics: DriftMetrics,
        available_models: List[CorrectionStrategy],
    ) -> CorrectionStrategy:
        """
        根据漂移特征选择OASIS模型变体

        启发式规则：
        - 低漂移 (score < 0.25): OASIS_B > OASIS_C > OASIS_A
        - 中漂移 (0.25 ≤ score < 0.40): OASIS_C > OASIS_B > OASIS_A
        - 高漂移 (score ≥ 0.40): OASIS_A > OASIS_C > OASIS_B
        """
        score = metrics.drift_score

        if score < 0.25:
            # 低漂移偏好
            preference = [
                CorrectionStrategy.OASIS_B,
                CorrectionStrategy.OASIS_C,
                CorrectionStrategy.OASIS_A,
            ]
        elif score < 0.40:
            # 中漂移偏好
            preference = [
                CorrectionStrategy.OASIS_C,
                CorrectionStrategy.OASIS_B,
                CorrectionStrategy.OASIS_A,
            ]
        else:
            # 高漂移偏好
            preference = [
                CorrectionStrategy.OASIS_A,
                CorrectionStrategy.OASIS_C,
                CorrectionStrategy.OASIS_B,
            ]

        # 返回第一个可用的模型
        for strategy in preference:
            if strategy in available_models:
                return strategy

        # 回退到Teacher
        return CorrectionStrategy.TEACHER

    def _normalize_value(self, obs: Observation, sample: KllFeedbackSample) -> Optional[float]:
        """将观测值归一化到 [0,1]"""
        if obs.filter_value is None:
            return None

        value_range = sample.prior.max_value - sample.prior.min_value
        if value_range <= 0:
            return 0.5

        normalized = (obs.filter_value - sample.prior.min_value) / value_range
        return max(0.0, min(1.0, normalized))

    def explain_decision(self, sample: KllFeedbackSample) -> str:
        """解释策略选择的原因（用于调试）"""
        metrics = self.compute_drift_metrics(sample)
        strategy = self.select_strategy(sample)

        lines = [
            f"Drift Gate Decision: {strategy.value}",
            f"",
            f"Metrics:",
            f"  - Observation count: {metrics.observation_count}",
            f"  - Mean selectivity error: {metrics.mean_selectivity_error:.4f}",
            f"  - Max selectivity error: {metrics.max_selectivity_error:.4f}",
            f"  - Observation coverage: {metrics.observation_coverage:.4f}",
            f"  - Drift score: {metrics.drift_score:.4f}",
            f"",
            f"Thresholds:",
            f"  - Low threshold: {self.low_threshold}",
            f"  - High threshold: {self.high_threshold}",
            f"  - Min observations: {self.min_observations}",
            f"",
            f"Decision logic:",
        ]

        if metrics.observation_count < self.min_observations:
            lines.append(f"  → Insufficient observations ({metrics.observation_count} < {self.min_observations})")
        elif metrics.drift_score < self.low_threshold:
            lines.append(f"  → Negligible drift ({metrics.drift_score:.4f} < {self.low_threshold})")
        elif metrics.drift_score < self.high_threshold:
            lines.append(f"  → Light drift ({self.low_threshold} ≤ {metrics.drift_score:.4f} < {self.high_threshold})")
        else:
            lines.append(f"  → Moderate-to-heavy drift ({metrics.drift_score:.4f} ≥ {self.high_threshold})")

        return "\n".join(lines)


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    from json_histogram_parser import load_feedback_sample
    from cdf_teacher import correct_quantiles

    # 加载样本
    sample = load_feedback_sample("test_data/sample_001.json")

    # 创建 Drift Gate
    gate = DriftGate(
        low_threshold=0.05,
        high_threshold=0.15,
        min_observations=3,
    )

    # 选择策略
    strategy = gate.select_strategy(sample)
    print(f"Selected strategy: {strategy.value}")

    # 解释决策
    print("\n" + gate.explain_decision(sample))

    # 应用校正
    if strategy == CorrectionStrategy.PRIOR:
        corrected = list(sample.prior.quantile_values)
        print("\nUsing stale prior (no correction)")

    elif strategy == CorrectionStrategy.TEACHER:
        corrected = correct_quantiles(sample)
        print("\nUsing Analytical Baseline (training-free)")

    elif strategy in [CorrectionStrategy.OASIS_A, CorrectionStrategy.OASIS_B, CorrectionStrategy.OASIS_C]:
        # 这里需要加载对应的模型
        print(f"\nUsing {strategy.value} model (learned correction)")
        # corrected = oasis_model.predict(sample)

    print(f"\nCorrected quantiles: {corrected[:3]}...")


if __name__ == "__main__":
    example_usage()
