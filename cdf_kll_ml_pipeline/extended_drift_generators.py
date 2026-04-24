#!/usr/bin/env python3
"""
Extended Drift Generators for OASIS
====================================

针对审稿意见Q3: 仿真数据多样性不足的问题，设计多种真实场景的数据漂移模式。

包含以下数据生成方式：
1. Batch Loading Drift - 批量加载模式
2. Seasonal/Periodic Drift - 周期性数据变化
3. Skew Evolution - 数据倾斜度演变
4. Schema Evolution - 值域范围变化
5. Outlier Injection - 异常值注入
6. Correlated Multi-column - 多列关联漂移
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class DriftPattern(Enum):
    """不同的数据漂移模式"""
    COMPOUND = "compound"           # 原论文的复合漂移
    BATCH_LOAD = "batch_load"       # 批量加载
    SEASONAL = "seasonal"           # 周期性变化
    SKEW_EVOLUTION = "skew_evol"    # 倾斜度演变
    RANGE_SHIFT = "range_shift"     # 值域偏移
    OUTLIER_BURST = "outlier"       # 异常值爆发
    MULTI_MODAL = "multimodal"      # 多模态分布


@dataclass
class DriftConfig:
    """漂移配置"""
    pattern: DriftPattern
    intensity: float  # 0.0 - 1.0
    params: dict


class ExtendedMemoryTable:
    """支持多种漂移模式的内存表"""
    
    def __init__(self, data: List[float], null_count: int, 
                 min_val: float = 0.0, max_val: float = 1.0):
        self.data = data
        self.null_count = null_count
        self.min_val = min_val
        self.max_val = max_val
        self.drift_history = []  # 记录漂移历史
        
    def total_rows(self) -> int:
        return len(self.data) + self.null_count
    
    def get_null_fraction(self) -> float:
        total = self.total_rows()
        return self.null_count / total if total > 0 else 0.0
    
    def query_conditional_sel(self, predicate: str, val: float, val_upper: Optional[float] = None) -> float:
        """Return selectivity among non-null rows only (conditional on row being non-null)."""
        non_null_count = len(self.data)
        if non_null_count == 0:
            return 0.0

        if predicate in {"<", "<="}:
            matches = sum(1 for x in self.data if x <= val)
        elif predicate in {">", ">="}:
            matches = sum(1 for x in self.data if x >= val)
        elif predicate == "BETWEEN":
            if val_upper is None:
                matches = 0
            else:
                matches = sum(1 for x in self.data if val <= x <= val_upper)
        else:  # "="
            epsilon = (self.max_val - self.min_val) * 0.005
            matches = sum(1 for x in self.data if abs(x - val) <= epsilon)

        return matches / non_null_count

    def get_quantiles(self, levels: List[float]) -> List[float]:
        """计算分位数"""
        if not self.data:
            return [self.min_val] * len(levels)
        sorted_data = sorted(self.data)
        n = len(sorted_data)
        out = []
        for p in levels:
            idx_float = p * (n - 1)
            idx_low = int(idx_float)
            idx_high = min(idx_low + 1, n - 1)
            frac = idx_float - idx_low
            interpolated = sorted_data[idx_low] * (1.0 - frac) + sorted_data[idx_high] * frac
            out.append(interpolated)
        return out
    
    # ========================================================================
    # 1. 批量加载漂移 (Batch Loading Drift)
    # ========================================================================
    
    def apply_batch_load_drift(self, rng: random.Random, q_mods: int,
                                batch_size_range: Tuple[int, int] = (100, 1000),
                                target_region: Optional[float] = None):
        """
        模拟ETL批量加载场景：
        - 每隔q_mods轮进行一次大批量插入
        - 数据集中向特定区域倾斜（如最近一天的数据）
        - 很少有删除或更新操作
        
        典型场景：
        - 日志表每日批量追加
        - 交易表按月加载历史数据
        """
        for _ in range(q_mods):
            # 大批量插入
            center = target_region if target_region else rng.uniform(0.7, 0.95)  # 偏向高值区域
            batch_size = rng.randint(*batch_size_range)
            
            # 批量数据有较陡的分布（如时间戳集中在近期）
            for _ in range(batch_size):
                # 使用Beta分布产生偏斜数据
                alpha, beta = 2.0, 0.5  # 偏向1.0
                value = np.random.beta(alpha, beta)
                # 缩放到目标区域附近
                value = center + (value - 0.5) * 0.3
                value = max(0.0, min(1.0, value))
                self.data.append(value)
            
            # 极少量随机删除（模拟数据归档）
            if len(self.data) > batch_size * 2 and rng.random() < 0.1:
                to_delete = min(rng.randint(10, 50), len(self.data) // 10)
                for _ in range(to_delete):
                    # 删除最早期的数据（低值区域）
                    if self.data:
                        min_idx = self.data.index(min(self.data))
                        self.data.pop(min_idx)
            
            self.drift_history.append(('batch_insert', batch_size, center))
    
    # ========================================================================
    # 2. 周期性漂移 (Seasonal/Periodic Drift)
    # ========================================================================
    
    def apply_seasonal_drift(self, rng: random.Random, q_mods: int,
                             period: int = 4,
                             amplitude: float = 0.2):
        """
        模拟周期性业务变化：
        - 数据分布随时间周期摆动（如工作日vs周末、白天vs夜间）
        - 中心位置周期性移动
        
        典型场景：
        - 电商订单在工作日和周末模式不同
        - 金融交易量在开盘和收盘时段不同
        """
        for i in range(q_mods):
            # 计算周期相位 (-1 到 1)
            phase = np.sin(2 * np.pi * i / period)
            center = 0.5 + phase * amplitude
            
            # 插入数据跟随周期变化
            batch = rng.randint(20, 80)
            for _ in range(batch):
                value = rng.normalvariate(center, 0.1)
                value = max(0.0, min(1.0, value))
                self.data.append(value)
            
            # 删除与当前周期相反的区域的数据
            opposite_center = 0.5 - phase * amplitude
            if len(self.data) > 1000:
                to_delete = rng.randint(10, 30)
                opposite_indices = [
                    idx for idx, val in enumerate(self.data)
                    if abs(val - opposite_center) < 0.15
                ]
                for idx in sorted(opposite_indices[:to_delete], reverse=True):
                    if idx < len(self.data):
                        self.data.pop(idx)
            
            self.drift_history.append(('seasonal', batch, center, phase))
    
    # ========================================================================
    # 3. 倾斜度演变 (Skew Evolution)
    # ========================================================================
    
    def apply_skew_evolution(self, rng: random.Random, q_mods: int,
                             initial_skew: float = 0.5,
                             target_skew: float = 0.9):
        """
        模拟数据倾斜度逐渐演变：
        - 从均匀分布逐渐变为高度偏斜
        - 热点区域逐渐集中
        
        典型场景：
        - 新服务上线初期数据均匀，后期头部用户集中
        - 社交媒体中少数热门内容吸引大部分互动
        """
        for i in range(q_mods):
            # 逐步增加倾斜度
            progress = i / max(1, q_mods - 1)
            current_skew = initial_skew + (target_skew - initial_skew) * progress
            
            # 使用幂律分布模拟倾斜
            batch = rng.randint(30, 100)
            for _ in range(batch):
                # 幂律分布: 大部分值在0附近，少数在1附近
                u = rng.random()
                value = 1.0 - (1.0 - u) ** (1.0 / (1.0 - current_skew + 0.1))
                # 偶尔有反转（模拟长尾）
                if rng.random() < 0.1:
                    value = 1.0 - value
                self.data.append(value)
            
            # 删除非热点区域的数据
            if len(self.data) > 500:
                to_delete = rng.randint(20, 60)
                # 删除中间区域，保留两端（热点和长尾）
                mid_indices = [
                    idx for idx, val in enumerate(self.data)
                    if 0.3 < val < 0.7
                ]
                for idx in sorted(mid_indices[:to_delete], reverse=True):
                    if idx < len(self.data):
                        self.data.pop(idx)
            
            self.drift_history.append(('skew_evol', batch, current_skew))
    
    # ========================================================================
    # 4. 值域范围变化 (Range Shift / Schema Evolution)
    # ========================================================================
    
    def apply_range_shift(self, rng: random.Random, q_mods: int,
                         shift_direction: str = 'expand',
                         shift_magnitude: float = 0.1):
        """
        模拟值域范围变化：
        - 新业务扩展导致值域扩大
        - 数据类型范围调整（如INT改为BIGINT）
        
        典型场景：
        - 用户ID池从百万级扩展到千万级
        - 价格字段从整数分变为带小数
        """
        for i in range(q_mods):
            # 动态调整min_val/max_val
            expansion = shift_magnitude * (i + 1) / q_mods
            
            if shift_direction == 'expand':
                # 向高值区域扩展
                new_max = min(1.0, self.max_val + expansion)
                # 插入新范围的数据
                batch = rng.randint(40, 120)
                for _ in range(batch):
                    # 在扩展区域生成数据
                    value = rng.uniform(self.max_val, new_max)
                    self.data.append(value)
                self.max_val = new_max
                
            elif shift_direction == 'shift':
                # 整体平移（同时调整min和max）
                shift = expansion * 0.5
                if self.max_val + shift <= 1.0:
                    # 平移现有数据
                    self.data = [min(1.0, x + shift * 0.1) for x in self.data]
                    self.min_val = min(1.0, self.min_val + shift * 0.1)
                    self.max_val = min(1.0, self.max_val + shift)
                    # 在新区域插入数据
                    batch = rng.randint(30, 80)
                    for _ in range(batch):
                        self.data.append(rng.uniform(self.max_val - shift, self.max_val))
            
            # 随机更新少量旧数据到新范围
            if self.data:
                update_count = min(rng.randint(5, 15), len(self.data) // 20)
                indices = rng.sample(range(len(self.data)), update_count)
                for idx in indices:
                    self.data[idx] = rng.uniform(
                        max(0.0, self.max_val - 0.2),
                        self.max_val
                    )
            
            self.drift_history.append(('range_shift', shift_direction, expansion))
    
    # ========================================================================
    # 5. 异常值爆发 (Outlier Burst)
    # ========================================================================
    
    def apply_outlier_burst(self, rng: random.Random, q_mods: int,
                           outlier_ratio: float = 0.15,
                           burst_frequency: int = 3):
        """
        模拟异常值周期性爆发：
        - 正常业务数据夹杂异常数据
        - 异常可能来自传感器故障、数据质量问题等
        
        典型场景：
        - IoT传感器间歇性故障产生异常读数
        - 促销活动期间的价格异常
        """
        for i in range(q_mods):
            is_burst = (i % burst_frequency == 0)
            
            batch = rng.randint(30, 90)
            for _ in range(batch):
                if is_burst and rng.random() < outlier_ratio:
                    # 生成异常值（远离正常分布）
                    if rng.random() < 0.5:
                        value = rng.uniform(0.0, 0.1)  # 低端异常
                    else:
                        value = rng.uniform(0.9, 1.0)  # 高端异常
                else:
                    # 正常数据（集中在中间区域）
                    value = rng.normalvariate(0.5, 0.15)
                    value = max(0.0, min(1.0, value))
                self.data.append(value)
            
            # 模拟"清理"操作：删除已识别的异常
            if not is_burst and len(self.data) > 200:
                to_clean = rng.randint(10, 30)
                # 删除极端值
                extreme_indices = [
                    idx for idx, val in enumerate(self.data)
                    if val < 0.05 or val > 0.95
                ]
                for idx in sorted(extreme_indices[:to_clean], reverse=True):
                    if idx < len(self.data):
                        self.data.pop(idx)
            
            self.drift_history.append(('outlier', batch, is_burst))
    
    # ========================================================================
    # 6. 多模态分布 (Multi-Modal Distribution)
    # ========================================================================
    
    def apply_multimodal_drift(self, rng: random.Random, q_mods: int,
                               n_modes: int = 3,
                               mode_separation: float = 0.25):
        """
        模拟多模态业务数据：
        - 不同用户群体/产品类别的数据呈现多个峰值
        - 各模态的权重随时间变化
        
        典型场景：
        - 不同会员等级用户的消费金额分布
        - 多品类商品的库存周转天数
        """
        # 初始化模态中心
        mode_centers = [0.2 + i * mode_separation for i in range(n_modes)]
        
        for i in range(q_mods):
            # 模态权重随时间变化
            t = i / max(1, q_mods - 1)
            weights = [
                0.3 + 0.4 * np.sin(2 * np.pi * t + j * 2 * np.pi / n_modes)
                for j in range(n_modes)
            ]
            weights = [max(0.1, w) for w in weights]
            total = sum(weights)
            weights = [w / total for w in weights]
            
            batch = rng.randint(40, 100)
            for _ in range(batch):
                # 根据权重选择模态
                mode = rng.choices(range(n_modes), weights=weights)[0]
                center = mode_centers[mode]
                value = rng.normalvariate(center, 0.05)
                value = max(0.0, min(1.0, value))
                self.data.append(value)
            
            # 某个模态的数据可能被批量删除（如某类用户流失）
            if rng.random() < 0.2 and len(self.data) > 300:
                target_mode = rng.randint(0, n_modes - 1)
                target_center = mode_centers[target_mode]
                to_delete = rng.randint(20, 50)
                mode_indices = [
                    idx for idx, val in enumerate(self.data)
                    if abs(val - target_center) < 0.1
                ]
                for idx in sorted(mode_indices[:to_delete], reverse=True):
                    if idx < len(self.data):
                        self.data.pop(idx)
            
            self.drift_history.append(('multimodal', batch, weights))
    
    # ========================================================================
    # 统一接口
    # ========================================================================
    
    def apply_drift_by_pattern(self, pattern: DriftPattern, rng: random.Random, 
                               q_mods: int, **kwargs):
        """根据模式应用相应的漂移"""
        if pattern == DriftPattern.COMPOUND:
            # 使用原有的复合漂移
            from simulate_memory_kll_dataset import MemoryTable
            # 复用原有逻辑...
            for _ in range(q_mods):
                self._apply_compound_round(rng, kwargs.get('persistent_center'))
        elif pattern == DriftPattern.BATCH_LOAD:
            self.apply_batch_load_drift(rng, q_mods, **kwargs)
        elif pattern == DriftPattern.SEASONAL:
            self.apply_seasonal_drift(rng, q_mods, **kwargs)
        elif pattern == DriftPattern.SKEW_EVOLUTION:
            self.apply_skew_evolution(rng, q_mods, **kwargs)
        elif pattern == DriftPattern.RANGE_SHIFT:
            self.apply_range_shift(rng, q_mods, **kwargs)
        elif pattern == DriftPattern.OUTLIER_BURST:
            self.apply_outlier_burst(rng, q_mods, **kwargs)
        elif pattern == DriftPattern.MULTI_MODAL:
            self.apply_multimodal_drift(rng, q_mods, **kwargs)
    
    def _apply_compound_round(self, rng: random.Random, persistent_center: Optional[float] = None):
        """原有复合漂移的一轮操作"""
        center = persistent_center if persistent_center else rng.uniform(0.1, 0.9)
        batch = rng.randint(10, 100)
        for _ in range(batch):
            self.data.append(max(0.0, min(1.0, rng.normalvariate(center, 0.05))))
        
        if self.data:
            batch = min(len(self.data), rng.randint(10, 100))
            for _ in range(batch):
                idx = rng.randint(0, len(self.data) - 1)
                del self.data[idx]
        
        if self.data:
            batch = min(len(self.data), rng.randint(10, 100))
            for _ in range(batch):
                idx = rng.randint(0, len(self.data) - 1)
                self.data[idx] = max(0.0, min(1.0, self.data[idx] + rng.uniform(-0.1, 0.1)))
        
        self.null_count = max(0, self.null_count + rng.randint(-50, 50))


def generate_mixed_dataset(output_dir: str, n_cases: int = 6000):
    """
    生成混合漂移模式的训练数据集
    
    包含：
    - 20% Compound drift (原有模式)
    - 15% Batch loading
    - 15% Seasonal
    - 15% Skew evolution
    - 10% Range shift
    - 10% Outlier burst
    - 15% Multi-modal
    """
    import json
    from pathlib import Path
    from datetime import datetime, timezone
    
    patterns = [
        (DriftPattern.COMPOUND, 0.20),
        (DriftPattern.BATCH_LOAD, 0.15),
        (DriftPattern.SEASONAL, 0.15),
        (DriftPattern.SKEW_EVOLUTION, 0.15),
        (DriftPattern.RANGE_SHIFT, 0.10),
        (DriftPattern.OUTLIER_BURST, 0.10),
        (DriftPattern.MULTI_MODAL, 0.15),
    ]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rng = random.Random(42)
    
    for case_idx in range(n_cases):
        # 随机选择漂移模式
        pattern = rng.choices([p[0] for p in patterns], weights=[p[1] for p in patterns])[0]
        
        # 初始化数据
        initial_size = rng.randint(5000, 15000)
        data = []
        for _ in range(initial_size):
            data.append(rng.normalvariate(0.5, 0.2))
        null_count = int(initial_size * rng.uniform(0.01, 0.1))
        
        table = ExtendedMemoryTable(data, null_count)
        
        # 应用漂移
        q_mods = rng.choice([1, 3, 5, 10, 15, 20])
        table.apply_drift_by_pattern(pattern, rng, q_mods)
        
        # 生成案例数据（与原有格式兼容）
        case_data = {
            "case_id": case_idx,
            "drift_pattern": pattern.value,
            "drift_intensity": q_mods,
            "data_stats": {
                "total_rows": table.total_rows(),
                "null_fraction": table.get_null_fraction(),
                "value_range": [table.min_val, table.max_val],
            },
            "drift_history": [
                {"step": i, "event": h[0], "details": h[1:]} 
                for i, h in enumerate(table.drift_history)
            ]
        }
        
        # 保存
        output_file = output_path / f"mixed_case_{case_idx:04d}_{pattern.value}.json"
        output_file.write_text(json.dumps(case_data, indent=2), encoding="utf-8")
    
    print(f"Generated {n_cases} mixed-pattern cases in {output_path}")


if __name__ == "__main__":
    # 测试各种漂移模式
    import argparse
    
    parser = argparse.ArgumentParser(description="Test extended drift generators")
    parser.add_argument("--pattern", choices=[p.value for p in DriftPattern], 
                       default="batch_load", help="Drift pattern to test")
    parser.add_argument("--q", type=int, default=10, help="Drift intensity")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization")
    
    args = parser.parse_args()
    
    # 运行测试
    rng = random.Random(42)
    pattern = DriftPattern(args.pattern)
    
    # 初始化
    data = [rng.normalvariate(0.5, 0.2) for _ in range(10000)]
    table = ExtendedMemoryTable(data, 100)
    
    # 记录初始分布
    initial_quantiles = table.get_quantiles([0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"Initial quantiles (10/25/50/75/90): {[f'{q:.3f}' for q in initial_quantiles]}")
    
    # 应用漂移
    table.apply_drift_by_pattern(pattern, rng, args.q)
    
    # 记录最终分布
    final_quantiles = table.get_quantiles([0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"Final quantiles (10/25/50/75/90): {[f'{q:.3f}' for q in final_quantiles]}")
    
    print(f"\nDrift history ({len(table.drift_history)} events):")
    for i, event in enumerate(table.drift_history[:5]):
        print(f"  {i+1}. {event}")
    if len(table.drift_history) > 5:
        print(f"  ... and {len(table.drift_history) - 5} more")
