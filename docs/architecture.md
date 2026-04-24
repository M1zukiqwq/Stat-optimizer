# 系统架构

## 整体流程

```
数据漂移 (INSERT/DELETE/UPDATE)
        ↓
统计信息过期 (stale KLL quantiles)
        ↓
查询优化器选择率估算偏差
        ↓
收集查询反馈 (estimated_sel vs actual_sel)
        ↓
Drift Gate 判断漂移程度
        ↓
选择校正策略:
  - Prior (无漂移): 不校正
  - Teacher (轻漂移): 保序回归，无需训练
  - OASIS (中重度漂移): 多头注意力 MLP
        ↓
输出修正后的分位数值 → 优化器使用新统计
```

## 三层校正策略

| 层级 | 方法 | 训练 | 适用场景 |
|------|------|------|----------|
| 1 | Prior (不校正) | 无 | 漂移可忽略 |
| 2 | Teacher (保序回归) | 无 | 轻度漂移 |
| 3 | OASIS (多头注意力 MLP) | 需要 | 中重度漂移 |

## 模块依赖关系

```
histogram_types.py          # 底层数据结构
    ↑
kll_codec.py                # KLL 编解码
histogram_math.py           # 数学工具 (CDF、保序回归)
    ↑
json_histogram_parser.py    # JSON → KllFeedbackSample
    ↑
cdf_teacher.py              # Teacher 解析基线
tensorizer.py               # 样本 → 特征张量
    ↑
mlp_histogram_model_v2.py   # OASIS v2 模型
train_mlp_model_v2.py       # 训练脚本
    ↑
drift_gate.py               # 策略路由
```

## 数据流

输入 JSON 格式:
```json
{
  "prior_kll": {
    "min": 0.0, "max": 1.0,
    "null_fraction": 0.05,
    "quantile_levels": [0.1, 0.2, ..., 0.9],
    "quantile_values": [0.08, 0.19, ..., 0.91]
  },
  "observations": [
    {
      "predicate_type": "<",
      "value": 0.35,
      "estimated_sel": 0.32,
      "actual_sel": 0.48,
      "timestamp": "2026-01-01T01:00:00Z"
    }
  ],
  "corrected_kll": {
    "quantile_values": [0.12, 0.25, ..., 0.93]
  }
}
```

- `prior_kll`: 过期的 KLL 草图统计
- `observations`: 查询反馈列表（estimated_sel 由 CBO 用过期统计算出，actual_sel 是真实值）
- `corrected_kll`: Ground truth 修正值（仅训练时有）
