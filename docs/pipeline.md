# ML Pipeline 详解

## 核心数据结构

### KllPrior (`histogram_types.py`)

归一化的 KLL 草图摘要，值域固定在 [0, 1]。包含 `min_value`(=0)、`max_value`(=1)、`null_fraction`、`quantile_levels` 和 `quantile_values`。

### FeedbackObservation (`histogram_types.py`)

一条查询反馈记录：
- `predicate_type`: `<`, `<=`, `>`, `>=`, `=`, `BETWEEN`
- `value`, `value_upper`: 谓词值（归一化到 [0,1]）
- `estimated_selectivity`: CBO 用过期统计估算的选择率
- `actual_selectivity`: 查询执行后的真实选择率
- `timestamp`: 查询时间

### KllFeedbackSample (`histogram_types.py`)

一个完整的训练/推理样本 = `KllPrior` + `List[FeedbackObservation]` + 可选的 `corrected_quantile_values` (ground truth)。

---

## Teacher: 解析基线 (`cdf_teacher.py`)

无需训练的修正方法：

1. 从先验分位数构建分段 CDF
2. 将每条观测转为 CDF 约束点：
   - `<` / `<=` → (value, actual_sel, weight)
   - `>` / `>=` → (value, 1 - actual_sel, weight)
   - `BETWEEN` → 调整两个端点的 CDF 值
   - `=` → 在值附近开小窗口
3. 合并先验点（权重 = beta / num_prior_points）和观测点
4. 加权保序回归（Isotonic Regression）
5. 逆 CDF 采样回分位数值
6. 覆盖率软阈值：观测覆盖不足时退回先验

关键参数：
- `beta` (0.7): 先验总权重
- `decay_lambda` (1/(14天)): 时间衰减速率
- `coverage_soft_threshold` (0.2): 覆盖率阈值

---

## OASIS v2 模型 (`mlp_histogram_model_v2.py`)

### 架构

```
输入: [prior_quantiles(9) | meta(3) | observation_slots(16×12) | mask(16)]
                                      ↓
                          Multi-Head Attention (3 heads)
                          每个 head: scores = obs_slots @ W + b
                          softmax → 加权池化 → pooled(D_obs)
                          concat 3 heads → pooled_multi(36)
                                      ↓
                          Prior Encoder: 9 → 32 (MLP + ReLU)
                                      ↓
                          Context Fusion: concat(prior_enc, meta, pooled_multi) → 71
                                      ↓
                          Main MLP: 71 → 128 → 128 → 64 → 64 → 9
                                      ↓
                          残差预测: corrected = prior + delta
```

### 特征编码 (`tensorizer.py`)

每条观测编码为 12 维向量：
- 谓词类型 one-hot (6 维): `<`, `<=`, `>`, `>=`, `=`, `BETWEEN`
- `value_norm` (1): 归一化谓词值
- `value_upper_norm` (1): BETWEEN 上界（否则 0）
- `estimated_selectivity` (1): 从先验 CDF 重算（非直接使用输入值，避免标签泄漏）
- `actual_selectivity` (1): 真实选择率
- `has_upper` (1): 是否有上界
- `span_norm` (1): BETWEEN 区间宽度

全局特征：
- `prior_norm` (9): 先验分位值（归一化到 [0,1]）
- `meta` (3): null_fraction, obs_count_ratio, bucket_count_ratio
- `observation_slots` (16×12): 最近 16 条观测，不足补零
- `mask` (16): 有效观测标记

### 训练 (`train_mlp_model_v2.py`)

- 目标: Teacher 输出的修正分位值（蒸馏标签）
- 优化器: Adam (beta1=0.9, beta2=0.999)
- 损失: MSE
- 正则化: L2 (alpha=1e-4) + 梯度裁剪 (max_norm=1.0)
- 默认超参: lr=3e-4, epochs=150, batch_size=32

### 序列化

模型保存为 JSON（所有权重序列化为 list），可跨平台加载。

---

## Drift Gate (`drift_gate.py`)

自动策略选择：

```
观测数 < 3             → Prior
drift_score < 0.05     → Prior
drift_score < 0.15     → Teacher
drift_score >= 0.15    → OASIS (A/B/C 变体)
```

`drift_score = 0.7 × mean_sel_error + 0.3 × (1 - coverage)`

---

## 基线方法

| 方法 | 文件 | 原理 |
|------|------|------|
| STHoles | `baselines.py` | 自适应多维度桶分裂/合并 |
| STGrid | `stgrid.py` | 等宽网格 + 频率梯度更新 |
| QM | `baselines.py` | 基于距离的量化点偏移 |
| QuickSel-H | `modern_baselines.py` | 混合高斯模型 EM 拟合 CDF |
| ISOMER | `modern_baselines.py` | 最大熵 + 迭代比例拟合 (IPF) |
