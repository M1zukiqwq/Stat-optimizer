# Baseline Implementation Details

针对审稿意见Q4 ("Baseline的实现细节缺失")，本文档详细说明所有baseline的实现细节，确保可复现性。

## 1. STHoles Baseline

### 算法来源
Bruno et al., "STHoles: A Workload-aware Multidimensional Histogram", SIGMOD 2001.

### 核心思想
通过查询反馈动态细化直方图桶边界，将查询范围作为"洞"(holes)插入到直方图中。

### 实现细节 (`baselines.py:correct_stholes`)

```python
def correct_stholes(prior_min, prior_max, prior_quantiles, observations, 
                    num_buckets=10, lr=0.5):
```

**参数设置：**
- `num_buckets=10`: 与OASIS相同的桶预算
- `lr=0.5`: 误差分配学习率，控制对反馈的信任程度

**算法步骤：**

1. **初始化** (Line 24-29)
   - 从prior直方图构建初始桶
   - 每个桶等概率 (p_mass = 1.0 / num_buckets)

2. **处理每个观测** (Line 31-109)
   
   a. **确定查询范围** [q_l, q_r] (Line 37-50)
   - `<` / `<=`: [prior_min, value]
   - `>` / `>=`: [value, prior_max]  
   - `BETWEEN`: [value_lower, value_upper]
   - `=`: [value-eps, value+eps]

   b. **分裂桶** (Line 52-68)
   - 确保桶边界与查询边界对齐
   - 对跨越查询边界的桶进行分裂

   c. **调整计数** (Line 70-90)
   - 计算查询范围内桶的当前概率总和: `total_p_in_q`
   - 计算误差: `error = actual_sel - total_p_in_q`
   - 按当前概率比例分配误差到查询范围内的桶
   - 归一化概率

   d. **合并回预算** (Line 92-108)
   - 当桶数超过预算时，合并密度最接近的相邻桶
   - 合并策略: 选择概率密度差最小的相邻桶对

3. **提取修正分位数** (Line 111-122)
   - 从最终桶分布构建CDF
   - 通过线性插值提取目标分位点

**复杂度分析：**
- 时间: O(K × B²)，K为观测数，B为桶预算
- 空间: O(B)

---

## 2. ISOMER Baseline

### 算法来源
Markl et al., "Consistent Selectivity Estimation via Maximum Entropy", VLDB 2007.

### 核心思想
更接近原始 ISOMER 的做法不是在固定细桶上做平滑回拉，而是：
- 先根据当前活动反馈谓词构造**精确对齐的区间划分**；
- 以 stale prior 为基准分布；
- 对每个反馈约束做**相对熵投影**（I-projection）；
- 当顺序漂移导致约束集不再一致时，优先丢弃更旧的约束。

### 实现细节 (`modern_baselines.py:correct_isomer`)

```python
def correct_isomer(prior_min, prior_max, prior_quantiles, observations,
                   num_buckets=10, max_iter=200, tol=1e-4):
```

**参数设置：**
- `num_buckets=10`: 最终输出桶数
- `max_iter=200`: 活动约束集的循环投影最大轮数
- `tol=1e-4`: 约束残差收敛阈值

**算法步骤：**

1. **把观测变成区间约束**
   - `<` / `<=` 映射到 `[prior_min, value]`
   - `>` / `>=` 映射到 `[value, prior_max]`
   - `BETWEEN` 映射到 `[lower, upper]`
   - `=` 映射到一个很小的局部区间

2. **构造精确划分**
   - 取 `prior` 桶边界与所有活动查询边界的并集；
   - 得到一个 query-aligned cell partition；
   - 每个 cell 的初始概率由 prior 直方图在该 cell 上的质量决定。

3. **对活动约束做循环相对熵投影**
   - 对一个区间约束，最优单步投影会同时缩放区间内和区间外质量；
   - 这比“只调整命中桶再归一化”的平滑器更接近 ISOMER 的最大熵思想；
   - 若多轮迭代后仍无法让所有约束满足 `tol`，则删除最老的活动约束并重试。

4. **提取分位数**
   - 对最终 cell 概率做前缀和；
   - 在目标分位点水平上按 cell 内线性插值，输出修正后的 quantiles。

**和旧版实现的关键区别：**
- 不再使用固定 `50+` 细桶；
- 不再使用 `alpha` 把分布强行往 prior 拉回；
- 不再把 ISOMER 实现成“强正则的一维平滑修正器”；
- 改为更接近论文描述的“query-aligned + max-entropy projection + invalid-QFR dropping”。

---

## 3. QuickSel-H Baseline

### 算法来源
Park et al., "QuickSel: Quick Selectivity Learning with Mixture Models", SIGMOD 2020.

### 核心思想
使用混合高斯模型拟合CDF，从查询反馈学习模型参数，然后采样得到修正直方图。

### 实现细节 (`modern_baselines.py:correct_quicksel_h`)

**当前审计结论（2026-03-12）：**
- 这不是对 QuickSel 的 faithful reimplementation，而是一个 "QuickSel-inspired" 启发式直方图修正器；
- 它只拟合单点 CDF 约束，`BETWEEN` 与 `=` 反馈基本被丢弃；
- 混合权重固定为均匀分布，只优化均值和方差；
- 使用 logistic CDF 近似正态 CDF，并通过简单梯度下降拟合；
- 因此它显著弱于论文中的原始 QuickSel 设定，实验里表现差并不一定意味着代码有 bug，更可能是 baseline 本身过弱。


```python
def correct_quicksel_h(prior_min, prior_max, prior_quantiles, observations,
                       num_buckets=10, n_components=5, max_iter=50):
```

**参数设置：**
- `n_components=5`: 高斯混合分量数
- `max_iter=50`: EM算法最大迭代

**算法步骤：**

1. **初始化混合模型** (Line 54-66)
   - 从prior直方图确定初始分量中心
   - 标准差: `vr / (2 * n_components)`
   - 等权重: `1.0 / n_components`

2. **收集CDF约束** (Line 68-91)
   - 从观测中提取 (value, target_cdf) 点对
   - 对于 `BETWEEN` 和 `=` 谓词跳过（不提供单点CDF信息）

3. **梯度下降优化** (Line 114-142)
   
   使用sigmoid近似正态CDF（计算效率）：
   ```python
   def mixture_cdf(x, means, stds, weights):
       result = 0.0
       for m, s, w in zip(means, stds, weights):
           z = (x - m) / max(s, 1e-6)
           phi = 1.0 / (1.0 + np.exp(-1.7 * z))  # Sigmoid近似
           result += w * phi
       return result
   ```
   
   损失函数：约束点的MSE
   ```python
   error = pred_cdf - target_cdf
   loss += error ** 2
   ```
   
   梯度计算：
   ```python
   dphi = 1.7 * phi * (1.0 - phi)  # Sigmoid导数
   grad_means[k] += 2 * error * weights[k] * (-dphi / s)
   grad_stds[k] += 2 * error * weights[k] * (-dphi * z / s)
   ```

4. **提取分位数** (Line 145-156)
   - 对目标分位点水平使用二分搜索求逆CDF
   - 搜索范围: [prior_min, prior_max]
   - 迭代次数: 50次保证精度

**与原始QuickSel的区别：**
- 原始QuickSel直接输出选择率估计
- QuickSel-H从拟合的混合模型采样得到修正直方图

---

## 4. Quantile-Move (QM) Baseline

### 核心思想
直接移动分位点位置以减小选择率估计误差。

### 实现细节 (`baselines.py:correct_qm`)

```python
def correct_qm(prior_min, prior_max, prior_quantiles, observations, lr=0.2):
```

**参数设置：**
- `lr=0.2`: 分位点移动步长

**算法步骤：**

1. **处理每个观测** (Line 138-159)
   - 使用当前分位点计算估计选择率
   - 计算误差: `error = actual - estimated`
   
2. **调整分位点** (Line 150-158)
   ```python
   for i in range(len(qs)):
       dist = 1.0 - abs(qs[i] - v)  # 距离当前值的接近程度
       if dist > 0.7:  # 只调整接近查询值的分位点
           shift = lr * error * dist
           qs[i] += shift
   qs.sort()  # 保持分位点有序
   ```

**特点：**
- 简单直观，但容易过拟合
- 只调整与查询值接近的分位点

---

## 5. 实现验证与测试

### 单元测试
```python
def test_baselines():
    # 构造已知答案的测试用例
    prior_q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    observations = [
        {"value": 0.5, "predicate_type": "<", "actual_sel": 0.6},
        {"value": 0.3, "predicate_type": ">", "actual_sel": 0.7},
    ]
    
    # 验证各baseline输出合理
    for method in [correct_stholes, correct_isomer, correct_quicksel_h]:
        result = method(0.0, 1.0, prior_q, observations)
        assert len(result) == 9
        assert all(0.0 <= q <= 1.0 for q in result)
        assert result == sorted(result)  # 单调性
```

### 超参数调优
所有baseline的超参数通过网格搜索在验证集上优化：
- STHoles: lr ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
- ISOMER: alpha ∈ {0.1, 0.2, 0.3, 0.5}
- QuickSel-H: n_components ∈ {3, 5, 7, 10}

### 可复现性保证
- 所有随机种子固定 (seed=42)
- NumPy版本: 1.24+
- Python版本: 3.11

---

## 6. 与论文实验的对应关系

| 论文Table | 指标 | Baseline实现 |
|-----------|------|-------------|
| Table 2 | Q-Error @ q=10 | `evaluate_q_error(correct_stholes(...), ...)` |
| Table 2 | Quantile MAE | `mean_absolute_error(corrected_quantiles, true_quantiles)` |
| Table 2 | Selectivity MAE | 在随机谓词上评估平均绝对误差 |

所有baseline共享相同的数据来源：
- 相同的prior直方图 H₀
- 相同的query-feedback 记录源
- 相同的ground truth H*

**但需要注意：** 当前 `run_synthetic_paper_suite.py` 中，OASIS 通过
`tensorize_sample(..., max_observations=16)` 只使用最近 `K=16` 条观测，而经典基线
(`STHoles` / `QuickSel-H` / `ISOMER`) 仍消费样本里的完整 observation 列表（通常 8--24 条）。
因此，当前已发表结果并不是严格的 "所有方法都在 K=16 窗口下" 的公平对比；如果要做严格窗口对齐，
下一步应该在 `method_boundaries()` 中先截断 observation 再全量重跑主实验。
