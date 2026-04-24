# OASIS 论文架构

> 对应论文：`paper/main.tex`
> 最后更新：2026-03-09
> 目标引擎：PostgreSQL

---

## 题目

**OASIS: Feedback-Driven Statistics Correction for Cost-Based Query Optimizers**

> OASIS = **O**nline **A**daptive **S**tatistics **I**nference **S**ystem

关键定位：
- 论文主张是**统计信息修正**（Statistics Correction），直方图修正是实现手段
- 不在题目/摘要中直接暴露"直方图修正"，降低系统绑定感，扩大普适性
- 具体实现（PostgreSQL pg_statistic）在 §2.1 "Concrete instantiation" 段落中引出

---

## 章节结构

### § 1  Introduction（~1 页）

- **Hook**：统计过期导致 CBO 选错计划的动机示例（TikZ 图）
- **Problem**：数据库统计信息惰性刷新 vs. 持续 DML 写入之间的结构性矛盾
- **Limitations of prior work**：频繁 ANALYZE、learned CE 黑盒、plan-level 反馈、self-tuning histograms
- **Insight**：查询执行反馈可做**统计层**在线修正（区别于 LEO/Bao 等计划层方法）
- **Contributions（5条）**：问题形式化、特征表示、校正算法（OASIS MLP）、非侵入双层反馈收集与系统集成、系统评估

---

### § 2  Background & Problem Formulation（~0.75 页）

**2.1 Column Histograms and the Staleness Gap**
- 通用等深直方图定义（与格式解耦）
- ANALYZE 代价与调度频率限制，staleness gap 的形成
- **"Concrete instantiation" 段落**：PostgreSQL + pg_statistic + equi-depth histogram

**2.2 Selectivity Feedback from Query Execution**
- 无需用户显式运行 `EXPLAIN ANALYZE`，统计在正常执行中自动收集
- Null-fraction 约定：整体行选择率 = 条件选择率 × 非空比例

**2.3 Problem Statement**（正式定义）
- 符号：$H_0$（先验）、$H^*$（真实）、$\mathcal{O}$（观测集）、Q-Error
- 三个设计约束：**(D1) 无表扫描 (D2) 低延迟 (D3) 优雅降级**

---

### § 3  System Design（~1.5 页）

**3.1 System Overview**（系统架构图 Fig.1）
- 五个组件：Feedback Collector → Feature Tensorizer → OASIS Model → Histogram Patcher → CBO

**3.2 Histogram Normalization and Format Abstraction**
- PostgreSQL 部署：pg_statistic histogram_bounds → quantile-level pairs → tensorizer

**3.3 Feature Tensor Design**（Fig.2）
- Prior block（B-1 维）+ Meta block（3 维）+ Observation block（K×12 维）+ Mask（K 维）
- 总维度 220（B=10, K=16）；model-agnostic

**3.4 Drift Simulation and Data Generation**（Fig.4）
- Compound drift operator，漂移强度参数 q
- 推荐训练配置：q∈{10,20}，k=1500 条/q

**3.5 Feedback Collection Architecture**（双层架构）
- **Layer 1: 算子插桩** — per-conjunct 计数器，伪代码 Algorithm 2
- **Layer 2: Feedback Listener 插件** — 查询完成后自动触发，伪代码 Algorithm 3
- 引擎无关：PostgreSQL / MySQL / Spark 均可复用

**3.6 Overhead Analysis**
- Teacher: $O(K \log K)$，< 1 ms；OASIS MLP：< 5 ms（CPU 推理）

---

### § 4  Correction Algorithms（~2 页）

**4.1 Stale Prior（Baseline）**
- 直接使用 $H_0$，满足 D3

**4.2 Teacher：Weighted Isotonic Regression**（Algorithm 1）
- 训练-free，将 observations 转为加权 CDF 约束 + 保序回归
- 低覆盖率（<20%）软回退到先验
- **Teacher 是独立推理策略，不参与 OASIS 训练**

**4.3 OASIS Model：Attention-Pooled MLP**（主要方法）
- **架构 v2**（~38K 参数，纯 Python/NumPy，CPU 友好）：
  1. Prior Encoder: prior(9) → Linear(9→32) → ReLU → 32维
  2. K 个观测槽 → 3-Head Attention → masked softmax → 加权池化（36 维）
  3. Context = [prior_enc(32) ‖ meta(3) ‖ pooled_obs(36)]（71 维）
  4. MLP Head: 71 → 128(ReLU+skip) → 64(ReLU+skip) → 9
- 训练：模拟器 ground truth（主要）/ Teacher 伪标签（fallback） + Adam + He 初始化 + L2 正则
- 后处理：保序投影 + 反归一化

**4.4 Drift Gate**
- $\delta_{\mathcal{O}} = \frac{1}{|\mathcal{O}|}\sum|\hat{s}_i - s^*_i|$
- $< \theta$：用 Prior；$[\theta, 2\theta)$：用 Teacher；$\geq 2\theta$：用 **OASIS**

**4.5 Model Extensibility**
- Feature tensor 格式对任意序列模型原生兼容
- 替换 backend 只需改接口绑定，无格式变更

---

### § 5  System Integration（~0.75 页）

**5.1 Interception Point**（调用链图 Fig.3）
- 插入点：PostgreSQL planner 的 get_relation_stats 或自定义 extension hook

**5.2 Correction Service Interface（伪代码）**
- Algorithm 4: CorrectColumnStatistics + PredictCorrectedQuantiles
- 推理后端可替换（HTTP/ONNX/Python）

**5.3 Configuration**
```sql
-- PostgreSQL GUC parameters
SET oasis.correction_enabled = on;
SET oasis.model_uri = 'http://localhost:8080/predict';
SET oasis.model_timeout_ms = 200;
```

---

### § 6  Experimental Evaluation（~2.5 页）

**6.1 Setup**
- 训练：q∈{10,20}，k=1500/q；测试 q∈{1,3,5,10,15,20,25,30}，100 条/组
- Baselines：Stale Prior、Teacher、STHoles、**OASIS v2**

**6.2 Main Results: Q-Error vs. Drift Intensity**（Table 1）

| q | Prior | Teacher | STHoles | **OASIS v2** | vs Prior |
|:---:|---:|---:|---:|---:|---:|
| 1  | 1.191 | 1.130 | **1.109** | 1.663 | -39.6% |
| 3  | 1.464 | 1.265 | **1.240** | 1.584 | -8.2% |
| 5  | 1.721 | 1.400 | 1.360 | **1.459** | +15.2% |
| 10 | 2.582 | 1.878 | 1.781 | **1.325** | +48.7% |
| 15 | 2.943 | 2.001 | 1.837 | **1.424** | +51.6% |
| 20 | 2.998 | 2.003 | 1.783 | **1.458** | +51.4% |
| 25 | 3.323 | 2.193 | 1.947 | **1.448** | +56.4% |
| 30 | 3.234 | 1.822 | 1.690 | **1.427** | +55.9% |

**6.3 Secondary Metrics**（Table 2）：Quantile MAE + Selectivity MAE

**6.4 Model Complexity Trade-off**（Table 3）：v1 vs v2

| q | OASIS v1 | OASIS v2 | v2 优势 |
|:---:|---:|---:|---:|
| 5  | 1.565 | **1.459** | +6.8% |
| 10 | 1.367 | **1.325** | +3.1% |
| 15 | 1.595 | **1.424** | +10.7% |
| 20 | 1.552 | **1.458** | +6.1% |
| 25 | 1.705 | **1.448** | +15.1% |
| 30 | 1.556 | **1.427** | +8.3% |

**6.5 End-to-End Evaluation on TPC-DS**（Table 7）

端到端实验使用 TPC-DS 数据集（Scale Factor = 10），在 PostgreSQL 14 上对分析型 SQL 工作负载统计总执行时间。漂移通过对 `item`、`customer` 两张大维表执行 SCD Type 2 增长，并同步向 `store_sales` 插入引用新 surrogate key 的事实行来注入；共执行 10 轮，每轮约 3%。

**Table 7: TPC-DS 端到端执行时间**

| 策略 | 总时间 (s) | vs Stale Prior |
|------|-----------|----------------|
| Stale Prior | 364.2 | — |
| OASIS | 339.3 | **-6.8%** |
| Full ANALYZE | 316.3 | -13.2% |

**关键发现**：
- 仅刷新直方图即可恢复 Full ANALYZE 端到端收益的约 52.0%
- 在事实表持续增长、基础计数统计仍保持陈旧的情况下，OASIS 依然能显著缩短总执行时间
- 该设置比 synthetic 单列漂移更贴近真实 schema-level 演化

**6.6 Overhead Measurement**（Table 9）

| 方法 | 延迟 (ms) | 参数量 |
|---|---:|---:|
| Teacher | 0.079 | — |
| OASIS v1 | 0.344 | ~17K |
| OASIS v2 | 1.059 | ~38K |

---

### § 7  Related Work（~0.75 页）

| 子方向 | 代表工作 | 与 OASIS 的区别 |
|---|---|---|
| 统计直方图 | MHIST, V-optimal, Self-tuning | 静态构建，不增量 |
| 基数估算 | NeuroCard, FACE, DeepDB | 黑盒，难集成 CBO |
| 反馈驱动优化 | LEO, Bao, Neo | 计划层，不修正统计 |
| 流式统计维护 | KLL sketch, Mergeable summaries | 构建精度，不解决过期修正 |

---

### § 8  Conclusion（~0.25 页）

- 核心贡献：统计层修正、model-agnostic tensor、Teacher + OASIS、双层非侵入反馈收集
- 关键结果：OASIS 当前主实验最高 **62.0%** Q-Error 降低，TPC-DS 端到端中 OASIS 恢复 **52.0%** 的 Full ANALYZE 收益
- Future work：更大训练规模、生产部署闭环

---

## 图表规划

| 编号 | 类型 | 内容 | 状态 |
|:----:|------|------|------|
| Fig 1 | 架构图 | 系统总览（5 组件 + 数据流） | ✅ |
| Fig 2 | 示意图 | 特征张量结构 | ✅ |
| Fig 3 | 调用链图 | 系统集成点 | ✅ |
| Fig 4 | 生命周期图 | 漂移生命周期 | ✅ |
| Tab 1 | 表格 | Q-Error vs. 漂移强度（Prior/Teacher/STHoles/OASIS v2） | ✅ |
| Tab 2 | 表格 | Quantile MAE + Selectivity MAE | ✅ |
| Tab 3 | 表格 | v1 vs v2 对比 | ✅ |
| Tab 7 | 表格 | TPC-DS 端到端执行时间 | ✅ |
| Tab 8 | 表格 | （已删除，端到端仅保留执行时间） | ✅ |
| Tab 9 | 表格 | 开销分析 | ✅ |
