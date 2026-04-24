# cdf_kll_ml_pipeline 设计文档

> 最后更新：2026-03-04
> 本文整合：README.md + kll_correction_design.md + presto_iceberg_kll_model_integration_interfaces.md

---

## 一、模块目标

基于查询执行反馈（通过双层架构自动收集 per-conjunct 选择率）修正 KLL 直方图：

```
输入 JSON (prior_kll + observations)
    → 特征张量化
    → OASIS MLP 模型推理
    → 输出 corrected_kll.quantile_values
```

---

## 二、快速开始

```bash
cd presto-cdf-simulation/cdf_kll_ml_pipeline

# 1. 生成训练数据（内存表模拟，推荐）
python3 simulate_memory_kll_dataset.py \
  --output-dir training_data_sim --k 512 --q 5

# 2. 训练模型
python3 train_histogram_model.py \
  --train-glob "training_data_sim/*.json" \
  --output-model artifacts/kll_mlp_model.json

# 3. 推理
python3 predict_histogram.py \
  --input samples/collected_feedback_sample.json \
  --model artifacts/kll_mlp_model.json \
  --output-json artifacts/predicted_kll.json
```

---

## 三、输入 JSON 协议

```json
{
  "prior_kll": {
    "min": 0.0, "max": 1.0, "null_fraction": 0.03,
    "quantile_levels": [0.1, 0.2, ..., 0.9],
    "quantile_values": [0.08, 0.19, ..., 0.91]
  },
  "observations": [
    {
      "predicate_type": "<",
      "value": 0.35,
      "estimated_sel": 0.30,
      "actual_sel": 0.42,
      "timestamp": "2026-01-01T10:00:00Z"
    }
  ],
  "corrected_kll": {
    "quantile_values": [0.09, 0.21, ..., 0.93]
  }
}
```

**字段说明：**
- `prior_kll`：ANALYZE 时的快照统计（过期的先验）
- `observations`：通过双层反馈收集架构自动收集的查询反馈，支持 `<`, `<=`, `>`, `>=`, `=`, `BETWEEN`
- `corrected_kll`：漂移后的真实分布（训练时的 ground truth，由模拟器自动生成）

**兼容字段：**
- `bucket_boundaries`（自动转换为 quantile_values）
- `prior_histogram`（老协议回退）

---

## 四、特征张量设计

```
feature_tensor = [
    prior_norm  (B-1 维),      ← 先验分位点（归一化到 [0,1]）
    meta        (3 维),         ← null_fraction / obs_count_ratio / bucket_count
    observations (K × 12 维),  ← 每条观测：谓词 one-hot(6) + 数值特征(6)
    mask        (K 维)          ← 1.0=有效, 0.0=填充
]
```

**观测 12 维特征（对应每条 observation）：**
| 特征 | 维度 | 说明 |
|---|---|---|
| 谓词 one-hot | 6 | `<`, `<=`, `>`, `>=`, `=`, `BETWEEN` |
| value（归一化） | 1 | |
| value_upper（归一化） | 1 | BETWEEN 上界，否则 0 |
| estimated_sel | 1 | 旧统计估算的选择率 |
| actual_sel | 1 | 真实选择率 |
| has_upper | 1 | 是否有 value_upper |
| span | 1 | BETWEEN 区间宽度 |

**关键约定：**
- 观测按**时间正序**排列（index 0 = 最旧，最新在末位），兼容 LSTM/Transformer 因果顺序
- 不足 K 条时**末尾填零**，mask 置 0
- 默认 B=10, K=16，总维度 = 9 + 3 + 192 + 16 = **220 维**

---

## 五、Null-Fraction 语义约定

`estimated_sel` 与 `actual_sel` 均为**整体行选择率**：

```
sel = P(条件成立 | 非空) × P(非空)
```

- `estimated_sel`：使用 ANALYZE 快照时的**先验** `null_fraction`（有意使用过期统计，模拟 CBO 行为）
- `actual_sel`：条件选择率 × **当前**（可能漂移后的）`null_fraction`

两个数据生成器均遵守此约定，保证训练/推理语义一致。

---

## 六、三种校正策略

### 6.1 Prior（不校正）
直接使用 KLL Sketch 中的旧分位点，作为 baseline。

### 6.2 Teacher（推理-only，不参与OASIS训练）

> ⚠️ **Teacher 是独立的推理策略，不是 OASIS 训练的标签源**（主要路径）。

算法流程：
1. 将 observations 转为 CDF 约束点（带时间衰减权重）
2. 加入先验分位点为软约束（权重 β）
3. 加权保序回归拟合单调合法 CDF
4. 反插值得到修正分位点

**覆盖率检测（鲁棒性机制）**：
- 每条非 `=` 观测在 value 两侧贡献 ±5% 宽度窗口
- 计算所有窗口并集对 `[0,1]` 区间的覆盖比例
- `coverage < 0.2` 时对修正结果做线性软回退至先验
- 正确区分"两端聚集采样"（coverage≈10%）与"均匀覆盖"（coverage≈1.0）

**退化条件**：观测 < 3 条时直接返回先验。

### 6.3 OASIS 模型（注意力池化 MLP）

**架构**（CPU 友好，纯 numpy，约 17K 参数）：
1. 将 K 个观测槽（各 12 维）用可学习注意力打分 → masked softmax → 加权汇聚为 12 维向量
2. 将 [prior(9) | meta(3) | pooled_obs(12)] 拼接为 24 维 context
3. 两层 MLP：24 → 128（ReLU）→ 64（ReLU）→ 9（输出分位点）

**训练标签来源（优先级递减）**：
1. ✅ **`corrected_kll.quantile_values`**：`simulate_memory_kll_dataset.py` 生成时写入的漂移后真实分布 — **主要路径，推荐**
2. ⚠️ Teacher 伪标签：仅当 JSON 没有 `corrected_kll` 时 — 降级方案
3. ❌ 两者均无：样本被丢弃

**训练**：Adam optimizer，He 初始化，L2 正则 α=1e-4

**推荐训练配置**：`--train-q-values 10 20 --k-train 1000`（各 1000 条，共 2000 条）

**后处理**：
- 裁剪至 [0,1]
- 保序投影（Pool Adjacent Violators）
- 反归一化至原始列值范围

---

## 七、数据生成器对比

| 生成器 | corrected_kll | 特点 | 用途 |
|---|---|---|---|
| `simulate_memory_kll_dataset.py` | ✅ 真实分布 | 模拟 ANALYZE→DML漂移→查询完整生命周期 | **OASIS 训练（推荐）** |
| `generate_synthetic_json_dataset.py` | ❌ 无 | 直接随机采样两套边界，速度快 | 快速调试/Teacher 伪标签训练 |

**内存表模拟器漂移操作**（每轮 `q` 次）：
- **insert**：向持久化的持续热点 (persistent hotspot) 注入 10-100 行制造定向偏斜
- **delete**：随机删除 10-100 行
- **update**：随机扰动 10-100 行的值 ±0.1
- **null_change**：调整 null 行数量 ±50

---

## 八、推理输出格式

```json
{
  "corrected_quantile_values": [...],
  "corrected_kll": {
    "quantile_levels": [...],
    "quantile_values": [...],
    "sketch_bytes_base64": "..."
  },
  "corrected_histogram": {
    "bucket_boundaries": [0.0, ..., 1.0]
  }
}
```

> 注：`sketch_bytes_base64` 使用 simulation JSON payload 的 base64 编码，**非 Presto/Iceberg 原生二进制 KLL bytes**。

---

## 九、PostgreSQL Extension 接入设计

### 9.1 调用链与插入点

```
PostgreSQL Planner
    → get_relation_stats()
    → [Hook] oasis_get_relation_stats_hook()
        → 读取 pg_statistic
        → 调用 OASIS 修正服务
        → 返回修正后的统计
```

插入点选择依据：histogram_bounds 已解析（quantile 可用）、可按列粒度修正、CBO 和下游透明。

### 9.2 PostgreSQL Extension 架构

**方案 A：C Extension（性能最优）**

```c
// oasis_extension.c
#include "postgres.h"
#include "catalog/pg_statistic.h"
#include "utils/syscache.h"

PG_MODULE_MAGIC;

// Hook 函数
static get_relation_stats_hook_type prev_get_relation_stats_hook = NULL;

static bool
oasis_get_relation_stats_hook(PlannerInfo *root, RangeTblEntry *rte,
                               AttrNumber attnum, VariableStatData *vardata)
{
    // 1. 调用原始 hook（如果存在）
    if (prev_get_relation_stats_hook &&
        prev_get_relation_stats_hook(root, rte, attnum, vardata))
        return true;

    // 2. 读取 pg_statistic
    HeapTuple statsTuple = SearchSysCache3(STATRELATTINH,
                                           ObjectIdGetDatum(rte->relid),
                                           Int16GetDatum(attnum),
                                           BoolGetDatum(rte->inh));

    // 3. 提取 histogram_bounds
    Datum histogram_datum = SysCacheGetAttr(STATRELATTINH, statsTuple,
                                            Anum_pg_statistic_stakind1, &isnull);

    // 4. 调用 OASIS 修正服务（HTTP/ONNX）
    float8 *corrected_bounds = oasis_correct_histogram(histogram_datum, ...);

    // 5. 更新 vardata
    vardata->statsTuple = build_corrected_stats_tuple(corrected_bounds);

    return true;
}

void _PG_init(void)
{
    // 注册 hook
    prev_get_relation_stats_hook = get_relation_stats_hook;
    get_relation_stats_hook = oasis_get_relation_stats_hook;

    // 定义 GUC 参数
    DefineCustomBoolVariable("oasis.correction_enabled", ...);
    DefineCustomStringVariable("oasis.model_uri", ...);
    DefineCustomIntVariable("oasis.model_timeout_ms", ...);
}
```

**方案 B：Python Extension（开发快速）**

使用 PL/Python 实现：

```python
# oasis_plpython.py
import psycopg2
import requests

def oasis_correct_column_stats(table_name, column_name):
    """
    读取 pg_statistic，调用 OASIS 模型，返回修正后的直方图
    """
    # 1. 读取当前统计
    conn = psycopg2.connect("dbname=postgres")
    cur = conn.cursor()
    cur.execute("""
        SELECT stakind1, stavalues1, stanumbers1
        FROM pg_statistic s
        JOIN pg_class c ON s.starelid = c.oid
        JOIN pg_attribute a ON s.staattnum = a.attnum AND a.attrelid = c.oid
        WHERE c.relname = %s AND a.attname = %s
    """, (table_name, column_name))

    stakind, histogram_bounds, null_frac = cur.fetchone()

    # 2. 读取 observations（从自定义表）
    cur.execute("""
        SELECT predicate_type, value, estimated_sel, actual_sel, timestamp
        FROM oasis_observations
        WHERE table_name = %s AND column_name = %s
        ORDER BY timestamp DESC LIMIT 16
    """, (table_name, column_name))
    observations = cur.fetchall()

    # 3. 调用模型服务
    response = requests.post('http://localhost:8080/predict', json={
        'prior_kll': {
            'quantile_values': histogram_bounds,
            'null_fraction': null_frac
        },
        'observations': observations
    }, timeout=0.2)

    corrected = response.json()['corrected_quantile_values']

    # 4. 更新 pg_statistic（需要超级用户权限）
    # 或者缓存在内存中，由 hook 读取

    return corrected
```

### 9.3 配置方式

**GUC 参数**（postgresql.conf 或 SET 命令）：

```sql
-- 启用 OASIS 修正
SET oasis.correction_enabled = on;

-- 模型服务地址
SET oasis.model_uri = 'http://localhost:8080/predict';

-- 超时时间（毫秒）
SET oasis.model_timeout_ms = 200;

-- Observations 存储位置
SET oasis.feedback_table = 'oasis_observations';
```

### 9.4 Observations 收集

**方案 A：pg_stat_statements + 触发器**

```sql
-- 创建 observations 表
CREATE TABLE oasis_observations (
    id SERIAL PRIMARY KEY,
    table_name TEXT,
    column_name TEXT,
    predicate_type TEXT,
    value DOUBLE PRECISION,
    estimated_sel DOUBLE PRECISION,
    actual_sel DOUBLE PRECISION,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- 创建索引
CREATE INDEX idx_oasis_obs_lookup
ON oasis_observations(table_name, column_name, timestamp DESC);

-- 使用 pg_stat_statements 收集查询统计
-- 通过 EXPLAIN ANALYZE 提取估算 vs 实际行数
```

**方案 B：Executor Hook**

```c
// 在 ExecutorEnd hook 中收集 per-operator 统计
static void
oasis_executor_end_hook(QueryDesc *queryDesc)
{
    // 遍历 plan tree，提取每个 Filter 节点的统计
    extract_filter_stats(queryDesc->planstate);

    // 写入 oasis_observations 表
    insert_observations(...);

    // 调用原始 hook
    if (prev_executor_end_hook)
        prev_executor_end_hook(queryDesc);
}
```

### 9.5 部署步骤

1. **编译 Extension**
```bash
cd oasis_extension
make
sudo make install
```

2. **加载 Extension**
```sql
CREATE EXTENSION oasis;
```

3. **配置参数**
```sql
ALTER SYSTEM SET oasis.correction_enabled = on;
ALTER SYSTEM SET oasis.model_uri = 'http://localhost:8080/predict';
SELECT pg_reload_conf();
```

4. **启动模型服务**
```bash
cd cdf_kll_ml_pipeline
python3 model_service.py &
```

5. **验证**
```sql
-- 查看当前配置
SHOW oasis.correction_enabled;

-- 运行测试查询
EXPLAIN ANALYZE SELECT * FROM title WHERE production_year > 2000;
```

---

## 十、文件说明

| 文件 | 作用 |
|---|---|
| `json_histogram_parser.py` | KLL JSON 协议解析，兼容 v1 `prior_histogram` 格式 |
| `kll_codec.py` | KLL 量化点 simulation 编解码（base64） |
| `histogram_math.py` | CDF 插值、保序回归等数学工具 |
| `histogram_types.py` | 数据类型定义（KllPrior, FeedbackObservation, KllFeedbackSample） |
| `tensorizer.py` | 特征张量化（时序正序、衰减权重、覆盖率 mask） |
| `cdf_teacher.py` | Teacher 推理策略（加权保序回归 + 覆盖率软回退） |
| `mlp_histogram_model.py` | OASIS 模型：注意力池化 MLP（纯 Python/NumPy） |
| `simulate_memory_kll_dataset.py` | 内存表漂移模拟数据生成器（含 ground truth） |
| `generate_synthetic_json_dataset.py` | 数学随机采样数据生成器（无 ground truth） |
| `json_to_tensor.py` | 导出训练输入 tensor |
| `train_histogram_model.py` | 训练入口 |
| `predict_histogram.py` | 推理入口（含超时软回退） |
