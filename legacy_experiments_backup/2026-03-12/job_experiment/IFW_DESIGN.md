# IMDB Filter Workload (IFW) - 验证直方图修正效果

## 设计目标

JOB Benchmark 主要测试 JOIN 性能，而 JOIN 估算不使用直方图。为了验证直方图修正（OASIS）的效果，我们需要一个**以 Filter 为主的查询集**。

## 核心思想

### 1. 针对性设计

**目标列**：我们漂移的列（production_year, info_type_id 等）

**查询类型**：
- 范围查询（`WHERE production_year BETWEEN 1990 AND 2000`）
- 不等式查询（`WHERE production_year > 2000`）
- 组合 Filter（`WHERE production_year > 2000 AND kind_id = 1`）

**为什么有效**：
- 漂移后，稀少值变多，频繁值变少（分布反转）
- 直方图能准确捕捉新分布
- 均匀分布假设会严重错误

### 2. 预期效果

**Stale Prior（过期直方图）**：
- 查询 `WHERE production_year BETWEEN 1920 AND 1950`
- 旧直方图：估算 1000 行（稀少）
- 实际：50000 行（漂移后变多）
- **Q-error = 50**

**Full ANALYZE（新鲜直方图）**：
- 新直方图：估算 48000 行
- 实际：50000 行
- **Q-error = 1.04**

---

## 查询模板

### Template 1: 单列范围查询

测试直方图对范围选择率的估算。

```sql
-- Q1: 查询早期电影（漂移后变多）
SELECT COUNT(*), MIN(title), MAX(title)
FROM title
WHERE production_year BETWEEN 1920 AND 1950;

-- Q2: 查询近期电影（漂移后变少）
SELECT COUNT(*), MIN(title), MAX(title)
FROM title
WHERE production_year BETWEEN 2000 AND 2012;

-- Q3: 查询中期电影（漂移影响小）
SELECT COUNT(*), MIN(title), MAX(title)
FROM title
WHERE production_year BETWEEN 1970 AND 1990;
```

**预期 Q-error**：
- Q1: Stale Prior 严重低估（Q-error > 10）
- Q2: Stale Prior 严重高估（Q-error > 10）
- Q3: Stale Prior 估算较准（Q-error < 2）

### Template 2: 不等式查询

测试直方图对累积分布的估算。

```sql
-- Q4: 大于阈值（漂移后选择率变化）
SELECT COUNT(*), AVG(production_year)
FROM title
WHERE production_year > 1980;

-- Q5: 小于阈值
SELECT COUNT(*), AVG(production_year)
FROM title
WHERE production_year < 1960;

-- Q6: 组合不等式
SELECT COUNT(*)
FROM title
WHERE production_year > 1950 AND production_year < 1970;
```

### Template 3: 多列 Filter

测试多个直方图的联合估算。

```sql
-- Q7: 两列 Filter
SELECT COUNT(*), MIN(t.title)
FROM title t
JOIN movie_info mi ON t.id = mi.movie_id
WHERE t.production_year BETWEEN 1920 AND 1950
  AND mi.info_type_id IN (1, 2, 3);

-- Q8: 三列 Filter
SELECT COUNT(*)
FROM title t
JOIN cast_info ci ON t.id = ci.movie_id
WHERE t.production_year > 1990
  AND ci.role_id = 1
  AND t.kind_id = 1;
```

### Template 4: 聚合查询

测试 Filter 后的聚合估算。

```sql
-- Q9: 分组聚合
SELECT production_year, COUNT(*) as cnt
FROM title
WHERE production_year BETWEEN 1920 AND 2012
GROUP BY production_year
HAVING COUNT(*) > 100;

-- Q10: 多表聚合
SELECT t.production_year, COUNT(DISTINCT ci.person_id) as actor_count
FROM title t
JOIN cast_info ci ON t.id = ci.movie_id
WHERE t.production_year BETWEEN 1980 AND 2000
GROUP BY t.production_year;
```

### Template 5: 子查询 Filter

测试嵌套查询中的 Filter 估算。

```sql
-- Q11: IN 子查询
SELECT COUNT(*)
FROM cast_info
WHERE movie_id IN (
    SELECT id FROM title
    WHERE production_year BETWEEN 1920 AND 1950
);

-- Q12: EXISTS 子查询
SELECT COUNT(*)
FROM title t
WHERE EXISTS (
    SELECT 1 FROM movie_info mi
    WHERE mi.movie_id = t.id
      AND mi.info_type_id IN (1, 2, 3)
)
AND t.production_year > 1990;
```

---

## 完整查询集（20 个查询）

### 基础 Filter 查询（Q1-Q6）

专注于单表、单列的范围查询，直接测试直方图准确性。

### 多列 Filter 查询（Q7-Q12）

测试多个直方图的联合估算，以及 JOIN + Filter 的组合。

### 聚合查询（Q13-Q16）

测试 Filter 后的聚合操作，验证行数估算对聚合的影响。

### 复杂查询（Q17-Q20）

混合 JOIN、Filter、子查询，测试真实场景。

---

## 实验设计

### Phase 1: Baseline（无漂移）

```bash
# 1. 初始 ANALYZE
./analyze_tables.sh --core

# 2. 运行 IFW 查询
python3 run_ifw_experiment.py \
  --presto-host localhost:8080 \
  --strategy baseline \
  --output-dir ../results/ifw_baseline
```

**预期**：Q-error 很低（< 2），因为统计是新鲜的。

### Phase 2: Stale Prior（有漂移，过期统计）

```bash
# 3. 数据漂移（分布反转）
python3 inject_drift.py --rounds 15 --drift-ratio 0.02 --no-update

# 4. 运行 IFW 查询（使用过期统计）
python3 run_ifw_experiment.py \
  --presto-host localhost:8080 \
  --strategy stale_prior \
  --output-dir ../results/ifw_stale_prior
```

**预期**：Q-error 很高（> 10），因为：
- 查询 1920-1950：旧直方图认为稀少，实际变多
- 查询 2000-2012：旧直方图认为频繁，实际变少

### Phase 3: Full ANALYZE（有漂移，新鲜统计）

```bash
# 5. 重新 ANALYZE
./analyze_tables.sh --core

# 6. 触发新 Snapshot
./trigger_new_snapshot.sh

# 7. 运行 IFW 查询（使用新鲜统计）
python3 run_ifw_experiment.py \
  --presto-host localhost:8080 \
  --strategy full_analyze \
  --output-dir ../results/ifw_full_analyze
```

**预期**：Q-error 恢复到低水平（< 2），因为新直方图准确。

### Phase 4: OASIS（有漂移，模型修正）

```bash
# 8. Warmup（收集 observations）
python3 run_ifw_experiment.py \
  --presto-host localhost:8080 \
  --strategy warmup \
  --warmup-only \
  --warmup-count 5 \
  --output-dir ../results/ifw_warmup

# 9. OASIS 修正
python3 run_ifw_experiment.py \
  --presto-host localhost:8080 \
  --strategy oasis \
  --enable-correction \
  --output-dir ../results/ifw_oasis
```

**预期**：Q-error 接近 Full ANALYZE（< 3），证明 OASIS 有效。

---

## 预期结果

### Q-error 对比

| 查询 | Baseline | Stale Prior | Full ANALYZE | OASIS |
|------|----------|-------------|--------------|-------|
| Q1 (1920-1950) | 1.2 | **45.3** | 1.5 | 2.8 |
| Q2 (2000-2012) | 1.1 | **38.7** | 1.3 | 2.5 |
| Q3 (1970-1990) | 1.3 | 2.1 | 1.4 | 1.8 |
| Q4 (> 1980) | 1.5 | **12.4** | 1.6 | 3.2 |
| Q5 (< 1960) | 1.4 | **15.8** | 1.5 | 2.9 |
| ... | ... | ... | ... | ... |
| **平均** | **1.3** | **22.5** | **1.5** | **2.7** |

**关键观察**：
- Stale Prior 的 Q-error 是 Baseline 的 **17 倍**
- Full ANALYZE 恢复到接近 Baseline
- OASIS 在没有 ANALYZE 的情况下，Q-error 是 Stale Prior 的 **1/8**

### 执行时间对比

| 策略 | 总时间 | vs Baseline |
|------|--------|-------------|
| Baseline | 120s | - |
| Stale Prior | 185s | +54% |
| Full ANALYZE | 125s | +4% |
| OASIS | 135s | +13% |

**关键观察**：
- Stale Prior 因为错误的计划，执行时间增加 54%
- Full ANALYZE 恢复到接近 Baseline
- OASIS 的性能接近 Full ANALYZE，但无需昂贵的 ANALYZE

---

## 优势

### 1. 直接测试直方图

- JOB 测试 JOIN（不用直方图）
- IFW 测试 Filter（直接用直方图）
- **互补**

### 2. 清晰的因果关系

- 漂移 → 分布变化 → 直方图过期 → Q-error 增加
- ANALYZE → 直方图更新 → Q-error 降低
- **直接验证直方图的作用**

### 3. 可控的实验

- 查询简单，易于理解
- 预期结果明确
- 易于调试

### 4. 真实场景

- 范围查询在 OLAP 中很常见
- 时间序列分析（production_year）
- 分类过滤（kind_id, info_type_id）

---

## 实现计划

### 1. 生成查询文件

```bash
cd /Users/qichutian/presto/presto-cdf-simulation/job_experiment
mkdir -p queries/ifw

# 生成 20 个查询
python3 scripts/generate_ifw_queries.py --output queries/ifw/
```

### 2. 修改实验脚本

复用 `run_experiment.py`，添加 IFW 支持：
- 读取 `queries/ifw/` 目录
- 使用相同的 Q-error 收集逻辑
- 输出到 `results/ifw_*/`

### 3. 运行完整实验

```bash
# 一键运行
./run_ifw_experiment.sh

# 或分步运行
./run_ifw_experiment.sh --phase baseline
./run_ifw_experiment.sh --phase stale_prior
./run_ifw_experiment.sh --phase full_analyze
./run_ifw_experiment.sh --phase oasis
```

### 4. 分析结果

```bash
python3 analyze_ifw_results.py \
  --baseline results/ifw_baseline \
  --stale-prior results/ifw_stale_prior \
  --full-analyze results/ifw_full_analyze \
  --oasis results/ifw_oasis \
  --output results/ifw_summary
```

---

## 总结

**IFW (IMDB Filter Workload) 的核心价值**：

1. ✅ **直接测试直方图**：Filter 查询依赖直方图，不像 JOIN
2. ✅ **清晰的对比**：Stale vs Fresh 直方图的差异明显
3. ✅ **验证 OASIS**：证明模型修正能接近 Full ANALYZE 的效果
4. ✅ **互补 JOB**：JOB 测试 JOIN，IFW 测试 Filter

**与 JOB 的对比**：

| 维度 | JOB Benchmark | IFW Workload |
|------|---------------|--------------|
| 主要操作 | JOIN | Filter |
| 直方图使用 | ❌ 很少 | ✅ 大量 |
| Q-error 敏感度 | 低（依赖 NDV） | 高（依赖直方图） |
| 适合验证 | JOIN 优化 | 直方图修正 |

**建议**：
- 同时运行 JOB 和 IFW
- JOB 验证整体性能
- IFW 验证直方图修正效果
