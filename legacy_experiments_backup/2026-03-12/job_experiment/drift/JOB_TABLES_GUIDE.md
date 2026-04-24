# JOB Benchmark 表分类与漂移策略（引擎无关）

## 完整表列表（21 张表）

### 📊 事实表（Fact Tables）- 13 张

这些表存储事务数据，数据量大且频繁变化，**需要注入漂移**：

| 表名 | 行数 | 说明 | 漂移优先级 |
|------|------|------|-----------|
| `cast_info` | 36M | 演员-电影关联 | ⭐⭐⭐ 高 |
| `movie_info` | 15M | 电影详细信息 | ⭐⭐⭐ 高 |
| `movie_keyword` | 4.5M | 电影-关键词关联 | ⭐⭐⭐ 高 |
| `name` | 4.2M | 人名信息 | ⭐⭐⭐ 高 |
| `char_name` | 3.1M | 角色名称 | ⭐⭐ 中 |
| `person_info` | 2.9M | 人物详细信息 | ⭐⭐⭐ 高 |
| `movie_companies` | 2.6M | 电影-公司关联 | ⭐⭐⭐ 高 |
| `title` | 2.5M | 电影标题（核心表） | ⭐⭐⭐ 高 |
| `movie_info_idx` | 1.4M | 电影索引信息 | ⭐⭐ 中 |
| `aka_name` | 901K | 人名别名 | ⭐⭐ 中 |
| `aka_title` | 361K | 电影别名 | ⭐⭐ 中 |
| `complete_cast` | 135K | 完整演员表 | ⭐ 低 |
| `movie_link` | 29K | 电影关联 | ⭐ 低 |

### 📚 维度表（Dimension Tables）- 8 张

这些表存储参考数据，数据量小且相对静态，**通常不需要漂移**：

| 表名 | 行数 | 说明 | 是否漂移 |
|------|------|------|---------|
| `company_name` | 235K | 公司名称 | ❌ 可选 |
| `keyword` | 135K | 关键词字典 | ❌ 可选 |
| `comp_cast_type` | 4 | 演员表类型 | ❌ 不需要 |
| `company_type` | 4 | 公司类型 | ❌ 不需要 |
| `info_type` | 113 | 信息类型 | ❌ 不需要 |
| `kind_type` | 7 | 电影类型 | ❌ 不需要 |
| `link_type` | 18 | 关联类型 | ❌ 不需要 |
| `role_type` | 12 | 角色类型 | ❌ 不需要 |

---

## 漂移策略建议

### 策略 1：完整测试（推荐）

漂移所有 13 张事实表，最真实地模拟生产环境：

```bash
python3 inject_drift.py \
    --rounds 15 \
    --drift-ratio 0.02
    # 默认漂移所有事实表
```

**优点**：
- ✅ 最真实的数据漂移场景
- ✅ 全面测试 CBO 在复杂场景下的表现
- ✅ 覆盖所有 JOB 查询涉及的表

**缺点**：
- ⏱️ 耗时较长（约 60-90 分钟）
- 💾 需要更多存储空间

---

### 策略 2：核心表测试（快速）

只漂移 6 张最核心的表，快速验证方案：

```bash
python3 inject_drift.py \
    --rounds 15 \
    --drift-ratio 0.02 \
    --tables title cast_info movie_info movie_companies name movie_keyword
```

**优点**：
- ⚡ 快速完成（约 30-45 分钟）
- 💾 存储需求较小
- ✅ 覆盖大部分 JOB 查询

**缺点**：
- ⚠️ 部分查询可能不受影响
- ⚠️ 不够全面

---

### 策略 3：自定义测试

根据具体查询选择相关表：

```bash
# 示例：只测试与演员相关的查询
python3 inject_drift.py \
    --rounds 15 \
    --drift-ratio 0.02 \
    --tables cast_info name person_info char_name
```

---

## 为什么维度表不需要漂移？

### 1. 数据特性不同

**事实表**：
- 记录业务事件（新电影上映、演员参演）
- 数据量大，持续增长
- 频繁的 INSERT/UPDATE/DELETE

**维度表**：
- 存储参考数据（电影类型、角色类型）
- 数据量小，相对稳定
- 很少变化

### 2. 对查询的影响不同

**事实表漂移**：
- 影响 JOIN 的基数估计
- 影响过滤条件的选择性
- 直接影响查询计划

**维度表漂移**：
- 影响很小（数据量小）
- 通常被缓存
- 统计信息变化不大

### 3. 真实场景对应

在真实的 IMDB 数据库中：
- 每天有新电影上映 → `title` 表增长
- 每天有新演员信息 → `cast_info` 表增长
- 电影类型（动作、喜剧）几乎不变 → `kind_type` 表稳定

---

## 原始方案的问题

### EXPERIMENT_GUIDE.md（原版）

只选了 6 张表：
```bash
title cast_info movie_info movie_companies name movie_keyword
```

**问题**：
- ❌ 缺少 `name` 表（4.2M 行，很重要！）
- ❌ 缺少 `char_name` 表（3.1M 行）
- ❌ 缺少 `person_info` 表（2.9M 行）
- ❌ 缺少其他中等事实表

**影响**：
- 部分 JOB 查询不受漂移影响
- 实验结果不够全面

---

## 推荐配置

### 学术论文/完整评估

```bash
# 漂移所有 13 张事实表
python3 inject_drift.py --rounds 15 --drift-ratio 0.02
```

### 快速原型/调试

```bash
# 只漂移 6 张核心表
python3 inject_drift.py --rounds 5 --drift-ratio 0.02 \
    --tables title cast_info movie_info movie_companies name movie_keyword
```

### 特定查询测试

```bash
# 根据查询涉及的表选择
# 例如：测试 JOB 1a-5a 查询
python3 inject_drift.py --rounds 15 --drift-ratio 0.02 \
    --tables title cast_info movie_companies company_name
```

---

## 验证表覆盖率

检查 JOB 查询涉及哪些表：

```bash
cd /home/tianqc/presto-optimizer/presto-cdf-simulation/job_experiment/queries/job

# 统计每个表在查询中出现的次数
for table in title cast_info movie_info movie_companies name movie_keyword \
             person_info movie_info_idx aka_title aka_name complete_cast \
             movie_link char_name company_name keyword; do
    count=$(grep -r "FROM.*$table\|JOIN.*$table" *.sql | wc -l)
    echo "$table: $count 次"
done | sort -t: -k2 -rn
```

**预期输出**（按频率排序）：
```
title: 113 次          # 所有查询都用到
cast_info: 95 次       # 大部分查询
movie_info: 78 次
movie_companies: 65 次
name: 60 次
movie_keyword: 55 次
...
```

---

## 总结

| 方案 | 表数量 | 耗时 | 适用场景 |
|------|--------|------|---------|
| 完整测试 | 13 张事实表 | 60-90 分钟 | 论文发表、完整评估 |
| 核心测试 | 6 张核心表 | 30-45 分钟 | 快速验证、原型开发 |
| 自定义测试 | 按需选择 | 可变 | 特定查询分析 |

**推荐**：首次运行使用"核心测试"验证方案可行性，正式实验使用"完整测试"。

---

**文档版本**: 2024-03-09
**作者**: OASIS Experiment Team
