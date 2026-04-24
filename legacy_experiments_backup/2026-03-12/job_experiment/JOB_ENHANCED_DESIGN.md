# JOB-Enhanced: 增强 JOB 查询以测试直方图效果

## 设计思路

原始 JOB 查询主要是等值 JOIN（`t.id = mc.movie_id`），CBO 对这类 JOIN 的基数估算主要依赖 NDV，不使用直方图。

**JOB-Enhanced 的改进**：在 JOIN 列上增加范围 filter，使得直方图能够影响基数估算。

### 示例

**原始 JOB 查询 10a.sql**：
```sql
SELECT MIN(mc.note), MIN(t.title), MIN(t.production_year)
FROM company_type AS ct,
     info_type AS it,
     movie_companies AS mc,
     movie_info_idx AS mi_idx,
     title AS t
WHERE ct.kind = 'production companies'
  AND it.info = 'top 250 rank'
  AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
  AND ct.id = mc.company_type_id
  AND t.id = mc.movie_id
  AND t.id = mi_idx.movie_id
  AND mc.movie_id = mi_idx.movie_id
  AND it.id = mi_idx.info_type_id
```

**JOB-Enhanced 10a.sql**：
```sql
SELECT MIN(mc.note), MIN(t.title), MIN(t.production_year)
FROM company_type AS ct,
     info_type AS it,
     movie_companies AS mc,
     movie_info_idx AS mi_idx,
     title AS t
WHERE ct.kind = 'production companies'
  AND it.info = 'top 250 rank'
  AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
  -- 新增：JOIN 列的范围 filter
  AND t.production_year BETWEEN 1920 AND 1950  -- 直方图敏感
  AND mc.company_type_id > 0                    -- 直方图敏感
  -- 原有 JOIN 条件
  AND ct.id = mc.company_type_id
  AND t.id = mc.movie_id
  AND t.id = mi_idx.movie_id
  AND mc.movie_id = mi_idx.movie_id
  AND it.id = mi_idx.info_type_id
```

### 关键改进点

1. **在 JOIN 列上增加范围 filter**：
   - `t.production_year BETWEEN 1920 AND 1950`
   - `mc.company_type_id > 0`
   - `ci.role_id IN (1, 2, 3)`

2. **保留原有 JOIN 结构**：
   - 不改变 JOIN 顺序和表关系
   - 只增加 filter 条件

3. **直方图的作用**：
   - CBO 需要先估算 `t.production_year BETWEEN 1920 AND 1950` 的选择率
   - 然后再估算 JOIN 的基数
   - 如果直方图过期，选择率估算错误，导致 JOIN 基数估算错误

---

## 实现方案

### 方案 1：自动增强（推荐）

创建脚本自动给 JOB 查询增加范围 filter：

```python
#!/usr/bin/env python3
"""
自动增强 JOB 查询，增加直方图敏感的 filter 条件
"""

import re
from pathlib import Path

# 增强规则：表名 -> filter 条件
ENHANCEMENT_RULES = {
    'title': [
        "t.production_year BETWEEN 1920 AND 1950",
        "t.kind_id IN (1, 2, 3)",
    ],
    'cast_info': [
        "ci.role_id IN (1, 2)",
    ],
    'movie_info': [
        "mi.info_type_id BETWEEN 1 AND 10",
    ],
    'movie_companies': [
        "mc.company_type_id > 0",
    ],
}

def enhance_job_query(query_sql: str) -> str:
    """给 JOB 查询增加范围 filter"""
    # 找到 WHERE 子句的位置
    where_match = re.search(r'\bWHERE\b', query_sql, re.IGNORECASE)
    if not where_match:
        return query_sql

    # 在 WHERE 后面插入新的 filter 条件
    where_pos = where_match.end()

    # 检测查询中使用了哪些表
    filters_to_add = []
    for table, filters in ENHANCEMENT_RULES.items():
        # 检查表是否在查询中（通过别名）
        if re.search(rf'\b{table}\b.*\bAS\b\s+(\w+)', query_sql, re.IGNORECASE):
            filters_to_add.extend(filters)

    if not filters_to_add:
        return query_sql

    # 插入新的 filter 条件
    enhanced_filters = '\n  AND '.join(filters_to_add)
    enhanced_query = (
        query_sql[:where_pos] +
        '\n  ' + enhanced_filters + '\n  AND' +
        query_sql[where_pos:]
    )

    return enhanced_query


def main():
    job_dir = Path('queries/job')
    enhanced_dir = Path('queries/job_enhanced')
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    for query_file in sorted(job_dir.glob('*.sql')):
        with open(query_file) as f:
            original_query = f.read()

        enhanced_query = enhance_job_query(original_query)

        output_file = enhanced_dir / query_file.name
        with open(output_file, 'w') as f:
            f.write(enhanced_query)

        print(f"✓ {query_file.name}")

    print(f"\n✓ 完成！生成了 {len(list(enhanced_dir.glob('*.sql')))} 个增强查询")


if __name__ == '__main__':
    main()
```

### 方案 2：手动选择（精确控制）

手动选择 10-20 个代表性的 JOB 查询，精心设计 filter 条件：

**选择标准**：
1. 包含 `title` 表（可以加 `production_year` filter）
2. 包含 `cast_info` 表（可以加 `role_id` filter）
3. 包含 `movie_info` 表（可以加 `info_type_id` filter）
4. JOIN 数量 >= 3（复杂度足够）

**推荐查询**：
- 10a, 10b, 10c（包含 title + movie_companies）
- 11a, 11b, 11c（包含 title + cast_info）
- 12a, 12b, 12c（包含 title + movie_info）
- 13a, 13b, 13c（包含 title + cast_info + movie_info）

---

## 预期效果

### Stale Prior（过期直方图）

- `t.production_year BETWEEN 1920 AND 1950` 的选择率被严重低估
- 导致 `title` 表的基数估算错误
- 进而导致 JOIN 顺序选择错误
- **Q-error 增加**

### Full ANALYZE（新鲜直方图）

- `t.production_year BETWEEN 1920 AND 1950` 的选择率准确
- `title` 表的基数估算准确
- JOIN 顺序选择正确
- **Q-error 降低**

### 对比 IFW

| 维度 | IFW | JOB-Enhanced |
|------|-----|--------------|
| 查询类型 | 单表/简单 JOIN + Filter | 复杂多表 JOIN + Filter |
| 直方图作用 | 直接影响 Filter 选择率 | 影响 JOIN 前的 Filter 选择率 |
| 复杂度 | 低（1-3 表） | 高（5-10 表） |
| 适用场景 | 微观测试直方图精度 | 宏观测试直方图对复杂查询的影响 |

**三个查询集互补**：
- **JOB**：测试 JOIN 顺序优化（不依赖直方图）
- **IFW**：测试 Filter 选择率估算（直接依赖直方图）
- **JOB-Enhanced**：测试直方图对复杂 JOIN 查询的综合影响

---

## 论文中的论述

在 §5.5 End-to-End Evaluation 中可以这样论述：

> To further validate the impact of histogram corrections on complex join queries, we create **JOB-Enhanced**, a variant of the JOB benchmark where we augment join predicates with range filters on histogram-sensitive columns (e.g., `production_year BETWEEN 1920 AND 1950`). This design allows us to test whether histogram corrections can improve cardinality estimation in realistic multi-table join scenarios, where the optimizer must first estimate filter selectivity before determining join order.
>
> Results show that on JOB-Enhanced queries, stale histograms lead to X% higher Q-Error compared to JOB baseline, because the optimizer underestimates the selectivity of range filters on drifted columns. After OASIS correction, Q-Error drops to near-optimal levels, demonstrating that histogram corrections benefit not only filter-intensive queries (IFW) but also complex join queries when range predicates are present.

---

## 实现建议

1. **先运行 IFW**：验证直方图修正在纯 Filter 场景下有效
2. **再运行 JOB-Enhanced**：验证直方图修正在复杂 JOIN 场景下也有效
3. **对比三个查询集**：
   - JOB：基线（直方图影响小）
   - IFW：直方图影响大（纯 Filter）
   - JOB-Enhanced：直方图影响中等（JOIN + Filter）

这样可以全面展示 OASIS 在不同查询模式下的效果。
