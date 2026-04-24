# Q-Error 提取验证指南

## 验证方法

我们提供三种验证方式，确保 Q-Error 提取逻辑正确：

### 方法一：单元测试（无需数据库）

```bash
cd job_experiment/experiment
python3 test_qerror_extraction.py
```

**测试内容**：
- Q-Error 计算公式验证（10个测试用例）
- 模拟 EXPLAIN JSON 解析验证
- 几何平均计算验证

### 方法二：真实数据库验证（推荐）

**前置条件**：
- PostgreSQL 已安装并运行
- IMDB 数据已加载

**运行验证**：

```bash
cd job_experiment/experiment

# 基本验证
python3 verify_qerror_real.py --dbname imdb --user your_username

# 使用特定查询文件
python3 verify_qerror_real.py \
    --dbname imdb \
    --user your_username \
    --query-file ../queries/job/10a.sql
```

**验证内容**：
1. **Simple Query** - 单表带 WHERE 条件
2. **Join Query** - 两表 JOIN
3. **JOB Query** - 复杂多表 JOIN

**预期输出**：
```
PostgreSQL Q-Error Extraction Real Verification
======================================================================
Connecting to: postgres@localhost:5432/imdb

✓ Connected to PostgreSQL

======================================================================
Test 1: Simple Query
======================================================================
Query: SELECT * FROM title WHERE production_year BETWEEN 2000 AND 2010 LIMIT 100

Raw EXPLAIN JSON (first 2000 chars):
[
  {
    "Plan": {
      "Node Type": "Limit",
      "Plan Rows": 100,
      "Actual Rows": 100,
      ...
    }
  }
]

Extracted 3 operators:
1. Seq Scan
   Relation: title
   Estimated: 125,000
   Actual:    98,234
   Q-Error:   1.27

...

Geometric Mean Q-Error: 1.45

✓ PASSED: Simple Query
```

### 方法三：手动验证

**步骤 1**: 在 psql 中运行 EXPLAIN ANALYZE

```sql
EXPLAIN (ANALYZE, FORMAT JSON)
SELECT * FROM title WHERE production_year BETWEEN 2000 AND 2010 LIMIT 100;
```

**步骤 2**: 保存 JSON 输出到文件

```bash
psql -U postgres -d imdb -c "
EXPLAIN (ANALYZE, FORMAT JSON)
SELECT * FROM title WHERE production_year BETWEEN 2000 AND 2010 LIMIT 100;
" > /tmp/explain_output.json
```

**步骤 3**: 使用 Python 验证提取

```python
import json
import sys
sys.path.insert(0, '.')
from run_simple_pg_experiment import PostgreSQLExperimentRunner

# 读取 JSON
with open('/tmp/explain_output.json') as f:
    explain_output = f.read()

# 解析
runner = PostgreSQLExperimentRunner.__new__(PostgreSQLExperimentRunner)
stats = runner.extract_plan_stats(explain_output)

# 手动验证
for stat in stats:
    est = stat['estimated_rows']
    act = stat['actual_rows']
    qe = stat['q_error']
    
    # 手动计算 Q-Error 验证
    import math
    expected_qe = max(est/act, act/est) if est > 0 and act > 0 else 1.0
    
    print(f"{stat['operator']}: est={est}, act={act}")
    print(f"  Extracted Q-Error: {qe}")
    print(f"  Expected Q-Error:  {expected_qe}")
    print(f"  Match: {abs(qe - expected_qe) < 0.01}")
    print()
```

## 验证检查清单

运行验证后，检查以下几点：

### ✅ 算子提取
- [ ] 提取到了 Seq Scan / Index Scan
- [ ] 提取到了 Join 算子（Hash Join / Nested Loop / Merge Join）
- [ ] 没有重复提取
- [ ] 没有遗漏明显的算子

### ✅ Q-Error 计算
- [ ] Q-Error >= 1（所有值）
- [ ] 估算 = 实际时，Q-Error = 1
- [ ] 估算 2x 实际时，Q-Error = 2
- [ ] 实际 2x 估算时，Q-Error = 2

### ✅ 聚合计算
- [ ] 几何平均合理（不会被极端值主导）
- [ ] 与手动计算的聚合值一致

## 常见问题排查

### 问题 1: 没有提取到任何算子

**症状**：`Extracted 0 operators`

**排查**：
```bash
# 检查 JSON 格式
python3 -c "
import json
with open('/tmp/explain_output.json') as f:
    data = json.load(f)
    print(json.dumps(data, indent=2)[:3000])
"

# 确认有 Plan 节点
python3 -c "
import json
with open('/tmp/explain_output.json') as f:
    data = json.load(f)
    if isinstance(data, list):
        data = data[0]
    print('Node Type:', data.get('Plan', {}).get('Node Type'))
    print('Has Plans:', 'Plans' in data.get('Plan', {}))
"
```

### 问题 2: Q-Error 值异常

**症状**：Q-Error 极大（>1000）或小于 1

**排查**：
```python
# 打印详细信息
for stat in stats:
    print(f"Operator: {stat['operator']}")
    print(f"  Estimated: {stat['estimated_rows']}")
    print(f"  Actual:    {stat['actual_rows']}")
    print(f"  Q-Error:   {stat['q_error']}")
    
    # 手动计算验证
    est, act = stat['estimated_rows'], stat['actual_rows']
    if act == 0 and est == 0:
        expected = 1.0
    elif act == 0:
        expected = est + 1
    elif est == 0:
        expected = act + 1
    else:
        expected = max(est/act, act/est)
    
    print(f"  Expected:  {expected}")
    print()
```

### 问题 3: 与论文结果不一致

可能原因：
1. **算子过滤不同**：确认提取了相同的算子类型
2. **聚合方式不同**：确认使用几何平均而非算术平均
3. **数据状态不同**：确认统计信息的 freshness

## 示例验证会话

```bash
# 1. 启动 psql
psql -U postgres -d imdb

# 2. 运行查询并查看计划
EXPLAIN (ANALYZE, VERBOSE)
SELECT MIN(t.title) 
FROM title t 
JOIN movie_info mi ON t.id = mi.movie_id 
WHERE t.production_year = 2005;

# 3. 查看 JSON 格式
EXPLAIN (ANALYZE, FORMAT JSON)
SELECT MIN(t.title) 
FROM title t 
JOIN movie_info mi ON t.id = mi.movie_id 
WHERE t.production_year = 2005;

# 4. 退出 psql，运行验证脚本
\q

python3 verify_qerror_real.py --dbname imdb --user postgres
```

## 验证通过标准

验证通过的条件：
1. ✅ 能连接到 PostgreSQL
2. ✅ 能成功执行 EXPLAIN ANALYZE
3. ✅ 提取到非零数量的算子
4. ✅ 所有 Q-Error >= 1
5. ✅ 几何平均计算合理

## 下一步

验证通过后，可以运行完整实验：

```bash
cd job_experiment
./run_stale_vs_analyze.sh
```
