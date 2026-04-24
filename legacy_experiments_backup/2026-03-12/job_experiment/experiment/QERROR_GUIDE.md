# PostgreSQL Q-Error 提取指南

## 什么是 Q-Error？

Q-Error（Quality Error）是衡量基数估算准确性的标准指标，定义如下：

```
Q-Error(estimated, actual) = max(estimated / actual, actual / estimated)
```

**关键特性**：
- Q-Error ≥ 1，1 表示完美估算
- 对称性：高估和低估同等对待
- 例如：
  - 估算 100，实际 100 → Q-Error = 1.0 ✓
  - 估算 200，实际 100 → Q-Error = 2.0 (高估 2x)
  - 估算 100，实际 200 → Q-Error = 2.0 (低估 2x)

## PostgreSQL EXPLAIN JSON 格式

PostgreSQL `EXPLAIN (ANALYZE, FORMAT JSON)` 返回以下结构：

```json
[{
  "Plan": {
    "Node Type": "Hash Join",
    "Plan Rows": 1000,      // 估算行数
    "Actual Rows": 500,     // 实际行数
    "Plans": [
      {
        "Node Type": "Seq Scan",
        "Relation Name": "title",
        "Plan Rows": 10000,
        "Actual Rows": 8000
      },
      ...
    ]
  },
  "Execution Time": 123.45
}]
```

## 提取的算子类型

我们关注以下算子的基数估算准确性：

| 算子类型 | 说明 |
|---------|------|
| Seq Scan | 顺序扫描 |
| Index Scan | 索引扫描 |
| Index Only Scan | 仅索引扫描 |
| Bitmap Heap Scan | 位图堆扫描 |
| Nested Loop | 嵌套循环 Join |
| Hash Join | 哈希 Join |
| Merge Join | 合并 Join |
| Hash | 哈希构建 |
| Materialize | 物化 |
| Sort | 排序 |

## 计算流程

```
1. 执行 EXPLAIN (ANALYZE, FORMAT JSON) <query>
2. 解析 JSON 获取计划树
3. 递归遍历所有节点
4. 对每个目标算子：
   - 提取 Plan Rows (estimated)
   - 提取 Actual Rows (actual)
   - 计算 Q-Error
5. 聚合所有算子的 Q-Error（几何平均）
```

## 与论文的对比

我们的 Q-Error 提取与论文一致：

| 方面 | 论文定义 | 本实现 |
|-----|---------|--------|
| Q-Error 公式 | max(est/act, act/est) | ✓ 相同 |
| 聚合方式 | 几何平均 | ✓ 相同 |
| 关注算子 | Join 和 Scan | ✓ 相同 |
| 特殊情况 (0行) | est+1 或 act+1 | ✓ 相同 |

## 示例输出

```
[15/113] Running 10a... ✓ 1250ms | ops=8 q-error=1.85
```

解读：
- 查询 10a 执行时间 1.25 秒
- 计划树中有 8 个目标算子
- 这些算子的几何平均 Q-Error 为 1.85

## 常见问题

### Q: 为什么有些查询没有 Q-Error 数据？

可能原因：
1. 查询执行超时或失败
2. EXPLAIN JSON 解析出错
3. 计划树中没有目标算子（不太可能）

### Q: Q-Error 多大算好？

参考标准：
- 1.0-2.0：优秀
- 2.0-5.0：可接受
- 5.0-10.0：偏差较大
- >10.0：严重偏差

### Q: 如何调试 Q-Error 提取？

查看结果 JSON 中的 `plan_raw` 字段：

```bash
python3 -c "
import json
with open('results/stale_prior_results.json') as f:
    data = json.load(f)
    for r in data['results']:
        if r['query_id'] == '10a':
            print(r['plan_raw'][:2000])
"
```

## 测试验证

运行测试套件确保 Q-Error 提取正确：

```bash
python3 test_qerror_extraction.py
```

测试包括：
1. Q-Error 计算公式验证
2. EXPLAIN JSON 解析验证
3. 几何平均计算验证
