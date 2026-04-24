# IMDB Filter Workload (IFW)

这些查询专门设计用于测试直方图的准确性。

## 查询分类

- **Q1-Q3**: 基础范围查询（单表、单列）
- **Q4-Q6**: 不等式查询（测试 CDF）
- **Q7-Q9**: 多列 Filter（测试联合估算）
- **Q10-Q12**: 聚合查询（测试 Filter + 聚合）
- **Q13-Q15**: 子查询（测试嵌套 Filter）
- **Q16-Q20**: 复杂查询（混合场景）

## 使用方法

```bash
# 运行所有查询
python3 run_experiment.py \
  --presto-host localhost:8080 \
  --query-dir queries/ifw \
  --strategy stale_prior \
  --output-dir results/ifw_stale_prior
```
