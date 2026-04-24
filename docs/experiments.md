# 实验指南

## 1. 合成数据消融实验

比较 6 种方法 (Prior / Teacher / STGrid / STHoles / QM / OASIS) 在不同漂移强度下的表现。

```bash
cd cdf_kll_ml_pipeline

# 默认: q ∈ {1,3,5,10,15,20,25,30}, 训练集 q={10,20}×1000, 测试集 128/组
python3 ablation_experiment.py

# 自定义漂移梯度
python3 ablation_experiment.py --q-values 1 5 10 20 40 --k-test 256

# 不绘图
python3 ablation_experiment.py --no-plot
```

输出: `ablation_work/ablation_results.csv` + `ablation_plot.png`

## 2. 统一论文实验套件

包含三个子实验，自动生成 LaTeX 表格和 PDF 图。

```bash
cd experiments

# 主消融实验 (Q-error / MAE / 选择率误差, 5 种方法 × 8 个 q 值)
python3 run_synthetic_paper_suite.py --suites main

# 敏感度分析 (观测窗口 K ∈ {4,8,16,32})
python3 run_synthetic_paper_suite.py --suites sensitivity

# 初始分布泛化 (6 种分布: Gaussian Mixture / Uniform / Power-law / Bimodal / Triangular / Exponential)
python3 run_synthetic_paper_suite.py --suites distribution

# 全部运行
python3 run_synthetic_paper_suite.py --suites all

# 强制重新训练模型
python3 run_synthetic_paper_suite.py --suites all --force-retrain
```

输出位置: `experiments/results/synthetic_paper_suite/`
- `main/summary.csv` + LaTeX 表格
- `sensitivity/summary.csv` + LaTeX 表格
- `distribution/summary.csv` + LaTeX 表格
- `figures/*.pdf` 消融图

## 3. 观测窗口敏感度

单独运行 K 值敏感度分析。

```bash
cd cdf_kll_ml_pipeline
python3 sensitivity_k.py --work-dir sensitivity_work
```

## 4. 直方图格式泛化

测试 OASIS 在不同直方图格式（等深 / 等宽 / V-optimal）间转换的精度损失。

```bash
cd cdf_kll_ml_pipeline
python3 extensibility_experiment.py --work-dir extensibility_work --k-test 200
```

## 5. 推理延迟测试

测量 Teacher 和 MLP 的单次推理延迟，验证在 Presto CBO 200ms 规划预算内。

```bash
cd cdf_kll_ml_pipeline
python3 latency_experiment.py
```

## 6. TPC-DS 端到端实验

在真实数据库上验证 Q-error 改善。

### PostgreSQL

```bash
# 1. 注入漂移
cd tpcds_experiment/drift
python3 inject_scd2_drift_pg.py --host localhost --dbname tpcds --user postgres

# 2. 运行实验
cd ../experiment
python3 run_simple_pg_experiment.py \
    --host localhost --dbname tpcds --user postgres \
    --query-dir ../queries --strategy stale_prior \
    --output-dir ../results/stale_prior

python3 run_simple_pg_experiment.py \
    --host localhost --dbname tpcds --user postgres \
    --query-dir ../queries --strategy full_analyze \
    --output-dir ../results/full_analyze
```

### MySQL

```bash
cd tpcds_experiment_mysql
# 同理使用 drift/ 和 experiment/ 子目录
```

## 评估指标

| 指标 | 定义 | 含义 |
|------|------|------|
| Q-Error | max(est/act, act/est) | 选择率估算偏差（1 = 完美） |
| Quantile MAE | 修正分位值与真值的平均绝对误差 | 分位数结构精度 |
| Selectivity MAE | 随机 `<` 谓词选择率的平均绝对误差 | 实际查询估算质量 |

## 关键实验参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--q-values` | 1,3,5,10,15,20,25,30 | 漂移强度（每次观测间的 DML 批次数） |
| `--k-train` | 1000 | 每个漂移等级的训练样本数 |
| `--k-test` | 128 | 每个漂移等级的测试样本数 |
| `--num-buckets` | 10 | 直方图桶数（B-1=9 个内部分位点） |
| `--max-observations` | 16 | 观测窗口大小 K |
| `--initial-rows` | 5000 | 内存表初始行数 |
| `--seed` | 42 | 随机种子 |
