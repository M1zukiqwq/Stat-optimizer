# OASIS: Online Adaptive Statistical Improvement System

OASIS 利用查询反馈在线修正数据库查询优化器的过期统计信息（直方图分位数），无需重新执行 ANALYZE。

## 问题

数据库 CBO 依赖统计信息估算查询选择率。数据因 DML 漂移后统计信息变旧，导致选择率估算偏差、查询计划次优。

## 解决方案

收集过去查询的 `estimated_selectivity` 与 `actual_selectivity`，通过 ML 模型预测修正后的分位数值，供优化器使用。

## 项目结构

```
.
├── cdf_kll_ml_pipeline/       # 核心管线：类型定义、校正方法、数据生成、实验脚本
├── experiments/                # 统一论文实验套件
├── tpcds_experiment/           # PostgreSQL TPC-DS 端到端实验
├── tpcds_experiment_mysql/     # MySQL TPC-DS 端到端实验
├── scripts/                    # 运维脚本（Iceberg 直方图同步等）
├── attention_heatmap.py        # 多头注意力可视化工具
├── docs/                       # 文档
└── paper/                      # LaTeX 论文源文件
```

## 快速开始

```bash
cd cdf_kll_ml_pipeline

# 1. 生成合成训练/测试数据并运行消融实验
python3 ablation_experiment.py

# 2. 运行完整论文实验套件（消融 + 敏感度 + 分布泛化）
cd ../experiments
python3 run_synthetic_paper_suite.py --suites all

# 3. 测量推理延迟
cd ../cdf_kll_ml_pipeline
python3 latency_experiment.py
```

## 依赖

- Python 3.8+
- NumPy
- matplotlib（可选，用于绘图）
- psycopg2（TPC-DS PostgreSQL 实验需要）
- PyMySQL（TPC-DS MySQL 实验需要）
