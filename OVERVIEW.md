# OASIS 项目总览

> 最后更新：2026-03-09
> 目标引擎：PostgreSQL

## 一、项目目标

本项目研究 **查询优化器中统计信息过期问题**，提出 **OASIS**（Online Adaptive Statistics Inference System）——一个基于查询执行反馈自动修正列直方图的系统。

**核心问题**：PostgreSQL 的列统计信息通过 `ANALYZE` 生成（存储在 `pg_statistic` 系统表中），而表数据持续写入，导致 CBO 使用过期直方图，选择率估算偏差 → 错误的 Join 顺序/算法选择 → 查询性能退化。

**解决方案**：不频繁重跑 ANALYZE，而是通过双层反馈收集架构（算子插桩 + 反馈监听插件）自动收集查询观测（谓词 + 估算选择率 + 实际选择率），在线修正直方图，使 CBO 获得更准确的统计信息。

---

## 二、目录结构

```
presto-cdf-simulation/
├── OVERVIEW.md                          ← 本文件：项目总览
├── PAPER_ARCHITECTURE.md                ← 论文架构文档
│
├── cdf_kll_ml_pipeline/                 ← 核心 ML Pipeline（详见 DESIGN.md）
│   ├── DESIGN.md                        ← 整合设计文档
│   ├── 核心模块 ──────────────────────
│   │   ├── json_histogram_parser.py     ← JSON 解析
│   │   ├── histogram_math.py            ← CDF 插值、保序回归
│   │   ├── histogram_types.py           ← 数据类型定义
│   │   └── tensorizer.py               ← 特征张量化（220维）
│   ├── 校正算法 ──────────────────────
│   │   ├── cdf_teacher.py               ← Teacher：无训练保序回归
│   │   ├── mlp_histogram_model.py       ← OASIS v1：单头注意力 MLP（~17K 参数）
│   │   ├── mlp_histogram_model_v2.py    ← OASIS v2：多头注意力残差 MLP（~38K 参数）★
│   │   └── baselines.py                 ← STHoles 经典基线
│   ├── 训练与推理 ─────────────────────
│   │   ├── simulate_memory_kll_dataset.py ← 内存表漂移模拟数据生成器
│   │   ├── train_mlp_model_v2.py        ← v2 训练脚本
│   │   └── predict_histogram.py         ← 推理入口
│   └── 实验脚本与结果 ────────────────
│       ├── ablation_experiment.py       ← 消融实验框架
│       ├── compare_v1_v2.py             ← v1 vs v2 对比实验
│       └── sensitivity_k.py             ← 窗口大小 K 灵敏度分析
│
├── paper/                               ← 论文 LaTeX 源码
│   ├── main.tex
│   ├── references.bib
│   └── figures/
│
├── ablation_study/                      ← 消融实验
│
└── job_experiment/                      ← JOB 端到端实验（PostgreSQL）
    ├── README.md
    ├── QUICKSTART.md
    ├── setup/
    ├── drift/
    ├── queries/
    └── results/
```

---

## 三、核心算法

### 3.1 数据表示：等深直方图

```
PostgreSQL pg_statistic
    → most_common_vals / histogram_bounds
    → 等价于等深直方图（bucket_boundaries）

归一化约束：min_value = 0.0, max_value = 1.0
```

### 3.2 特征张量结构（220 维）

```
feature_tensor = [
    prior_norm (B-1 维),      ← 先验分位点（归一化）
    meta (3 维),               ← null_fraction / obs_count_ratio / bucket_count
    observations (K × 12 维), ← 每条观测：谓词 one-hot(6) + 数值特征(6)
    mask (K 维)                ← 有效观测标记
]
默认 B=10, K=16 → 9 + 3 + 192 + 16 = 220 维
```

### 3.3 校正策略

| 策略 | 描述 | 参数量 | 延迟 | 是否需要训练 |
|---|---|---|---|---|
| **Prior** | 直接使用过期统计，baseline | — | — | 否 |
| **Teacher** | 先验 CDF + 观测约束 + 保序回归 | — | 0.08ms | 否 |
| **STHoles** | 经典自调优直方图 | — | — | 否 |
| **OASIS v1** | 单头注意力 + 2层 MLP | ~17K | 0.34ms | 是 |
| **OASIS v2** | 3头注意力 + Prior Encoder + 4层残差 MLP | ~38K | 1.06ms | 是 |

### 3.4 OASIS v2 模型架构（论文主模型）

```
输入 (220维)
  ↓ Prior Encoder: prior(9) → Linear(9→32) → ReLU → 32维
  ↓ 拆分为 K=16 个观测槽（各 12 维）
  ↓ 3-Head Attention → masked softmax → 3 × 加权池化 → concat (36维)
  ↓ Context = [prior_enc(32) | meta(3) | pooled_obs(36)] = 71维
  ↓ 128(ReLU) → 128(ReLU+skip) → 64(ReLU) → 64(ReLU+skip) → 9
  ↓ Residual: output = prior_norm + delta
输出: 9 个修正分位点（归一化）
```

---

## 四、实验结果

### 4.1 主结果（v2 heavy 模型，训练 q∈{10,20}，k=1500）

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

### 4.2 v1 vs v2 对比

| q | OASIS v1 | OASIS v2 | v2 优势 |
|:---:|---:|---:|---:|
| 5  | 1.565 | **1.459** | +6.8% |
| 10 | 1.367 | **1.325** | +3.1% |
| 15 | 1.595 | **1.424** | +10.7% |
| 20 | 1.552 | **1.458** | +6.1% |
| 25 | 1.705 | **1.448** | +15.1% |
| 30 | 1.556 | **1.427** | +8.3% |

### 4.3 JOB 端到端（IMDB，q=15）

| Method | Total time (s) | vs Stale |
|---|---:|---:|
| Stale Prior | 348.8 | — |
| OASIS | 311.2 | -10.8% |
| Full ANALYZE | 297.1 | -14.8% |

### 4.4 延迟测试

| 方法 | 延迟 (ms) | 参数量 |
|---|---:|---:|
| Teacher | 0.079 | — |
| OASIS v1 | 0.344 | ~17K |
| OASIS v2 | 1.059 | ~38K |

**关键结论**：
- OASIS v2 在 q≥5 时全面超越所有 baseline，q=25 时 Q-Error 降低 **56.4%** vs Prior
- q≤3 时模型分布外推，建议通过 Drift Gate 默认使用 Teacher
- 所有方法延迟远低于 CBO 200ms 规划预算
