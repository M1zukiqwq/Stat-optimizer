# IMDB 端到端实验流程（PostgreSQL 版）

> 本文档描述 PostgreSQL 环境下的端到端实验步骤
> 对比策略：Stale Prior（过期统计）vs OASIS（在线修正）vs Full ANALYZE（新鲜统计）

## 实验目标

评估 OASIS 在 JOB Benchmark 上的端到端性能，对应论文 §6.5 Table 7 & 8。

---

## 前置条件

1. **PostgreSQL 14+** 已安装并运行
2. **Python 3.8+** 及 `psycopg2` 库（`pip install psycopg2-binary`）
3. **IMDB 数据已下载**在 `setup/imdb_data/`
4. **JOB 查询文件**在 `queries/job/`（113 个 SQL 文件）

---

## 实验步骤

### Step 0: 环境变量设置

```bash
export PGHOST=localhost
export PGPORT=5432
export PGUSER=postgres
export PGDATABASE=imdb
```

---

### Step 1: 数据加载

```bash
cd job_experiment/setup

# 1.1 创建数据库
psql -U postgres -c "CREATE DATABASE imdb;"

# 1.2 创建表
psql -U postgres -d imdb -f 2_create_tables_pg.sql

# 1.3 加载数据（使用 COPY，15-30 分钟）
python3 3_load_data_pg.py \
  --data-dir ./imdb_data/imdb \
  --dbname imdb \
  --user postgres

# 验证数据加载
for table in title cast_info movie_info movie_companies name movie_keyword; do
    count=$(psql -U postgres -d imdb -t -c "SELECT COUNT(*) FROM $table;")
    echo "$table: $count rows"
done
```

---

### Step 2: 初始 ANALYZE（生成统计信息）

```bash
# 使用默认统计参数（100 buckets）
psql -U postgres -d imdb -f 4_initial_analyze_pg.sql
```

**此时状态**：统计信息新鲜，数据分布与统计一致。

---

### Step 3: 启动模型服务

```bash
cd ../../cdf_kll_ml_pipeline

# 安装依赖
pip install -r requirements-service.txt

# 启动服务
python3 model_service.py &

# 验证
curl http://localhost:8080/health
```

---

### Step 4: 注入数据漂移（使统计过期）

使用**分布反转策略**让漂移更有效：

```bash
cd ../job_experiment/drift

# 注入 q=15 轮漂移
python3 inject_drift_pg.py \
  --dbname imdb \
  --user postgres \
  --rounds 15 \
  --drift-ratio 0.02
```

**分布反转策略说明**：
- **INSERT**：让稀少值变多（如 title.production_year=1920-1950）
- **DELETE**：让频繁值变少（如 title.production_year=2000-2012）
- **效果**：CBO 在两个方向上同时犯错，导致次优计划

**此时状态**：统计信息已过期（基于旧数据），但实际数据分布已改变。

---

### Step 5: Stale Prior 测试（过期统计，不修正）

> 🎯 **目的**：测量生产环境中"统计过期但不修正"的真实性能，作为 baseline。

```bash
cd ../experiment

python3 run_experiment_pg.py \
  --dbname imdb \
  --user postgres \
  --query-dir ../queries/job \
  --strategy stale_prior \
  --output-dir ../results/stale_prior \
  --no-correction
```

**输出**：
- `stale_prior_results.json`：包含每个查询的执行时间和 Q-error

**预期时间**：20-30 分钟

---

### Step 6: OASIS 测试（过期统计，模型修正）

> 🎯 **目的**：验证 OASIS 在统计过期场景下能否通过模型修正恢复性能。

```bash
# 6.1 Warmup：收集初始 observations（用于模型修正）
python3 run_experiment_pg.py \
  --dbname imdb \
  --user postgres \
  --query-dir ../queries/job \
  --strategy warmup \
  --output-dir ../results/warmup \
  --warmup-only \
  --warmup-count 20

# 6.2 OASIS 测试（启用模型修正）
python3 run_experiment_pg.py \
  --dbname imdb \
  --user postgres \
  --query-dir ../queries/job \
  --strategy oasis \
  --output-dir ../results/oasis \
  --enable-correction
```

**预期时间**：15-25 分钟

---

### Step 7: Full ANALYZE 测试（理想上界）

> 🎯 **目的**：测量"重新ANALYZE后"的理想性能，作为上界参照。

```bash
# 7.1 重新 ANALYZE（刷新统计信息）
cd ../setup
psql -U postgres -d imdb -f 4_initial_analyze_pg.sql

# 7.2 测试（统计信息已刷新）
cd ../experiment
python3 run_experiment_pg.py \
  --dbname imdb \
  --user postgres \
  --query-dir ../queries/job \
  --strategy full_analyze \
  --output-dir ../results/full_analyze \
  --no-correction
```

**预期时间**：15-25 分钟

---

### Step 8: 结果分析

```bash
python3 analyze_results.py \
  --stale-prior ../results/stale_prior \
  --oasis ../results/oasis \
  --full-analyze ../results/full_analyze \
  --output ../results/summary
```

查看结果：
```bash
# 查看 Table 7（论文格式）
cat ../results/summary/table7.csv

# 查看详细日志
less ../results/oasis/oasis_results.json
```

---

## 预期结果

基于论文 §6.5，预期在 q=15 时的结果：

### Table 7: JOB 端到端执行时间

| Method | Total time (s) | Success | vs. Stale |
|---|---:|---:|---:|
| **Stale Prior** | 348.8 | 113/113 | — |
| **OASIS v2** | 311.2 | 113/113 | **-10.8%** |
| **Full ANALYZE** | 297.1 | 113/113 | -14.8% |

### Table 8: JOB Q-Error 对比

| Method | Mean Q-Error | vs. Stale |
|---|---:|---:|
| **Stale Prior** | 8.63 | — |
| **OASIS v2** | 7.32 | **-15.2%** |
| **Full ANALYZE** | 5.64 | -34.6% |

**关键观察**：
- OASIS 在无需重新 ANALYZE 的情况下，恢复了 Full ANALYZE 约 73% 的性能提升
- 执行时间减少 10.8%，Q-Error 降低 15.2%
- 推理延迟 < 1.1ms，远低于查询规划开销

---

## 故障排查

### Q-error 改善不明显

**可能原因**：
1. **observations 不足**：检查是否已完成 warmup 步骤
2. **漂移强度不够**：尝试增加 `--rounds` 参数到 20
3. **模型未加载**：检查 model_service 日志确认模型已加载

### 查询超时

```bash
# 增加超时时间（修改 run_experiment_pg.py 中的 timeout 参数）
# 或跳过已知慢查询
python3 run_experiment_pg.py --skip-queries 33c,17f
```

### PostgreSQL 连接失败

```bash
# 检查 PostgreSQL 是否运行
pg_isready -h localhost -p 5432

# 检查数据库是否存在
psql -U postgres -l | grep imdb
```

---

## 清理

```bash
# 停止模型服务
pkill -f model_service.py

# 删除实验数据
rm -rf results/

# 删除 feedback 数据
rm -rf /tmp/oasis-feedback/

# 删除 PostgreSQL 数据库（可选）
psql -U postgres -c "DROP DATABASE IF EXISTS imdb;"
```

---

## 时间估算

| 阶段 | 预计时间 |
|---|---:|
| 数据加载 | 15-30 分钟 |
| 初始 ANALYZE | 5-10 分钟 |
| 注入漂移 | 5-10 分钟 |
| **Stale Prior 测试** | 20-30 分钟 |
| **OASIS Warmup** | 5-10 分钟 |
| **OASIS 测试** | 15-25 分钟 |
| 重新 ANALYZE | 5-10 分钟 |
| **Full ANALYZE 测试** | 15-25 分钟 |
| **总计** | **~1.5-2.5 小时** |

---

## PostgreSQL 优势

相比 Presto + Iceberg 方案：

| 方面 | Presto + Iceberg | PostgreSQL |
|------|------------------|------------|
| 数据加载 | Spark (30-60分钟) | COPY (15-30分钟) |
| ANALYZE | 10-20分钟 | 5-10分钟 |
| 统计存储 | Puffin + KLL Sketch | pg_statistic + histogram_bounds |
| 反馈收集 | EventListener SPI | pg_stat_statements + hooks |
| 配置方式 | Session properties | GUC parameters |
| 依赖组件 | Hive Metastore + Spark | 无额外依赖 |
| 总实验时间 | ~2-3小时 | ~1.5-2.5小时 |

**PostgreSQL 优势**：
- ✅ ANALYZE 更快（无需 Spark）
- ✅ 数据加载更简单（COPY 命令）
- ✅ 无需 Hive Metastore
- ✅ 统计系统更轻量
- ✅ 部署更简单

---

**文档版本**: 2026-03-09 (PostgreSQL 版)
**作者**: OASIS Experiment Team
