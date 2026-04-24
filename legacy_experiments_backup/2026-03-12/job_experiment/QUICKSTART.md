# JOB Benchmark 端到端实验 - 快速开始（PostgreSQL）

## 实验逻辑说明

本实验旨在评估 **统计过期场景** 下 OASIS 的修正效果，正确流程应为：

1. **Stale Prior**（有漂移，不修正）← 生产环境的真实问题状态，**真正的baseline**
2. **OASIS**（有漂移，模型修正）← 本文提出的解决方案
3. **Full ANALYZE**（重新ANALYZE后）← 理想上界，但生产不可行

> ⚠️ **注意**：实验前无需测试"无漂移baseline"，因为这不是我们要解决的问题。我们要解决的是"统计已过期"场景下的选择率估算偏差。

---

## 一、一键运行（推荐）

```bash
cd job_experiment

# 完整实验（包含数据准备、漂移注入、对比测试）
./run_full_experiment_pg.sh

# 跳过数据准备（如果已经加载过数据）
./run_full_experiment_pg.sh --skip-setup

# 自定义漂移强度（默认 q=15）
./run_full_experiment_pg.sh --drift-rounds 20
```

---

## 二、分步运行（详细流程）

### Step 1: 数据准备

```bash
cd setup

# 1.1 下载 IMDB 数据集（~3.6GB，需要 10-15 分钟）
./1_download_imdb.sh

# 1.2 创建 PostgreSQL 数据库和表
psql -U postgres -c "CREATE DATABASE imdb;"
psql -U postgres -d imdb -f 2_create_tables_pg.sql

# 1.3 加载数据（使用 COPY，需要 15-30 分钟）
python3 3_load_data_pg.py \
  --data-dir ./imdb_data/imdb \
  --dbname imdb \
  --user postgres \
  --host localhost \
  --port 5432

# 1.4 生成初始统计（需要 5-10 分钟）
psql -U postgres -d imdb -f 4_initial_analyze_pg.sql
```

**此时状态**：统计信息新鲜，数据分布与统计一致。

---

### Step 2: 启动模型服务

```bash
cd ../..
cd cdf_kll_ml_pipeline

# 安装依赖
pip install -r requirements-service.txt

# 启动服务
python3 model_service.py &

# 验证
curl http://localhost:8080/health
```

---

### Step 3: 注入漂移（使统计过期）

```bash
cd ../job_experiment/drift

# 注入 q=15 轮漂移（默认 drift-ratio=0.02）
# 这将模拟真实生产环境中统计过期的场景
python3 inject_drift_pg.py \
  --dbname imdb \
  --user postgres \
  --host localhost \
  --port 5432 \
  --rounds 15 \
  --drift-ratio 0.02
```

**此时状态**：统计信息已过期（基于旧数据），但实际数据分布已改变。

---

### Step 4: Stale Prior 测试（过期统计，不修正）

> 🎯 **目的**：测量生产环境中"统计过期但不修正"的真实性能，作为后续对比的baseline。

```bash
cd ../experiment

python3 run_experiment_pg.py \
  --dbname imdb \
  --user postgres \
  --host localhost \
  --port 5432 \
  --query-dir ../queries/job \
  --strategy stale_prior \
  --output-dir ../results/stale_prior \
  --no-correction
```

---

### Step 5: OASIS 测试（过期统计，模型修正）

> 🎯 **目的**：验证 OASIS 在统计过期场景下能否通过模型修正恢复性能。

```bash
# 5.1 Warmup：收集初始 observations（用于模型修正）
python3 run_experiment_pg.py \
  --dbname imdb \
  --user postgres \
  --host localhost \
  --port 5432 \
  --query-dir ../queries/job \
  --strategy warmup \
  --output-dir ../results/warmup \
  --warmup-only \
  --warmup-count 20

# 5.2 OASIS 测试（启用模型修正）
python3 run_experiment_pg.py \
  --dbname imdb \
  --user postgres \
  --host localhost \
  --port 5432 \
  --query-dir ../queries/job \
  --strategy oasis \
  --output-dir ../results/oasis \
  --enable-correction
```

---

### Step 6: Full ANALYZE 测试（理想上界）

> 🎯 **目的**：测量"重新ANALYZE后"的理想性能，作为上界参照（实际生产不可行）。

```bash
# 6.1 重新 ANALYZE（刷新统计信息）
cd ../setup
psql -U postgres -d imdb -f 4_initial_analyze_pg.sql

# 6.2 测试（统计信息已刷新）
cd ../experiment
python3 run_experiment_pg.py \
  --dbname imdb \
  --user postgres \
  --host localhost \
  --port 5432 \
  --query-dir ../queries/job \
  --strategy full_analyze \
  --output-dir ../results/full_analyze \
  --no-correction
```

---

### Step 7: 结果分析

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

## 三、实验设计说明

### 为什么要先漂移再测试？

| 实验组 | 统计状态 | 数据状态 | 说明 |
|--------|----------|----------|------|
| **Stale Prior** | 过期（基于旧数据） | 已漂移（新分布） | 生产环境的真实问题场景 |
| **OASIS** | 过期（基于旧数据） | 已漂移（新分布） | 用模型修正过期统计 |
| **Full ANALYZE** | 新鲜（基于新数据） | 已漂移（新分布） | 理想上界，但开销大 |

**核心对比**：
- Stale Prior vs OASIS：证明模型修正的有效性
- OASIS vs Full ANALYZE：证明模型接近理想上界

### 为什么不测试"无漂移baseline"？

"无漂移"（统计新鲜）不是我们要解决的问题。OASIS 的目标是解决**统计已过期**场景下的选择率估算偏差。因此：
- **Stale Prior** 才是真正的 baseline
- 如果想验证数据加载正确，可以单独运行 Fresh Stats 测试，但不应作为论文 Table 7 的对比组

---

## 四、预期结果

基于论文 §6.5，预期在 q=15 时的结果：

| Method | Total time (s) | Success | vs. Stale |
|---|---:|---:|---:|
| **Stale Prior** | ~349 | 113/113 | — |
| **OASIS** | ~311 | 113/113 | **-10.8%** |
| **Full ANALYZE** | ~297 | 113/113 | -14.8% |

**关键观察**：
- Stale Prior 会因统计过期导致次优计划选择
- OASIS 应显著减少执行时间，接近 Full ANALYZE 的性能
- Full ANALYZE 是理想上界，但实际不可行（开销太大）

---

## 五、故障排查

### 问题 1：数据加载失败

```bash
# 检查 PostgreSQL 连接
psql -U postgres -c "SELECT version();"

# 检查表是否创建
psql -U postgres -d imdb -c "\dt"
```

### 问题 2：模型服务连接失败

```bash
# 检查服务状态
curl http://localhost:8080/health

# 查看服务日志
tail -f ../../cdf_kll_ml_pipeline/model_service.log
```

### 问题 3：查询超时

```bash
# 增加超时时间（修改 run_experiment_pg.py 中的 timeout 参数）
# 或跳过已知慢查询
python3 run_experiment_pg.py --skip-queries 33c,17f ...
```

### 问题 4：OASIS 效果不明显

可能原因：
1. **observations 不足**：检查是否已完成 warmup 步骤
2. **漂移强度不够**：尝试增加 `--rounds` 参数
3. **模型未加载**：检查 model_service 日志确认模型已加载

---

## 六、清理

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

## 七、时间估算

| 阶段 | 预计时间 |
|---|---:|
| 下载数据 | 10-15 分钟 |
| 加载数据 | 15-30 分钟 |
| 初始 ANALYZE | 5-10 分钟 |
| 注入漂移 | 5-10 分钟 |
| **Stale Prior 测试** | 20-30 分钟 |
| **OASIS Warmup** | 5-10 分钟 |
| **OASIS 测试** | 15-25 分钟 |
| 重新 ANALYZE | 5-10 分钟 |
| **Full ANALYZE 测试** | 15-25 分钟 |
| **总计** | **~1.5-2.5 小时** |

使用 `./run_full_experiment_pg.sh` 可以自动化整个流程。

---

## PostgreSQL vs Presto 对比

| 方面 | Presto + Iceberg | PostgreSQL |
|------|------------------|------------|
| 数据加载 | Spark (30-60分钟) | COPY (15-30分钟) |
| ANALYZE | 10-20分钟 | 5-10分钟 |
| 统计存储 | Puffin + KLL Sketch | pg_statistic + histogram_bounds |
| 反馈收集 | EventListener SPI | pg_stat_statements + hooks |
| 配置方式 | Session properties | GUC parameters |
| 总实验时间 | ~2-3小时 | ~1.5-2.5小时 |

**PostgreSQL 优势**：
- ✅ ANALYZE 更快（无需 Spark）
- ✅ 数据加载更简单（COPY 命令）
- ✅ 无需 Hive Metastore
- ✅ 统计系统更轻量
