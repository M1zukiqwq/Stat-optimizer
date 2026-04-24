# PostgreSQL JOB Benchmark 实验环境

本目录包含在 PostgreSQL 上运行 JOB Benchmark 端到端实验的完整工具链。

## 文件说明

### 文档
- `POSTGRES_EXPERIMENT_MANUAL.md` - 完整实验手册（从编译到运行）

### 脚本
- `run_postgres_experiment.py` - PostgreSQL 实验运行器（执行查询，收集时间和 Q-error）
- `inject_drift_postgres.py` - 数据漂移注入脚本
- `analyze_postgres_results.py` - 结果分析脚本
- `run_experiment.sh` - 快速启动脚本（交互式菜单）

## 快速开始

### 1. 编译和安装 PostgreSQL

详见 `POSTGRES_EXPERIMENT_MANUAL.md` 第一部分。

```bash
# 设置环境变量
export PG_HOME=$HOME/postgres_workspace/pg_install
export PG_DATA=$HOME/postgres_workspace/pg_data
export PATH=$PG_HOME/bin:$PATH

# 下载、编译、安装
cd ~/postgres_workspace
wget https://ftp.postgresql.org/pub/source/v16.2/postgresql-16.2.tar.gz
tar -xzf postgresql-16.2.tar.gz
cd postgresql-16.2
./configure --prefix=$PG_HOME --enable-debug --enable-cassert
make -j8 && make install

# 初始化数据库
initdb -D $PG_DATA -E UTF8 --locale=C
pg_ctl -D $PG_DATA start
createdb imdb
```

### 2. 导入 IMDB 数据

详见 `POSTGRES_EXPERIMENT_MANUAL.md` 第二部分。

```bash
# 下载数据
cd ~/postgres_workspace
mkdir imdb_data && cd imdb_data
wget http://homepages.cwi.nl/~boncz/job/imdb.tgz
tar -xzf imdb.tgz

# 创建 schema 和导入数据（见手册）
psql -d imdb -f create_schema.sql
# ... 导入 CSV 文件
psql -d imdb -f create_indexes.sql
psql -d imdb -c "ANALYZE VERBOSE;"
```

### 3. 下载 JOB 查询

```bash
cd ~/postgres_workspace
git clone https://github.com/gregrahn/join-order-benchmark.git
```

### 4. 安装 Python 依赖

```bash
cd ~/postgres_workspace
python3 -m venv venv
source venv/bin/activate
pip install psycopg2-binary numpy pandas
```

### 5. 运行实验

#### 方式 1：使用交互式脚本（推荐）

```bash
cd /path/to/postgres_setup
chmod +x run_experiment.sh
./run_experiment.sh
```

#### 方式 2：手动运行

```bash
# Baseline 实验
python run_postgres_experiment.py \
    --db-name imdb \
    --query-dir ~/postgres_workspace/join-order-benchmark/queries \
    --strategy baseline \
    --output-dir results/baseline

# 注入漂移
python inject_drift_postgres.py \
    --db-name imdb \
    --drift-rounds 15 \
    --drift-ratio 0.02

# Stale Prior 实验
python run_postgres_experiment.py \
    --db-name imdb \
    --query-dir ~/postgres_workspace/join-order-benchmark/queries \
    --strategy stale_prior \
    --output-dir results/stale_prior

# Full ANALYZE 实验
psql -d imdb -c "ANALYZE VERBOSE;"
python run_postgres_experiment.py \
    --db-name imdb \
    --query-dir ~/postgres_workspace/join-order-benchmark/queries \
    --strategy full_analyze \
    --output-dir results/full_analyze

# 分析结果
python analyze_postgres_results.py \
    --baseline results/baseline/baseline_results.json \
    --stale results/stale_prior/stale_prior_results.json \
    --full-analyze results/full_analyze/full_analyze_results.json \
    --output results/comparison.json \
    --latex results/latex_tables.tex
```

## 实验流程

```
1. Baseline 实验
   ↓
2. 注入数据漂移（15 轮 × 2%）
   ↓
3. Stale Prior 实验（不更新统计）
   ↓
4. Full ANALYZE 实验（重新收集统计）
   ↓
5. 分析结果
```

## 预期结果

根据 Presto 实验结果，PostgreSQL 预期会有更显著的改进：

| 指标 | Presto | PostgreSQL（预期） |
|------|--------|-------------------|
| Q-Error 降低 | 15.2% | > 20% |
| 执行时间降低 | 10.8% | > 15% |
| 受益查询比例 | 16.3% | > 20% |

原因：PostgreSQL 的优化器更依赖直方图统计信息。

## 故障排查

### 问题 1：psycopg2 安装失败

```bash
# 使用二进制版本
pip install psycopg2-binary
```

### 问题 2：PostgreSQL 启动失败

```bash
# 查看日志
tail -f $PG_DATA/logfile

# 检查端口
netstat -tuln | grep 5432
```

### 问题 3：内存不足

```bash
# 降低 shared_buffers
psql -d imdb -c "ALTER SYSTEM SET shared_buffers = '512MB';"
pg_ctl -D $PG_DATA restart
```

## 修改 PostgreSQL 源码

如果需要修改 PostgreSQL 源码：

```bash
cd ~/postgres_workspace/postgresql-16.2

# 修改代码
# src/backend/optimizer/     - 查询优化器
# src/backend/utils/adt/     - 统计信息处理

# 重新编译
make -j8 && make install

# 重启数据库
pg_ctl -D $PG_DATA restart
```

## 常用命令

```bash
# 启动数据库
pg_ctl -D $PG_DATA start

# 停止数据库
pg_ctl -D $PG_DATA stop

# 连接数据库
psql -d imdb

# 查看统计信息
psql -d imdb -c "SELECT * FROM pg_stats WHERE tablename = 'title';"

# 查看执行计划
psql -d imdb -c "EXPLAIN ANALYZE SELECT * FROM title WHERE production_year = 2000;"
```

## 联系方式

如有问题，请参考 `POSTGRES_EXPERIMENT_MANUAL.md` 或联系实验负责人。
