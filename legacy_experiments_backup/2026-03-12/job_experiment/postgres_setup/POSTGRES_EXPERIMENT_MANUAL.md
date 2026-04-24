# PostgreSQL JOB Benchmark 实验手册

本手册指导在 Ubuntu 服务器上从源码编译 PostgreSQL，导入 IMDB 数据，并运行 JOB Benchmark 端到端实验。

**环境要求**：
- Ubuntu 服务器（无 root 权限）
- 至少 8GB 内存
- 至少 20GB 磁盘空间
- Python 3.8+

---

## 第一部分：编译和安装 PostgreSQL

### 1.1 准备工作目录

```bash
# 创建工作目录
mkdir -p ~/postgres_workspace
cd ~/postgres_workspace

# 设置环境变量（建议添加到 ~/.bashrc）
export PG_HOME=$HOME/postgres_workspace/pg_install
export PG_DATA=$HOME/postgres_workspace/pg_data
export PATH=$PG_HOME/bin:$PATH
export LD_LIBRARY_PATH=$PG_HOME/lib:$LD_LIBRARY_PATH
```

将以下内容添加到 `~/.bashrc`：
```bash
echo "export PG_HOME=$HOME/postgres_workspace/pg_install" >> ~/.bashrc
echo "export PG_DATA=$HOME/postgres_workspace/pg_data" >> ~/.bashrc
echo "export PATH=\$PG_HOME/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$PG_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

### 1.2 检查依赖

```bash
# 检查必需的编译工具
gcc --version
make --version
python3 --version

# 检查可选依赖（如果没有，编译时会跳过相关功能）
pkg-config --version
libreadline-dev --version  # 可能需要管理员安装
zlib1g-dev --version       # 可能需要管理员安装
```

### 1.3 下载 PostgreSQL 源码

```bash
cd ~/postgres_workspace

# 下载 PostgreSQL 16.2（最新稳定版）
wget https://ftp.postgresql.org/pub/source/v16.2/postgresql-16.2.tar.gz

# 解压
tar -xzf postgresql-16.2.tar.gz
cd postgresql-16.2
```

### 1.4 配置和编译

```bash
# 配置（安装到用户目录）
./configure --prefix=$PG_HOME \
    --enable-debug \
    --enable-cassert \
    --enable-depend \
    CFLAGS="-O0 -g3"

# 编译（使用多核加速，根据服务器核心数调整 -j 参数）
make -j8

# 安装到 $PG_HOME
make install

# 验证安装
$PG_HOME/bin/postgres --version
```

**预期输出**：
```
postgres (PostgreSQL) 16.2
```

### 1.5 初始化数据库集群

```bash
# 创建数据目录
mkdir -p $PG_DATA

# 初始化数据库
$PG_HOME/bin/initdb -D $PG_DATA -E UTF8 --locale=C

# 配置 PostgreSQL
cat >> $PG_DATA/postgresql.conf << EOF

# 性能优化配置
shared_buffers = 2GB
effective_cache_size = 6GB
maintenance_work_mem = 512MB
work_mem = 64MB
max_connections = 100

# 统计信息配置
default_statistics_target = 100
random_page_cost = 1.1

# 日志配置
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_statement = 'none'
log_duration = off
log_min_duration_statement = 1000

# 查询优化
enable_partitionwise_join = on
enable_partitionwise_aggregate = on
EOF
```

### 1.6 启动 PostgreSQL

```bash
# 启动数据库
$PG_HOME/bin/pg_ctl -D $PG_DATA -l $PG_DATA/logfile start

# 检查状态
$PG_HOME/bin/pg_ctl -D $PG_DATA status

# 创建测试数据库
$PG_HOME/bin/createdb imdb
```

**停止数据库**（需要时）：
```bash
$PG_HOME/bin/pg_ctl -D $PG_DATA stop
```

---

## 第二部分：导入 IMDB 数据

### 2.1 下载 JOB Benchmark 数据

```bash
cd ~/postgres_workspace
mkdir -p imdb_data
cd imdb_data

# 下载 IMDB 数据集（约 3.6GB）
wget http://homepages.cwi.nl/~boncz/job/imdb.tgz

# 解压
tar -xzf imdb.tgz
```

### 2.2 创建数据库 Schema

创建文件 `create_schema.sql`：

```sql
-- 创建所有表
CREATE TABLE aka_name (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    name text,
    imdb_index character varying(12),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE aka_title (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    title text,
    imdb_index character varying(12),
    kind_id integer NOT NULL,
    production_year integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note text,
    md5sum character varying(32)
);

CREATE TABLE cast_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    movie_id integer NOT NULL,
    person_role_id integer,
    note text,
    nr_order integer,
    role_id integer NOT NULL
);

CREATE TABLE char_name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL,
    imdb_index character varying(12),
    imdb_id integer,
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE comp_cast_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(32) NOT NULL
);

CREATE TABLE company_name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL,
    country_code character varying(255),
    imdb_id integer,
    name_pcode_nf character varying(5),
    name_pcode_sf character varying(5),
    md5sum character varying(32)
);

CREATE TABLE company_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(32) NOT NULL
);

CREATE TABLE complete_cast (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer,
    subject_id integer NOT NULL,
    status_id integer NOT NULL
);

CREATE TABLE info_type (
    id integer NOT NULL PRIMARY KEY,
    info character varying(32) NOT NULL
);

CREATE TABLE keyword (
    id integer NOT NULL PRIMARY KEY,
    keyword text NOT NULL,
    phonetic_code character varying(5)
);

CREATE TABLE kind_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(15) NOT NULL
);

CREATE TABLE link_type (
    id integer NOT NULL PRIMARY KEY,
    link character varying(32) NOT NULL
);

CREATE TABLE movie_companies (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    company_id integer NOT NULL,
    company_type_id integer NOT NULL,
    note text
);

CREATE TABLE movie_info (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text
);

CREATE TABLE movie_info_idx (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text
);

CREATE TABLE movie_keyword (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    keyword_id integer NOT NULL
);

CREATE TABLE movie_link (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    linked_movie_id integer NOT NULL,
    link_type_id integer NOT NULL
);

CREATE TABLE name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL,
    imdb_index character varying(12),
    imdb_id integer,
    gender character varying(1),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE person_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text
);

CREATE TABLE role_type (
    id integer NOT NULL PRIMARY KEY,
    role character varying(32) NOT NULL
);

CREATE TABLE title (
    id integer NOT NULL PRIMARY KEY,
    title text NOT NULL,
    imdb_index character varying(12),
    kind_id integer NOT NULL,
    production_year integer,
    imdb_id integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    series_years character varying(49),
    md5sum character varying(32)
);
```

### 2.3 导入数据

```bash
cd ~/postgres_workspace/imdb_data

# 执行 schema 创建
$PG_HOME/bin/psql -d imdb -f create_schema.sql

# 导入数据（使用 COPY 命令）
for file in *.csv; do
    table_name=$(basename $file .csv)
    echo "Loading $table_name..."
    $PG_HOME/bin/psql -d imdb -c "\COPY $table_name FROM '$file' WITH (FORMAT csv, HEADER true, DELIMITER ',')"
done
```

### 2.4 创建索引和外键

创建文件 `create_indexes.sql`：

```sql
-- 主键索引（已通过 PRIMARY KEY 自动创建）

-- 外键索引（加速 JOIN）
CREATE INDEX idx_aka_name_person ON aka_name(person_id);
CREATE INDEX idx_aka_title_movie ON aka_title(movie_id);
CREATE INDEX idx_cast_info_person ON cast_info(person_id);
CREATE INDEX idx_cast_info_movie ON cast_info(movie_id);
CREATE INDEX idx_cast_info_role ON cast_info(role_id);
CREATE INDEX idx_movie_companies_movie ON movie_companies(movie_id);
CREATE INDEX idx_movie_companies_company ON movie_companies(company_id);
CREATE INDEX idx_movie_info_movie ON movie_info(movie_id);
CREATE INDEX idx_movie_info_type ON movie_info(info_type_id);
CREATE INDEX idx_movie_info_idx_movie ON movie_info_idx(movie_id);
CREATE INDEX idx_movie_keyword_movie ON movie_keyword(movie_id);
CREATE INDEX idx_movie_keyword_keyword ON movie_keyword(keyword_id);
CREATE INDEX idx_movie_link_movie ON movie_link(movie_id);
CREATE INDEX idx_person_info_person ON person_info(person_id);
CREATE INDEX idx_title_kind ON title(kind_id);

-- 常用查询列索引
CREATE INDEX idx_title_production_year ON title(production_year);
CREATE INDEX idx_movie_info_info ON movie_info(info);
CREATE INDEX idx_keyword_keyword ON keyword(keyword);
```

执行索引创建：
```bash
$PG_HOME/bin/psql -d imdb -f create_indexes.sql
```

### 2.5 收集统计信息

```bash
# 收集所有表的统计信息
$PG_HOME/bin/psql -d imdb -c "ANALYZE VERBOSE;"

# 验证数据导入
$PG_HOME/bin/psql -d imdb -c "
SELECT
    schemaname,
    tablename,
    n_live_tup as row_count
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
"
```

---

## 第三部分：准备实验代码

### 3.1 安装 Python 依赖

```bash
cd ~/postgres_workspace
python3 -m venv venv
source venv/bin/activate

pip install psycopg2-binary numpy pandas
```

### 3.2 创建 PostgreSQL 实验运行器

创建文件 `run_postgres_experiment.py`（见下一个文件）

---

## 第四部分：运行实验

### 4.1 准备 JOB 查询

```bash
# 下载 JOB 查询
cd ~/postgres_workspace
git clone https://github.com/gregrahn/join-order-benchmark.git
cd join-order-benchmark

# 查询文件在 queries/ 目录下
ls queries/*.sql | wc -l  # 应该有 113 个查询
```

### 4.2 运行 Baseline 实验

```bash
cd ~/postgres_workspace
source venv/bin/activate

# Baseline: 无漂移，fresh statistics
python run_postgres_experiment.py \
    --db-name imdb \
    --query-dir join-order-benchmark/queries \
    --strategy baseline \
    --output-dir results/baseline
```

### 4.3 注入数据漂移

创建漂移注入脚本 `inject_drift_postgres.py`（见后续文件）

```bash
# 注入 15 轮漂移（每轮 2%）
python inject_drift_postgres.py \
    --db-name imdb \
    --drift-rounds 15 \
    --drift-ratio 0.02
```

### 4.4 运行 Stale Prior 实验

```bash
# Stale Prior: 有漂移，不更新统计信息
python run_postgres_experiment.py \
    --db-name imdb \
    --query-dir join-order-benchmark/queries \
    --strategy stale_prior \
    --output-dir results/stale_prior
```

### 4.5 运行 Full ANALYZE 实验

```bash
# Full ANALYZE: 有漂移后重新收集统计信息
$PG_HOME/bin/psql -d imdb -c "ANALYZE VERBOSE;"

python run_postgres_experiment.py \
    --db-name imdb \
    --query-dir join-order-benchmark/queries \
    --strategy full_analyze \
    --output-dir results/full_analyze
```

### 4.6 分析结果

创建结果分析脚本 `analyze_postgres_results.py`（见后续文件）

```bash
python analyze_postgres_results.py \
    --baseline results/baseline/baseline_results.json \
    --stale results/stale_prior/stale_prior_results.json \
    --full-analyze results/full_analyze/full_analyze_results.json \
    --output results/comparison.json
```

---

## 第五部分：修改 PostgreSQL 源码（可选）

### 5.1 源码位置

```bash
cd ~/postgres_workspace/postgresql-16.2

# 主要目录结构
# src/backend/optimizer/     - 查询优化器
# src/backend/utils/adt/     - 统计信息处理
# src/include/optimizer/     - 优化器头文件
```

### 5.2 重新编译

```bash
cd ~/postgres_workspace/postgresql-16.2

# 修改代码后重新编译
make -j8
make install

# 重启数据库
$PG_HOME/bin/pg_ctl -D $PG_DATA restart
```

---

## 附录：常用命令

### 数据库管理

```bash
# 启动数据库
$PG_HOME/bin/pg_ctl -D $PG_DATA start

# 停止数据库
$PG_HOME/bin/pg_ctl -D $PG_DATA stop

# 重启数据库
$PG_HOME/bin/pg_ctl -D $PG_DATA restart

# 查看状态
$PG_HOME/bin/pg_ctl -D $PG_DATA status

# 连接数据库
$PG_HOME/bin/psql -d imdb
```

### 统计信息管理

```sql
-- 收集所有表统计信息
ANALYZE VERBOSE;

-- 收集单表统计信息
ANALYZE table_name;

-- 查看统计信息
SELECT * FROM pg_stats WHERE tablename = 'title';

-- 查看直方图
SELECT
    attname,
    n_distinct,
    array_length(most_common_vals::text::text[], 1) as mcv_count,
    array_length(histogram_bounds::text::text[], 1) as histogram_buckets
FROM pg_stats
WHERE tablename = 'title';
```

### 查询分析

```sql
-- 查看执行计划
EXPLAIN SELECT * FROM title WHERE production_year = 2000;

-- 查看执行计划 + 实际执行
EXPLAIN ANALYZE SELECT * FROM title WHERE production_year = 2000;

-- 查看详细执行计划
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT * FROM title WHERE production_year = 2000;
```

---

## 故障排查

### 问题 1：编译失败

```bash
# 检查依赖
sudo apt-get install build-essential libreadline-dev zlib1g-dev

# 如果没有 sudo 权限，联系管理员安装
```

### 问题 2：数据库启动失败

```bash
# 查看日志
tail -f $PG_DATA/logfile

# 检查端口占用
netstat -tuln | grep 5432

# 修改端口（如果 5432 被占用）
echo "port = 5433" >> $PG_DATA/postgresql.conf
```

### 问题 3：内存不足

```bash
# 降低 shared_buffers
$PG_HOME/bin/psql -d imdb -c "ALTER SYSTEM SET shared_buffers = '512MB';"
$PG_HOME/bin/pg_ctl -D $PG_DATA restart
```

---

## 预期实验结果

根据 Presto 实验结果，PostgreSQL 预期会有更显著的改进，因为：

1. PostgreSQL 的优化器更依赖直方图统计信息
2. PostgreSQL 对 Filter 和 Join 算子都使用直方图进行选择性估计
3. PostgreSQL 的 cost model 更精细

预期指标：
- Q-Error 降低：> 20%（vs Presto 的 15.2%）
- 执行时间降低：> 15%（vs Presto 的 10.8%）
- 受益查询比例：> 20%（vs Presto 的 16.3%）
