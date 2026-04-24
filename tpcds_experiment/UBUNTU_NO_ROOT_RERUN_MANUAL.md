# Ubuntu 无 root 权限重跑 TPC-DS 实验手册

这份手册假设你已经把 PostgreSQL/`psql` 编译好，并且手头有可用的 PostgreSQL 安装目录，例如：

```bash
export PG_HOME="$HOME/pg14"
export PATH="$PG_HOME/bin:$PATH"
```

如果你是从当前仓库源码编出来的，`PG_HOME` 指向你的 `make install` 目标目录即可。

## 0. 前置假设
- 推荐统一使用 `tpcds-kit` 仓库：`https://github.com/M1zukiqwq/tpcds-kit.git`。

- 系统已经有 `gcc`、`make`、`git`、`python3`。
- 机器没有 root 权限，但允许在 `$HOME` 下编译和运行程序。
- 当前仓库位于：`$HOME/postgres-cdf-simulation`。
- 下面命令以 `2026-03-12` 为基准整理；如路径不同，请按实际目录替换。

## 1. 启动你自己的 PostgreSQL 实例

```bash
export PGDATA="$HOME/pgdata-tpcds"
export PGPORT=5433
export PGUSER="$USER"

initdb -D "$PGDATA"
pg_ctl -D "$PGDATA" -l "$PGDATA/server.log" -o "-p $PGPORT" start
createdb -p "$PGPORT" tpcds
```

验证：

```bash
psql -p "$PGPORT" -d tpcds -c 'select version();'
```

## 2. 下载并编译 TPC-DS kit

这里统一使用 `M1zukiqwq/tpcds-kit`：

```bash
mkdir -p "$HOME/work"
cd "$HOME/work"
git clone https://github.com/M1zukiqwq/tpcds-kit.git
cd tpcds-kit/tools
make OS=LINUX
```

编译完成后，确认这两个工具存在：

```bash
ls -l dsdgen dsqgen
```

如果 `make OS=LINUX` 因为本机缺少词法/语法工具失败，只能先确保机器上已有这些构建依赖，或者自己在 `$HOME` 下准备一套可用工具链后再继续。

## 3. 生成更大的 TPC-DS 数据

下面默认以 `SF=5` 为例；如果机器磁盘和时间允许，可以改成 `SF=10`、`SF=30` 或 `SF=100`。

```bash
export TPCDS_ROOT="$HOME/work/tpcds-kit"
export TPCDS_SF=5
export TPCDS_DATA_DIR="$HOME/tpcds_sf${TPCDS_SF}_data"

mkdir -p "$TPCDS_DATA_DIR"
cd "$TPCDS_ROOT/tools"
./dsdgen -scale "$TPCDS_SF" -dir "$TPCDS_DATA_DIR" -force
```

生成后你会看到一批 `.dat` 文件，例如 `store_sales.dat`、`item.dat`、`customer.dat`。

## 4. 准备查询 SQL

当前仓库的实验脚本只要求一个装满 `.sql` 文件的目录。最稳妥的做法是直接复用你已经验证过、能在 PostgreSQL 上运行的 TPC-DS 99 条查询。

如果你只是想先把官方模板物化成独立 `.sql` 文件，可以参考 `tpcds-kit` README 里的 `dsqgen` 用法：

```bash
export REPO_ROOT="$HOME/postgres-cdf-simulation"
export QUERY_DIR="$REPO_ROOT/tpcds_experiment/generated_queries_pg_sf5"

mkdir -p "$QUERY_DIR"
cd "$TPCDS_ROOT/tools"

./dsqgen \
  -DIRECTORY ../query_templates \
  -INPUT ../query_templates/templates.lst \
  -VERBOSE Y \
  -QUALIFY Y \
  -SCALE "$TPCDS_SF" \
  -DIALECT netezza \
  -OUTPUT_DIR "$QUERY_DIR"
```

检查一下：

```bash
find "$QUERY_DIR" -name '*.sql' | wc -l
```

上面这条是上游 README 给出的模板展开方式；`DIALECT netezza` 只是它文档里的示例，不保证生成结果能直接在 PostgreSQL 上执行。

因此，真正跑实验时建议二选一：

- 优先使用已经修复成 PostgreSQL 可执行版本的查询集，并把 `.sql` 文件放进 `tpcds_experiment/generated_queries_pg_sf5/`。
- 如果你确实从 `dsqgen` 原始产物起步，先用仓库里的 `tpcds_experiment/tools/fix_tpcds_queries_for_pg.py` 做一次修复，再抽样用 `psql -f` 验证语法。

如果你已经有本地 `tpcds-kit`，也可以直接使用：

```bash
cd "$REPO_ROOT"
./tpcds_experiment/tools/prepare_tpcds_assets.sh --kit-dir /path/to/tpcds-kit --scale "$TPCDS_SF" --raw-query-dir "$REPO_ROOT/tpcds_experiment/generated_queries_pg_raw_sf5" --fixed-query-dir "$REPO_ROOT/tpcds_experiment/generated_queries_pg_sf5"
```

它会同时生成 `.dat`、原始查询和修复后的 PostgreSQL 查询；上面这条命令里已经显式把修复后的查询目录固定到 `generated_queries_pg_sf5/`。

## 5. 建表

这个仓库没有内置完整 TPC-DS DDL，因此最稳妥的做法是准备一份你自己的 PostgreSQL TPC-DS schema 文件，例如：

```bash
export TPCDS_SCHEMA_SQL="$HOME/work/tpcds_schema_postgres.sql"
psql -p "$PGPORT" -d tpcds -f "$TPCDS_SCHEMA_SQL"
```

导入前先确认关键表存在：

```bash
psql -p "$PGPORT" -d tpcds -c '\dt item'
psql -p "$PGPORT" -d tpcds -c '\dt customer'
psql -p "$PGPORT" -d tpcds -c '\dt store_sales'
psql -p "$PGPORT" -d tpcds -c '\dt date_dim'
```

## 6. 导入数据

TPC-DS `.dat` 默认用 `|` 分隔，而且每行末尾还有一个额外的 `|`。导入前先把行尾这个分隔符去掉，再喂给 `\copy` 会更稳：

```bash
cd "$TPCDS_DATA_DIR"

for table in *.dat; do
  name="${table%.dat}"
  echo "Loading $name"
  sed 's/|$//' "$TPCDS_DATA_DIR/$table" | \
    psql -p "$PGPORT" -d tpcds -c "\\copy $name from stdin with (format csv, delimiter '|', null '')"
done
```

导完后建议先跑一次统计：

```bash
psql -p "$PGPORT" -d tpcds -c 'analyze;'
```

## 7. 准备实验环境变量

```bash
export REPO_ROOT="$HOME/postgres-cdf-simulation"
export PG_HOST=localhost
export PG_PORT="$PGPORT"
export PG_USER="$USER"
export PG_DB=tpcds
export PGPASSWORD=''
export QUERY_DIR="$REPO_ROOT/tpcds_experiment/generated_queries_pg_sf5"
```

如果你是自己编的 PostgreSQL，最好确认实验用的就是你刚编好的 `psql`：

```bash
which psql
psql --version
```

## 8. 注入修复后的 SCD2 漂移

下面示例仍用 `10` 轮、每轮 `5%`。如果 `SF` 变大，通常可以保持这个比例不变。

```bash
cd "$REPO_ROOT"

python3 tpcds_experiment/drift/inject_scd2_drift_pg.py \
  --dbname "$PG_DB" \
  --user "$PG_USER" \
  --password "$PGPASSWORD" \
  --host "$PG_HOST" \
  --port "$PG_PORT" \
  --rounds 10 \
  --drift-ratio 0.05 \
  --item-facts-per-key 10 \
  --customer-facts-per-key 5 \
  --date-shift-days 30
```

修复后的逻辑和旧版相比有四个关键差异：

- 只从一份固定 donor `store_sales` 快照派生新事实，避免“新生成的数据继续喂给下一轮”。
- `customer` 会自动补齐 `c_rec_start_date` / `c_rec_end_date`，并像 `item` 一样显式关闭旧版本后再插入新版本，不再是 append-only。
- 如果你的 `customer` 只有 `c_last_review_date`、没有 `c_last_review_date_sk`，脚本会自动补齐并根据 `date_dim` 回填。
- 每个新 surrogate key 都从对应 donor key 派生固定数量事实，更接近可控的 SCD2/fact-growth 注入。
- 实验执行脚本会把漂移注入状态持久化到 `tpcds_experiment_state`，不再通过 `item.i_rec_end_date IS NOT NULL` 这类会被 TPC-DS 基线历史版本误伤的条件来判断。

## 9. 运行三阶段实验

```bash
cd "$REPO_ROOT/tpcds_experiment"

PG_HOST="$PG_HOST" \
PG_PORT="$PG_PORT" \
PG_USER="$PG_USER" \
PG_DB="$PG_DB" \
PG_PASSWORD="$PGPASSWORD" \
QUERY_DIR="$QUERY_DIR" \
./run_tpcds_histogram_experiment.sh
```

脚本会依次运行：

- `stale_prior`
- `histogram_only`
- `full_analyze`

并把结果写到：

```bash
$REPO_ROOT/tpcds_experiment/results/tpcds_histogram_comparison_sf5/
```

同时还会额外产出：
- `results/tpcds_histogram_comparison_sf5/histogram_only_vs_stale_summary.json`
- `results/tpcds_histogram_comparison_sf5/full_analyze_vs_stale_summary.json`
- 对于比 `stale_prior` 慢 `20%` 以上的查询，`plan diff` 在 `results/tpcds_histogram_comparison_sf5/plan_diffs/<strategy>/`

## 10. 汇总结果

```bash
cd "$REPO_ROOT/tpcds_experiment"

python3 -c "
import json, math
for name in ['stale_prior', 'histogram_only', 'full_analyze']:
    path = f'results/tpcds_histogram_comparison_sf5/{name}_results.json'
    with open(path) as f:
        data = json.load(f)
    ok = [r for r in data['results'] if r['status'] == 'success']
    total = sum(r['execution_time_ms'] for r in ok) / 1000
    qerrs = [r['qerror']['geometric_mean'] for r in ok if r.get('qerror')]
    geom = math.exp(sum(math.log(x) for x in qerrs) / len(qerrs)) if qerrs else float('nan')
    print(name, 'time_s=', round(total, 2), 'geom_qerror=', round(geom, 3), 'ok=', len(ok), '/', len(data['results']))
"
```

## 11. 推荐的 SF 选择

- `SF=5`：现在的默认规模，适合先验证整套流程，也能比较稳定地放大统计漂移效应。
- `SF=10`：比默认规模更重，适合进一步验证结论稳定性。
- `SF=30`：如果机器内存和磁盘还可以，这是比较均衡的正式实验点。
- `SF=100`：更接近重负载实验，但导入和查询时间会明显增加。

如果你是第一次在无 root 机器上重跑，我建议先用默认的 `SF=5` 打通，再升到 `SF=10` 或 `SF=30`。

## 12. 常见问题

- `Query directory not found`：说明没有把 `.sql` 查询放到 `tpcds_experiment/generated_queries_pg_sf5/`，或者没设置 `QUERY_DIR`。
- `connection to server failed`：确认 `pg_ctl` 启动的是你自己的实例，而且端口和 `PG_PORT` 一致。
- `permission denied`：确认所有数据目录都在 `$HOME` 下，并且脚本对这些目录有写权限。
- `dsqgen` 生成的 SQL 不兼容：换成你已经验证过的 PostgreSQL 版本查询集，再通过 `QUERY_DIR` 指向它。
