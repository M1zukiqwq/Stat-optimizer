# TPC-DS Benchmark 实验手册（MySQL 版本）

## 实验目标
在 MySQL 8.0.31+ 上使用 TPC-DS 数据集测试 SCD Type 2 维度增长带来的数据漂移，对比三种策略：

1. **Stale Prior**：注入维度/事实漂移后，不刷新统计信息直接运行查询。
2. **Histogram Only**：仅刷新 MySQL 单列直方图（`ANALYZE TABLE ... UPDATE HISTOGRAM`）。
3. **Full ANALYZE**：执行 `ANALYZE TABLE` 并刷新直方图，再运行查询。

这个目录是对 `tpcds_experiment/` 的 MySQL 对应实现，单独放在：`tpcds_experiment_mysql/`。

## 前提说明
- 推荐统一使用 `tpcds-kit` 仓库：`https://github.com/M1zukiqwq/tpcds-kit.git`。
- 需要 **MySQL 8.0.31+**：这样 `EXPLAIN ANALYZE`、`INTERSECT`、`EXCEPT` 都更稳定可用。
- Python 侧依赖是 `PyMySQL`，见 `tpcds_experiment_mysql/requirements.txt`。
- 默认 MySQL 密码按你的要求设成了 `tianqichu123`；如需覆盖，运行前设置 `MYSQL_PASSWORD`。

## 环境变量
```bash
export MYSQL_HOST=localhost
export MYSQL_PORT=3306
export MYSQL_USER=root
export MYSQL_PASSWORD=tianqichu123
export MYSQL_DB=tpcds
```

## Step 1：准备 TPC-DS 数据与查询
推荐直接使用本目录里的资产准备脚本：

```bash
cd tpcds_experiment_mysql
./tools/prepare_tpcds_assets_mysql.sh --kit-dir /path/to/tpcds-kit --scale 5 --fixed-query-dir ./generated_queries_mysql_sf5 --raw-query-dir ./generated_queries_mysql_raw_sf5
```

它会完成：
- 编译 `dsdgen` / `dsqgen`
- 生成 `.dat` 数据文件
- 生成原始 `dsqgen` 查询
- 修复为 MySQL 可执行版本，输出到 `generated_queries_mysql_sf5`

## Step 2：准备 MySQL 版 schema 并导入数据
这个仓库没有内置完整 MySQL 版 TPC-DS schema。最稳妥的做法是准备一份你自己的 MySQL schema，例如：

```bash
mysql -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" -e "CREATE DATABASE IF NOT EXISTS $MYSQL_DB"
mysql -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DB" < /path/to/tpcds_schema_mysql.sql
```

导入 `.dat` 时，推荐使用 `LOAD DATA LOCAL INFILE`，并先去掉每行尾部额外的 `|`：

```bash
TPCDS_DATA_DIR=/path/to/generated_data_sf5

for table in "$TPCDS_DATA_DIR"/*.dat; do
  name="$(basename "$table" .dat)"
  echo "Loading $name"
  sed 's/|$//' "$table" > "/tmp/${name}.clean"
  mysql --local-infile=1 -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DB" \
    -e "LOAD DATA LOCAL INFILE '/tmp/${name}.clean' INTO TABLE ${name} FIELDS TERMINATED BY '|' LINES TERMINATED BY '\n';"
done
```

## Step 3：注入 MySQL 版 SCD2 漂移
```bash
python3 drift/inject_scd2_drift_mysql.py \
  --dbname "$MYSQL_DB" \
  --user "$MYSQL_USER" \
  --password "$MYSQL_PASSWORD" \
  --host "$MYSQL_HOST" \
  --port "$MYSQL_PORT" \
  --rounds 10 \
  --drift-ratio 0.05
```

当前 MySQL 版漂移脚本会：
- 对 `item` 执行标准 SCD2：关闭旧版本、插入新版本
- 对 `customer` 自动补齐 `c_rec_start_date` / `c_rec_end_date` 两列，再执行同样的 SCD2 关闭/插入逻辑
- 如果库里没有 `c_last_review_date_sk`、只有 `c_last_review_date`，脚本会自动补列并根据 `date_dim` 回填
- 从固定 donor `store_sales` 快照派生新的事实，避免新生成数据继续反哺下一轮

## Step 4：运行三阶段实验
```bash
./run_tpcds_histogram_experiment_mysql.sh
```

默认行为：
- 查询目录：默认优先用 `generated_queries_mysql_sf5/`，不存在时再回退到 `generated_queries_mysql/`
- 漂移状态：只通过 `tpcds_experiment_state_mysql` 判断是否已注入
- 输出目录：`results/tpcds_histogram_comparison_mysql_sf5/`
- 回归统计：`histogram_only_vs_stale_summary.json`、`full_analyze_vs_stale_summary.json`
- `plan diff`：`results/tpcds_histogram_comparison_mysql_sf5/plan_diffs/<strategy>/`

三阶段定义：
1. `stale_prior`
2. `histogram_only`：执行 `ANALYZE TABLE ... UPDATE HISTOGRAM`
3. `full_analyze`：执行 `ANALYZE TABLE`，再更新直方图

## Step 5：查看结果
```bash
python3 -c "
import json, math
for name in ['stale_prior', 'histogram_only', 'full_analyze']:
    path = f'results/tpcds_histogram_comparison_mysql_sf5/{name}_results.json'
    with open(path) as f:
        data = json.load(f)
    ok = [r for r in data['results'] if r['status'] == 'success']
    total = sum(r['execution_time_ms'] for r in data['results'] if r['status'] in ('success', 'timeout') and r['execution_time_ms']) / 1000
    qerrs = [r['qerror']['geometric_mean'] for r in ok if r.get('qerror')]
    geom = math.exp(sum(math.log(x) for x in qerrs) / len(qerrs)) if qerrs else float('nan')
    print(name, 'time_s=', round(total, 2), 'geom_qerror=', round(geom, 3) if qerrs else 'N/A', 'ok=', len(ok), '/', len(data['results']))
"
```

## 注意事项
- 如果你的 MySQL 版本低于 8.0.31，`INTERSECT` / `EXCEPT` 兼容性可能不够，建议升级。
- `Histogram Only` 在 MySQL 里是用列直方图来近似 PostgreSQL 的 `analyze_only_histogram` 实验阶段，语义不是完全等价，但实验目的接近。
- 运行器基于 `EXPLAIN ANALYZE` 的文本输出提取 Q-error，是 best-effort 解析，不像 PostgreSQL `FORMAT JSON` 那样稳定。
