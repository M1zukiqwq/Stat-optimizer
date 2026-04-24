# Ubuntu 无 root 权限重跑 TPC-DS（MySQL 版本）

这份手册对应 `tpcds_experiment_mysql/`，目标是在无 root 环境下，用你自己的 MySQL 8 实例重跑 TPC-DS SCD2/histogram 实验。

## 0. 前置假设
- 推荐统一使用 `tpcds-kit` 仓库：`https://github.com/M1zukiqwq/tpcds-kit.git`。
- 你已经有一个可用的 MySQL 8.0.31+ 实例。
- 机器上有 `python3`、`gcc`、`make`、`mysql` 客户端。
- 你可以用 `pip` 给当前用户安装 Python 依赖。
- 当前仓库位于：`$HOME/postgres-cdf-simulation`。

## 1. 安装 Python 依赖
```bash
python3 -m pip install --user -r "$HOME/postgres-cdf-simulation/tpcds_experiment_mysql/requirements.txt"
```

## 2. 设置 MySQL 环境变量
按你的要求，下面默认密码写成 `tianqichu123`：

```bash
export MYSQL_HOST=localhost
export MYSQL_PORT=3306
export MYSQL_USER=root
export MYSQL_PASSWORD=tianqichu123
export MYSQL_DB=tpcds
```

## 3. 准备 tpcds-kit 并生成资产
```bash
export REPO_ROOT="$HOME/postgres-cdf-simulation"
export TPCDS_ROOT="/path/to/tpcds-kit"
export TPCDS_SF=5

cd "$REPO_ROOT/tpcds_experiment_mysql"
./tools/prepare_tpcds_assets_mysql.sh --kit-dir "$TPCDS_ROOT" --scale "$TPCDS_SF" --fixed-query-dir ./generated_queries_mysql_sf5 --raw-query-dir ./generated_queries_mysql_raw_sf5
```

执行后会生成：
- `generated_queries_mysql_sf5/`：默认给实验脚本使用的 SF=5 修复后 MySQL 查询
- `generated_queries_mysql_raw_sf5/`：对应的 SF=5 原始模板展开结果
- `tpcds-kit` 侧的数据文件目录（默认在 kit 目录下）

## 4. 建库建表
```bash
mysql -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" \
  -e "CREATE DATABASE IF NOT EXISTS $MYSQL_DB"

mysql -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DB" \
  < /path/to/tpcds_schema_mysql.sql
```

## 5. 导入 `.dat`
```bash
TPCDS_DATA_DIR="/path/to/tpcds-kit/generated_data_sf${TPCDS_SF}"

for table in "$TPCDS_DATA_DIR"/*.dat; do
  name="$(basename "$table" .dat)"
  echo "Loading $name"
  sed 's/|$//' "$table" > "/tmp/${name}.clean"
  mysql --local-infile=1 -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" "$MYSQL_DB" \
    -e "LOAD DATA LOCAL INFILE '/tmp/${name}.clean' INTO TABLE ${name} FIELDS TERMINATED BY '|' LINES TERMINATED BY '\n';"
done
```

## 6. 运行 SCD2 漂移注入
```bash
cd "$REPO_ROOT/tpcds_experiment_mysql"
python3 drift/inject_scd2_drift_mysql.py \
  --dbname "$MYSQL_DB" \
  --user "$MYSQL_USER" \
  --password "$MYSQL_PASSWORD" \
  --host "$MYSQL_HOST" \
  --port "$MYSQL_PORT" \
  --rounds 10 \
  --drift-ratio 0.05 \
  --item-facts-per-key 10 \
  --customer-facts-per-key 5 \
  --date-shift-days 30
```

当前 MySQL 版脚本和旧版 append-only customer 逻辑相比有三个关键变化：
- `customer` 会自动补 `c_rec_start_date` / `c_rec_end_date`
- 每轮会先关闭旧 customer 版本，再插入新版本
- 如果你的 `customer` 只有 `c_last_review_date`、没有 `c_last_review_date_sk`，脚本会自动补齐并根据 `date_dim` 回填
- 漂移状态会写入 `tpcds_experiment_state_mysql`，后续重跑只依赖这张状态表判断

## 7. 运行三阶段实验
```bash
cd "$REPO_ROOT/tpcds_experiment_mysql"
./run_tpcds_histogram_experiment_mysql.sh
```

脚本默认读取：
- 查询目录：`generated_queries_mysql_sf5/`（不存在时可手动设 `QUERY_DIR`）
- 结果目录：`results/tpcds_histogram_comparison_mysql_sf5/`
- 回归统计：`histogram_only_vs_stale_summary.json`、`full_analyze_vs_stale_summary.json`
- `plan diff`：`results/tpcds_histogram_comparison_mysql_sf5/plan_diffs/<strategy>/`

## 8. 汇总结果
```bash
cd "$REPO_ROOT/tpcds_experiment_mysql"
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

## 9. 常见问题
- `Query directory not found`：确认 `generated_queries_mysql_sf5/` 已生成，或者手动设置 `QUERY_DIR`。
- `PyMySQL missing`：执行 `pip install --user -r tpcds_experiment_mysql/requirements.txt`。
- `EXPLAIN ANALYZE` 不支持：确认 MySQL 版本至少 8.0.18；若还要跑 `INTERSECT/EXCEPT`，建议 8.0.31+。
- `Histogram` 语句失败：确认实例启用了 MySQL 8 直方图特性，并且表列名没有改动。
