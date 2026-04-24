# JOB Benchmark 实验手册
## 实验目标
对比三种策略在 PostgreSQL 上的查询性能（总执行时间 + Q-Error）：
1. **Stale Prior**：统计信息过期（注入 30% 数据漂移后，不做任何 ANALYZE）
2. **OASIS**：`SET analyze_only_histogram = on` 后做 ANALYZE（只更新直方图，不更新基本计数统计）
3. **Full ANALYZE**：`SET analyze_only_histogram = off` 后做完整 ANALYZE

所有步骤在 `postgres-cdf-simulation/job_experiment/` 目录下执行。

---

## 环境变量
```bash
export PG_PORT=5433
export PG_USER=postgres
export PGPASSWORD=""   # 如有密码请填在这里
```

---

## Step 1：删除旧数据库，重新建库建表

```bash
# 删除旧库
psql -h localhost -p 5433 -U postgres -d postgres -c "DROP DATABASE IF EXISTS imdb;"

# 新建库 + 建表
psql -h localhost -p 5433 -U postgres -d postgres -c "CREATE DATABASE imdb;"
psql -h localhost -p 5433 -U postgres -d imdb -f setup/create_tables_pg.sql
```

---

## Step 2：导入 IMDB 原始数据

使用 Python 脚本（比 `\COPY` 更健壮，自动处理 CSV 格式问题）：

```bash
python3 setup/load_with_python.py
```

**注意**：数据源路径已在脚本中写死为 `setup/imdb_data/`，数据约 7000 万行，预计耗时 **30~60 分钟**。
加载完成后脚本会自动对所有表执行 `ANALYZE`，输出每张表的行数。

---

## Step 3：验证数据完整性

```bash
psql -h localhost -p 5433 -U postgres -d imdb -c "
SELECT relname AS table_name, n_live_tup AS row_count
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_live_tup DESC;"
```

**期望结果**（大致行数）：

| 表名 | 预期行数 |
|---|---|
| cast_info | ~36,000,000 |
| movie_info | ~14,800,000 |
| movie_keyword | ~4,500,000 |
| name | ~4,100,000 |
| title | ~2,500,000 |

---

## Step 4：注入 30% 数据漂移

使用专用漂移注入脚本（15 轮 × 2% = 共 30% 漂移）。

漂移策略：
- **title 表**：DELETE 2000~2012 年的新电影；INSERT 1920~1950 年的老电影（分布反转）
- **cast_info 表**：DELETE 主要角色记录；INSERT 边角角色记录
- **movie_info 表**：DELETE 数值型信息；INSERT 老电影信息

```bash
python3 drift/inject_drift_pg.py \
  --dbname imdb \
  --user postgres \
  --host localhost \
  --port 5433 \
  --rounds 15 \
  --drift-ratio 0.02
```

预计耗时：**10~20 分钟**。

---

## Step 5：确认统计信息已过期（可选验证）

```bash
psql -h localhost -p 5433 -U postgres -d imdb -c "
SELECT relname, n_live_tup, n_dead_tup, last_analyze
FROM pg_stat_user_tables
WHERE relname IN ('title','cast_info','movie_info');"
```

此时 `last_analyze` 应该比 `n_live_tup` 变化前更旧，即统计信息**过期**。

---

## Step 6：运行完整对比实验

```bash
PG_PORT=5433 PG_USER=postgres ./run_histogram_experiment.sh
```

脚本会自动执行三个阶段：
1. **Stale Prior**：直接运行 111 个 JOB 查询（统计已过期）
2. **OASIS**：`SET analyze_only_histogram = on; ANALYZE <table>;` 后再运行 111 个查询
3. **Full ANALYZE**：`SET analyze_only_histogram = off; ANALYZE <table>;` 后再运行 111 个查询

每个查询有 **30 秒**超时限制，整体预计耗时 **20~30 分钟**。

结果保存在：`./results/histogram_comparison/`

---

## Step 7：查看实验结果

```bash
# 直接查看终端最后输出的汇总表格，或手动解析：
python3 -c "
import json, math

def calc(strat):
    try:
        with open(f'results/histogram_comparison/{strat}_results.json') as f:
            data = json.load(f)
            total = sum(r['execution_time_ms'] for r in data['results'] if r['status'] == 'success')
            qerrors = [r['qerror']['geometric_mean'] for r in data['results'] 
                       if r['status'] == 'success' and r.get('qerror')]
            geom = math.exp(sum(math.log(q) for q in qerrors)/len(qerrors)) if qerrors else -1
            ok = sum(1 for r in data['results'] if r['status']=='success')
            return total/1000, geom, ok, len(data['results'])
    except Exception as e:
        return -1, -1, 0, 0

print('+----------------+----------+---------+-----------+')
print('| Strategy       | Time (s) | Q-Error | Succeeded |')
print('+----------------+----------+---------+-----------+')
for label, strat in [('Stale Prior','stale_prior'),('OASIS','histogram_only'),('Full ANALYZE','full_analyze')]:
    t, q, ok, total = calc(strat)
    t_s = f'{t:.1f}' if t >= 0 else 'N/A'
    q_s = f'{q:.2f}' if q >= 0 else 'N/A'
    print(f'| {label:<14} | {t_s:>8} | {q_s:>7} | {ok}/{total:<7} |')
print('+----------------+----------+---------+-----------+')
"
```

---

## 常见问题

### Q：Step 2 数据加载很慢，可以跳过吗？
A：不可以，必须从干净的原始数据开始，否则漂移统计不准确。如果之前已经加载过并且只是想重跑实验，可跳过 Step 1-2，直接从 Step 3 验证开始。

### Q：histogram_only 为什么上次没有结果文件？
A：因为最初的 `run_simple_pg_experiment.py` 的 `--strategy` 参数不支持 `histogram_only` 这个值（只支持 `stale_prior` 和 `full_analyze`），导致 Python 脚本调用失败，但 `|| true` 吞掉了错误。现已在脚本中修复，加入了 `histogram_only` 选项。

### Q：SET analyze_only_histogram 有效吗？
这是我们自己在 PG 14 源码中添加的 GUC 参数。确认已编译并生效：
```bash
psql -h localhost -p 5433 -U postgres -d imdb -c "SHOW analyze_only_histogram;"
```
如果报错 `unrecognized configuration parameter`，说明当前运行的 PG 实例不是我们修改过的版本，需要先用修改过的二进制文件启动。
