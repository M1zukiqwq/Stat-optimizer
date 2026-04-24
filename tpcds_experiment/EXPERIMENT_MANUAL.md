# TPC-DS Benchmark 实验手册 (SCD Type 2 Dimension Growth)

## 实验目标
在 PostgreSQL 上使用 TPC-DS 数据集测试缓慢维度变化（SCD Type 2）引起的数据漂移，对比三种策略的查询性能（总执行时间 + Q-Error）：
1. **Stale Prior**：维度表经历 SCD2 增长（产生大量全新的 Surrogate Keys），但不做任何 ANALYZE（统计信息过期）
2. **OASIS**：`SET analyze_only_histogram = on` 后做 ANALYZE（只更新直方图边界，不更新 `reltuples`/`nullfrac` 等基础计数统计）
3. **Full ANALYZE**：`SET analyze_only_histogram = off` 后做完整 ANALYZE

本实验用于验证：在 **SCD Type 2 维度增长 + 同步事实表增长** 造成的 Out-of-Bounds surrogate-key 漂移下，仅刷新 1D 直方图边界也能恢复相当一部分优化器收益，同时比 synthetic 单列漂移更贴近真实的 schema-level 关系结构。

所有步骤在 `postgres-cdf-simulation/tpcds_experiment/` 目录下执行。

---

## 环境变量
> 推荐使用的 `tpcds-kit` 仓库：`https://github.com/M1zukiqwq/tpcds-kit.git`

```bash
export PG_PORT=5433
export PG_USER=postgres
export PGPASSWORD=""   # 如有密码请填在这里
```

---

## Step 1：准备 TPC-DS 数据

如果你的 PostgreSQL 实例（端口 5433）中尚未加载完整的 TPC-DS 1GB 数据集：

1. 克隆并编译 [TPC-DS 工具](https://github.com/M1zukiqwq/tpcds-kit.git) 或使用现成的 `dsdgen`。
2. 生成数据 (`./dsdgen -SCALE 1 -DIR /tmp/tpcds_data`)。
3. 创建名为 `tpcds` 的数据库，并执行 DDL 创建所有的变体表。
4. 使用 `\COPY` 导入数据到 `tpcds` 数据库中。

*如果已经准备好了 `tpcds` 数据库及数据，可直接跳过此步骤。*

---

## Step 2：注入 SCD Type 2 数据漂移

使用专用注入脚本，对最大的两张维度表 `item` 和 `customer` 模拟 SCD Type 2 更新。同时会从一份固定的 `store_sales` donor 快照中派生新事实行，分别关联到新生成的 item/customer surrogate keys，避免“合成数据继续自我复制”的失真。脚本现在会在 `customer` 表上自动补齐 `c_rec_start_date` / `c_rec_end_date` 两个生命周期列，并像 `item` 一样先关闭旧版本、再插入新版本。同时兼容两种 customer schema：如果库里没有 `c_last_review_date_sk`、只有 `c_last_review_date`，脚本会自动补列并根据 `date_dim` 回填。

当前实验脚本会把“是否已注入漂移”的状态写入数据库中的 `tpcds_experiment_state` 表，而不再用 `item.i_rec_end_date IS NOT NULL` 这类基线自带历史版本会误判的条件来判断。

```bash
mkdir -p tpcds_experiment/drift
python3 tpcds_experiment/drift/inject_scd2_drift_pg.py \
  --dbname tpcds \
  --user postgres \
  --password "$PGPASSWORD" \
  --host localhost \
  --port 5433 \
  --rounds 10 \
  --drift-ratio 0.03
```

预估耗时：约 2-5 分钟。

---

## Step 3：运行完整对比实验

运行实验执行脚本。该脚本会自动加载 99 个 TPC-DS 查询在这三个阶段下依次运行：

```bash
PG_PORT=5433 PG_USER=postgres ./run_tpcds_histogram_experiment.sh
```

脚本自动执行的内容：
1. **Stale Prior**：不做 ANALYZE，直接跑 TPC-DS。
2. **OASIS**：执行 `SET analyze_only_histogram = on; ANALYZE item, customer, store_sales;` 后跑 TPC-DS。
3. **Full ANALYZE**：执行 `SET analyze_only_histogram = off; ANALYZE item, customer, store_sales;` 后跑 TPC-DS。

注意：当前脚本默认按 `TPCDS_SF=5` 运行，优先读取 `tpcds_experiment/generated_queries_pg_sf5/`；如果该目录不存在，才会回退到 `tpcds_experiment/generated_queries_pg/` 和 `tpcds_experiment/generated_queries/`。如果你把查询生成到别处，请在运行前设置 `QUERY_DIR=/path/to/sql_dir`。

结果汇总在终端输出，并且详情默认保存在：`./results/tpcds_histogram_comparison_sf5/`。

另外，脚本会在跑完 `histogram_only` 和 `full_analyze` 后，额外生成相对 `stale_prior` 的回归统计：
- 汇总 JSON：`results/tpcds_histogram_comparison_sf5/histogram_only_vs_stale_summary.json`、`results/tpcds_histogram_comparison_sf5/full_analyze_vs_stale_summary.json`
- 对于比过期统计慢 `20%` 以上的查询，`plan diff` 会写到：`results/tpcds_histogram_comparison_sf5/plan_diffs/<strategy>/`

---

## 查看实验结果

可以直接看终端输出，或是运行以下简易脚本解析结果：

```bash
python3 -c "
import json, math

def calc(strat):
    try:
        with open(f'results/tpcds_histogram_comparison_sf5/{strat}_results.json') as f:
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
