# PostgreSQL 简单端到端实验：Stale Prior vs Full ANALYZE

这是一个简化的 PostgreSQL 端到端实验，用于快速验证 **统计过期对查询性能的影响**。

## 实验目标

对比两种策略在 JOB Benchmark 上的表现：
1. **Stale Prior**: 数据已漂移但统计未更新（模拟生产环境问题）
2. **Full ANALYZE**: 重新 ANALYZE 后运行（理想上界）

## 前置条件

1. **PostgreSQL 14+** 已安装并运行
2. **IMDB 数据已加载**到数据库
3. **Python 依赖**:
   ```bash
   pip3 install psycopg2-binary
   ```

## Q-Error 提取验证

在运行完整实验前，可以先验证 Q-Error 提取逻辑是否正确：

```bash
cd job_experiment/experiment

# 运行测试套件
python3 test_qerror_extraction.py
```

预期输出：
```
============================================================
Overall Test Results:
============================================================
✓ PASSED: Q-Error Calculation
✓ PASSED: EXPLAIN JSON Parsing  
✓ PASSED: Geometric Mean
============================================================
All tests PASSED
```

**Q-Error 计算说明**:
- Q-Error = max(estimated / actual, actual / estimated)
- Q-Error ≥ 1，1 表示完美估算
- 统计所有 Join 和 Scan 算子的估算行数 vs 实际行数

## 快速开始

### 1. 数据准备（如果还没做）

```bash
cd setup

# 下载 IMDB 数据
./1_download_imdb.sh

# 创建数据库
psql -U postgres -c "CREATE DATABASE imdb;"

# 创建表
psql -U postgres -d imdb -f 2_create_tables.sql

# 加载数据（使用 COPY，比 INSERT 快很多）
psql -U postgres -d imdb <<'EOF'
\COPY title FROM 'imdb_data/imdb/title.csv' WITH (FORMAT csv, HEADER false, NULL '');
\COPY cast_info FROM 'imdb_data/imdb/cast_info.csv' WITH (FORMAT csv, HEADER false, NULL '');
\COPY movie_info FROM 'imdb_data/imdb/movie_info.csv' WITH (FORMAT csv, HEADER false, NULL '');
\COPY movie_keyword FROM 'imdb_data/imdb/movie_keyword.csv' WITH (FORMAT csv, HEADER false, NULL '');
\COPY name FROM 'imdb_data/imdb/name.csv' WITH (FORMAT csv, HEADER false, NULL '');
-- 其他表...
EOF

# 初始 ANALYZE
psql -U postgres -d imdb -f 4_initial_analyze.sql
```

### 2. 运行对比实验

```bash
cd job_experiment

# 基本用法（使用默认配置）
./run_stale_vs_analyze.sh

# 或者自定义参数
PG_USER=myuser PG_DB=mydb ./run_stale_vs_analyze.sh
```

**环境变量**:
- `PG_HOST`: PostgreSQL 主机 (默认: localhost)
- `PG_PORT`: PostgreSQL 端口 (默认: 5432)
- `PG_USER`: 用户名 (默认: postgres)
- `PG_DB`: 数据库名 (默认: imdb)
- `PG_PASSWORD`: 密码 (默认: 空)
- `DRIFT_ROUNDS`: 漂移轮数 (默认: 15)
- `DRIFT_RATIO`: 每轮漂移比例 (默认: 0.02)
- `TIMEOUT`: 查询超时秒数 (默认: 300)

### 3. 查看结果

实验完成后，结果保存在 `results/simple_comparison/`:

```bash
# 查看 JSON 结果
cat results/simple_comparison/stale_prior_results.json
cat results/simple_comparison/full_analyze_results.json

# 控制台会输出汇总表格
```

预期输出示例：
```
+---------------+---------------+----------+
| Strategy      | Total Time    | Q-Error  |
+---------------+---------------+----------+
| Stale Prior   |       348.5s |     8.63 |
| Full ANALYZE  |       297.1s |     5.64 |
+---------------+---------------+----------+

Improvement: 14.8% faster with Full ANALYZE
Q-Error reduction: 34.6%
```

## 实验原理

### 为什么这个对比重要？

1. **Stale Prior** 代表生产环境的真实问题：
   - 数据持续写入（INSERT/UPDATE/DELETE）
   - ANALYZE 每天/每周才运行一次
   - 统计信息过期导致 CBO 选择错误计划

2. **Full ANALYZE** 代表理想情况：
   - 统计信息完全准确
   - CBO 做出最优选择
   - 但频繁 ANALYZE 在生产环境不可行（全表扫描开销大）

3. **OASIS 的目标**：
   - 无需重新 ANALYZE
   - 通过查询反馈在线修正统计
   - 达到接近 Full ANALYZE 的性能

## 文件说明

| 文件 | 说明 |
|------|------|
| `run_stale_vs_analyze.sh` | 主实验脚本，运行完整对比流程 |
| `experiment/run_simple_pg_experiment.py` | PostgreSQL 实验运行器（收集时间和 Q-error） |
| `results/simple_comparison/` | 实验结果输出目录 |

## 故障排查

### 连接失败
```bash
# 检查 PostgreSQL 运行状态
pg_isready -h localhost -p 5432

# 检查数据库是否存在
psql -U postgres -l | grep imdb
```

### 缺少表
```bash
# 在 psql 中查看所有表
psql -U postgres -d imdb -c "\dt"
```

### 查询超时
某些 JOB 查询（如 33c）在统计过期时可能执行很久。脚本默认跳过这些查询，你可以：
```bash
# 增加超时时间
TIMEOUT=600 ./run_stale_vs_analyze.sh

# 跳过特定查询
# 编辑脚本中的 --skip-queries 参数
```

### Q-error 为 N/A
如果 Q-error 显示为 N/A，可能是：
1. EXPLAIN (ANALYZE, FORMAT JSON) 输出解析失败
2. 查询执行出错

检查原始输出文件中的 `qerror` 字段。

## 下一步

完成基础对比后，你可以：

1. **集成 OASIS 修正**: 在 Stale Prior 和 Full ANALYZE 之间加入 OASIS 模型修正
2. **不同漂移强度**: 测试 q=5, 10, 20, 25, 30 的情况
3. **Ablation Study**: 对比 Teacher、OASIS v1、OASIS v2 的效果

参考 `QUICKSTART.md` 了解完整的 OASIS 实验流程。
