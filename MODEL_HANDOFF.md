# MODEL_HANDOFF

## 1. 项目一句话
OASIS 是一个用查询反馈在线修正单列直方图统计的系统；核心卖点是：**不改优化器、不重扫表、直接输出 CBO 可消费的统计对象**，并通过 **statistics-format conversion interface** 兼容 PostgreSQL 这类 `MCV + histogram` 紧耦合统计布局。

## 2. 当前论文主线
- **主问题**：`ANALYZE` 代价高，统计信息在两次刷新之间持续过期；
- **核心方法**：把统计修正建模成 `stale histogram + feedback window -> corrected quantiles` 的回归问题；
- **模型**：attention-pooled MLP，默认只看最近 `K=16` 条 observation；
- **系统点**：
  - feedback collection piggyback 在已有 predicate evaluation 上；
  - 每个 conjunct 最多写 1 条 observation；
  - stats-format adapter 支持 standalone histogram 与 `MCV + histogram` 两类布局；
- **实验主张**：
  - 主 synthetic suite 上 OASIS 在中高 drift 明显优于传统方法；
  - 对不同初始分布有泛化；
  - 对 PostgreSQL TPC-DS 有端到端 OASIS 收益；
  - `SCD Type 2` / `Fact growth` 应该改在 `TPC-DS` 工作流里单独验证，而不是继续用 synthetic surrogate。

## 3. 当前“官方”结果目录
### 主论文 B=10 套件
- `experiments/results/synthetic_paper_suite_tree_isomer_v3/manifest.json`
- 这是 `paper/main.tex` 当前主实验的基础数据源；
- 关键配置：`B=10`, `K=16`, `stholes_mode=tree`, `train_samples_per_q=1000`。

### TPC-DS 漂移实验
- `SCD Type 2` / `Fact growth` 的后续验证应迁移到 `tpcds_experiment/`；
- synthetic mixed-profile transfer 结果不再作为论文主线或 paper-facing artifact 保留。

## 4. 这次刚确认的关键事实
### A. 论文主文已经移除 synthetic growth-transfer 段落
- `SCD Type 2` / `Fact growth` 不再出现在 `paper/main.tex`；
- 相关 synthetic 代码和 appendix artifact 也已删除，避免与 `TPC-DS` 主线冲突。

### B. QuickSel-H 目前是弱启发式，不是 faithful QuickSel
代码：`cdf_kll_ml_pipeline/modern_baselines.py:22`

当前实现问题/局限：
- 丢弃 `BETWEEN` 和 `=` 反馈；
- 固定均匀 mixture weights；
- 只用单点 CDF 拟合；
- logistic 近似 normal CDF；
- 本质上是 QuickSel-inspired heuristic，不宜在论文里写成“exact adaptation”。

### C. 当前 paper-facing synthetic runner 已对齐到 `K=16`
- OASIS：`tensorize_sample(..., max_observations=16)`，只取最近 16 条；
- Classical baselines：`experiments/run_synthetic_paper_suite.py` 中通过 `obs_to_dicts(..., max_obs=max_obs)` 同样只取最近 16 条；
- 若后续新增别的实验入口，仍要单独检查是否复用了相同的 observation 截断逻辑。

## 5. 论文里最值得小心的口径
### 可以继续坚持的
- OASIS 是 statistics-level correction，不是 plan-level override；
- MCV + histogram adapter 是一个系统创新点；
- 初始分布泛化 + PostgreSQL 端到端收益共同构成当前论文主线。

### 不要再过度表述的
- 不要把 QuickSel-H 叫作 faithful / exact QuickSel；
- 若新增其他实验入口，不要默认它们也已经严格复用了相同的 `K=16` observation window 截断逻辑；
- 不要再把 synthetic `SCD Type 2` / `Fact growth` 当成 paper-facing 主结果；这部分应转到 `TPC-DS` 流程。

## 6. 关键代码入口
### 主 synthetic 套件
- `experiments/run_synthetic_paper_suite.py`

### TPC-DS 漂移套件
- `tpcds_experiment/EXPERIMENT_MANUAL.md`

### 训练与张量化
- `cdf_kll_ml_pipeline/tensorizer.py`
- `cdf_kll_ml_pipeline/mlp_histogram_model_v2.py`
- `cdf_kll_ml_pipeline/train_mlp_model_v2.py`

### baseline
- `cdf_kll_ml_pipeline/baselines.py`
- `cdf_kll_ml_pipeline/modern_baselines.py`

### 论文
- `paper/main.tex`
- `paper/appendix/system_details.tex`
- `paper/appendix/mcv_adapter.tex`

## 7. 如果下一个模型接手，建议优先做什么
1. **若追求实验公平性**：优先检查新增实验入口是否同样复用了 `K=16` observation 截断逻辑，然后再重跑 `main + distribution`；
2. **若追求 baseline 可信度**：重做 QuickSel-H，使其至少支持 interval constraints 和可学习 mixture weights；
3. **若追求投稿版本稳定**：继续压缩篇幅，优先砍实验文字重复而不是再动结果；
4. **若追求系统故事完整**：继续扩展 `TPC-DS` 的真实 `SCD Type 2` / `Fact growth` 结果，但与 synthetic 单列结果分开收束。

## 8. 一条最重要的提醒
当前最容易“接手后误判”的点是：**paper-facing synthetic runner 已经把 OASIS 和 classical baselines 都对齐到最近 `K=16` 条 observation，但别的实验入口未必自动继承了这套约束。** 如果后续模型新增 runner，这是第一优先级检查项。
