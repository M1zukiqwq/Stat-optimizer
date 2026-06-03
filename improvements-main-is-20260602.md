# OASIS 投稿前修改优先级

**Paper:** `paper/main_is.tex`  
**Target venue:** Information Systems  
**Generated:** 2026-06-02

## Top 5 Before Submission

1. **重写 Proposition/rank section。**  
   Location: `paper/main_is.tex:294`, `paper/main_is.tex:309`, `paper/main_is.tex:540`  
   Fix: 用 feedback-induced cell-mass partition 表述：cells 由 stale support / feedback endpoints / projection partition 诱导，feasible set 维度为 `C-1-r`。把 “ISOMER, STHoles, QuickSel-H all return same marginal” 改成 “same hard projection operator with any prior returns same marginal when feasible set is singleton”。  
   Effort: Medium.

2. **补强 PostgreSQL sparse K sweep。**  
   Location: `paper/main_is.tex:1009`, `paper/main_is.tex:1027`  
   Fix: 至少扩展到 12 configurations，或明确标成 representative case study。主文表加 Fresh Plan / Recovery / New Deviations 列，使用已有 generated table 的完整指标格式。清理或隔离旧的 `experiments/results/exp2_sparse_sweep_20260601`。  
   Effort: High.

3. **给 Router 做非 oracle + safety 诊断。**  
   Location: `paper/main_is.tex:514`, `paper/main_is.tex:1198`  
   Fix: 增加 residual distributions、tie rate、choice counts、always-ISOMER/OASIS/random/oracle 对照、plan new-deviation tail。把 “supply optimizer safety” 改成 “empirically reduces observed safety failures under reported workloads”。  
   Effort: Medium.

4. **清理 supplement / artifact 不一致。**  
   Location: `paper/mr_supplement.tex:56`, `paper/supplementary.tex:516`, `experiments/results/feedback_noise_robustness_v3/table_feedback_noise_robustness.tex:10`  
   Fix: 删除 “old objective”；noise table 加 Router/calibrated_hybrid 列或改 prose；更新 `information_systems_format_check.md` 的页数和 abstract count；确认投稿包不包含冲突旧结果。  
   Effort: Low-Medium.

5. **压缩 abstract 和弱化过强 claim。**  
   Location: `paper/main_is.tex:65`, `paper/main_is.tex:80`, `paper/main_is.tex:87`, `paper/main_is.tex:1218`  
   Fix: abstract 压到 250 words 内；把 “proportional” 改成 “tracks/concentrates in”；把 TPC-H 统一成 “six-query sanity check tracks fresh-statistics behavior”；把 “provably ties” 加上 finite projection assumptions。  
   Effort: Low.

## Major Revision Additions Likely Requested by Reviewers

- Multi-config sparse PostgreSQL K sweep with plan-safety columns.
- Router ablation and failure analysis, especially cases where residual chooses a worse plan-shape candidate.
- Formal statement on finite partition/rank and relation to quantile-boundary histograms.
- Baseline appendix with QuickSel-H adaptation details, STHoles hyperparameters, and LQM status as internal learned-query-driven control.
- Per-query TPC-H table or appendix: query IDs, plan signatures, Time/Fresh tail, stale-vs-fresh plan differences.
- Reproducibility appendix: exact scripts, seeds, data generation, generated table provenance.

## Claim Edits

- `paper/main_is.tex:80`: “gain is proportional to the nullspace” -> “gain tracks the residual degrees of freedom left by feedback”.
- `paper/main_is.tex:142`: “every consistent method coincides” -> “any method using the same finite projection and constraints coincides”.
- `paper/main_is.tex:315`: remove STHoles/QuickSel-H from proposition conclusion, or prefix with “their projected variants under our evaluation protocol”.
- `paper/main_is.tex:535`: “cannot increase in-window residual” is true but not enough for safety; add “this does not by itself guarantee plan-shape safety”.
- `paper/main_is.tex:1074`: “reproduce runtime” -> “within the reported 18-instance distribution, track Time/Fresh with zero plan-shape deviations”.
- `paper/main_is.tex:1151`: “first account” -> “to our knowledge, first explicit statistics-layer account...”.

## Artifact Hygiene

- Keep `experiments/results/exp2_sparse_v3_20260601` as the sparse evidence used by the paper.
- Remove, rename, or clearly mark `experiments/results/exp2_sparse_sweep_20260601` as obsolete because its k=2 values contradict the manuscript.
- Update `paper/information_systems_format_check.md`: current PDF is 36 pages; abstract is not 230 words under a simple count.
- Ensure all submitted supplements include only one version of ablation and Router/noise results.

