# Paper Review: OASIS: Repairing Stale and Correlation-Blind Query-Optimizer Statistics from Query Feedback

**Venue assumption:** database systems / Information Systems-style journal review  
**Review date:** 2026-06-15  
**Review language:** Chinese  
**Reviewer:** Codex AI Assistant

---

## Overall Recommendation

- **Decision:** Borderline / Weak Reject for a top database venue; Major Revision for a journal
- **Overall Score:** 6.4/10
- **Confidence:** 4/5

## Summary

本文研究用 query feedback 在不重新扫描表的情况下维护优化器统计信息。核心观点是一个二分：单列统计修复中，反馈投影/最大熵完成已经接近最优，学习先验几乎没有额外收益；真正有价值的是跨列相关性，因为 conjunctive feedback 直接观测了独立性假设遗漏的联合质量。论文把这一思路包装为 OASIS，包含统计格式转换、反馈一致性投影和 residual-gated router，并在真实 PostgreSQL planner 注入实验中展示 row Q-error 和 fresh-plan match 的显著改善。

我认为这篇稿件的主线清楚，问题重要，写作成熟，实验层次也比纯模拟工作更接近系统落地。但当前版本的主要风险是 novelty 和 evidence boundary：核心单列算子明确是 ISOMER/IPF，PostgreSQL planner 实验中 OASIS 与 ISOMER 几乎打平；跨列 IPF 修复是本文最有潜力的增量，但尚未进入端到端 planner/runtime 注入；router 的“不退化/安全”表述也主要由 in-window residual 支撑，不能推出 plan-shape safety。作为审稿人，我会要求大修后再考虑接收。

## Strengths

- **问题重要且定位准确。** Stale statistics 和 attribute-value independence 都是 CBO 中长期存在的真实痛点，论文从 statistics layer 而不是替换优化器入手，部署视角合理。
- **叙事主线强。** “单列容易、跨列才是瓶颈”的 dichotomy 很容易理解，也能把 identifiability、single-column、cross-column、planner injection 串起来。
- **理论解释有助于理解现象。** Proposition 说明当反馈约束满秩时，不同 prior 在同一 I-projection 下会收敛到同一结果；这解释了 PostgreSQL planner 实验中 OASIS 和 ISOMER 几乎相同的现象。
- **实验包含真实系统触点。** PostgreSQL `pg_statistic` 注入、JOB conjunctive predicates、TPC-H DML drift sanity check 都比只在合成数据上报 selectivity Q-error 更有说服力。
- **限制写得较诚实。** 论文明确承认没有生产 refresh history、TPC-H runtime 只是 sanity check、cross-column 还不是端到端 planner injection。

## Weaknesses

- **新颖性边界仍不够硬。** 单列 hard operator 是 ISOMER/IPF；跨列 joint repair 也是二维 max-entropy/IPF 投影。论文需要更清楚地区分自己和 multidimensional STHoles、ISOMER-style consistency repair、PostgreSQL extended statistics、adaptive cardinality feedback 的实质差异。
- **最强系统结果没有证明 OASIS 超过 ISOMER。** PostgreSQL planner-only 实验中 OASIS row Q-error 2.377、ISOMER 2.390、Router 2.386，差异可忽略。论文把这解释为 rank mechanism 的预期结果，这是合理的，但会削弱“OASIS system”的贡献感。
- **跨列贡献没有端到端系统验证。** 跨列结果在 real pairs 和 JOB sampled conjunctive predicates 上有效，但没有展示修复后的 joint statistics 如何进入 PostgreSQL optimizer、如何改变真实多表查询计划或 runtime。
- **安全性表述偏强。** 文中多次使用 safe、never degrades below maximum-entropy default 等表述，但 router 只优化 in-window feedback residual；它不观察 plan shape，也不能保证 future predicates 或 downstream cost 不退化。文中 limitations 已承认这一点，但摘要和贡献表述仍偏强。
- **实验外部有效性不足。** 漂移来自受控模型或 benchmark stream，不是生产统计刷新历史；single-column 数据集只有 Power/Forest/Census，cross-column real-pair 表只有 8 对，TPC-H 只有 6 个 date-sensitive queries。
- **统计显著性和尾部分析不足。** 多数核心表格只有 geometric mean，没有置信区间、seed-level 分布、失败案例或 worst-case tail。TPC-H 有尾部文字说明，但其他部分不足。
- **baseline 可比性有潜在争议。** QuickSel-H 是 adapted histogram-interface baseline，不是原系统；LQM 是内部控制；STHoles/QuickSel-H 与“统一投影后比较 prior”的设定容易让读者怀疑是否削弱了原算法的优势。

## Detailed Evaluation

### 1. Originality & Novelty

**Score:** 6.2/10 | **Confidence:** 4/5

论文的新颖性主要在 framing 和系统化拆分，而不是核心优化算子。单列部分主动承认 hard operator 是 ISOMER/IPF，这很诚实，但也意味着单列 repair 的技术贡献更多是解释和实证确认。跨列 feedback-driven joint repair 是更有价值的方向，但二维 IPF/最大熵修复与早期 multidimensional feedback histograms 的关系需要进一步澄清。

### 2. Technical Quality & Soundness

**Score:** 7.0/10 | **Confidence:** 4/5

技术路线总体合理。Proposition 是正确的线性代数/凸投影解释，能解释 projection-based methods 在 pinned regime 下重合。不过其假设较强：consistent and noise-free feedback、相同 projection、精确 cell-level representation；实际系统中会有反馈噪声、并发更新、predicate mismatch、sampling/counting error 和 stale constraints conflict。论文描述了 drop-old-constraints 和 soft projection，但缺少形式化分析或系统性实验。

### 3. Experimental Validation

**Score:** 5.9/10 | **Confidence:** 4/5

实验覆盖层次不错，但最关键的 claims 还差一层验证。Single-column 证明了 ISOMER 类方法有效且学习 prior 收益小；PostgreSQL injection 证明了统计注入可影响计划；cross-column 证明了 conjunctive feedback 能降低 Q-error。缺口是：跨列 repair 尚未进入真实 optimizer plan/runtime；single-column planner 结果几乎等于 ISOMER；runtime 实验明确不声称优于 stale，且范围很窄。

### 4. Clarity & Presentation

**Score:** 8.0/10 | **Confidence:** 4/5

论文写作明显经过打磨，贡献线和限制线都很清楚。主要问题是术语层面：OASIS、ISOMER、hard projection、soft projection、Router、Hybrid、learned prior 的关系需要更早、更简洁地固定。当前读者可能读到实验表格时才意识到 OASIS 在多个配置下几乎就是 ISOMER。

### 5. Significance & Impact

**Score:** 6.8/10 | **Confidence:** 4/5

如果跨列反馈统计能真正进入优化器并稳定改善复杂 workload，影响会很高。当前证据已经显示方向有价值，但系统影响还停在“planner-facing evidence + sanity check”。作为数据库 journal，大修后有潜力；作为顶会系统论文，目前说服力略弱。

### 6. Reproducibility

**Score:** 7.0/10 | **Confidence:** 3/5

稿件声明公开代码、数据来源和随机种子，并且仓库中存在表格、脚本和 artifact 目录。正文对核心参数 K=16、B=10、G=12 也有说明。仍建议补充更直接的 reproduction protocol：每个表格/图对应的命令、运行时间、硬件环境、PostgreSQL 配置、随机种子列表。

### 7. Visual Elements Quality

图表总体可读，Figure 1 对 pinned/free intuition 有帮助，downstream broken-axis 图能压缩展示不同量级。但 broken axis 容易被审稿人质疑视觉夸张，应确保 caption 明确；跨列和 PostgreSQL 表格建议加入 confidence interval 或 per-seed distribution。

### 8. Ethics & Limitations

伦理风险很低。Generative AI disclosure、data availability、funding/competing interests 都有。Limitations 写得比较到位，但摘要/贡献中的安全性措辞应与 limitations 保持一致。

## Questions for Authors

1. OASIS 与 ISOMER 的实质系统差异是什么？在什么 workload/feedback regime 下 OASIS 会明显不同于 ISOMER，而不是只复现 ISOMER？
2. 跨列 joint repair 能否注入 PostgreSQL 或另一个优化器并影响真实 multi-predicate/multi-table query plan？
3. Router 使用 in-window residual，是否存在 residual 低但 future predicate 或 plan shape 更差的案例？有没有负例分析？
4. 对 noisy feedback、并发更新、COUNT(*) 不可得或 executor row-count 粒度不足的情况，系统如何处理？
5. K、B、G、feedback workload 与 future workload mismatch 对结果敏感吗？

## Suggestions for Rebuttal / Revision

- 把贡献重心更明确地放到“identifiability-guided diagnosis + cross-column feedback repair”，减少“new single-column method”的暗示。
- 增加跨列端到端实验：至少在 PostgreSQL/JOB 上展示修复 joint selectivity 后的 plan shape、row estimates、runtime 或 optimizer cost impact。
- 将安全性 claim 改为“residual-gated empirical guard”或“does not degrade in evaluated regimes”，避免理论保证口吻。
- 加入 sensitivity analysis：K、B、G、反馈稀疏度、feedback/future predicate distribution mismatch、噪声/冲突约束。
- 给核心结果添加 error bars、per-seed tables、tail/worst-case cases，尤其是 cross-column 和 planner injection。
- 更正或统一 OASIS/ISOMER/Router/Hybrid 命名，让读者能一眼看出每个 column 对应的算法。

## Minor Issues

- `\journal{a database systems journal}` 仍是占位符，投稿前应替换为目标期刊。
- “never degrades below the maximum-entropy default” 建议改成只针对 in-window residual 的陈述。
- Cross-column 表中 Pearson r 不能完全代表 nonlinear dependence，建议报告 mutual information 或 rank correlation 作为补充。
- QuickSel-H/LQM 的非原版性质应在实验表 caption 或脚注中再次提醒。
- TPC-H runtime 中 stale 更快的现象处理得诚实，但容易被误读，应在摘要和 introduction 避免给出 runtime improvement 暗示。

## Detailed Comments by Section

### Abstract / Introduction

摘要清楚，但结论密度很高。“cuts row Q-error from 27.8 to 2.4 and lift fresh-plan match from 56% to 96%” 很强；建议同时说明这是 single-column statistics injection，不是 full cross-column OASIS end-to-end。

### Problem / Identifiability

Pinned/free framing 很好。Proposition 正确但偏基础，贡献在解释力而不是理论深度。建议把 exact cell-rank 和 operational rank 的差异讲得更直观，并说明 feedback endpoints 不覆盖未来 predicate 区域时会发生什么。

### Method

Repair operator 是成熟技术，论文应更突出工程接口和 cross-column use case。Router 的定位要降调：它是 empirical selector，不是安全证明。

### Experiments

实验设计层次清楚，但当前 evidence ladder 的最高一层仍是 single-column planner injection。建议把 cross-column 的 JOB 结果推进到真实 optimizer decision，否则“independence repair earns its keep for the optimizer”仍显得偏估计层面。

### Conclusion

结论准确但略强。建议把“safe for composition and join estimators”改成“empirically safe in evaluated composition and join-estimator settings”。

---

**End of Review**
