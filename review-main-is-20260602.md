# OASIS Information Systems 审稿报告

**Venue:** Information Systems  
**Review date:** 2026-06-02  
**Language:** Chinese  
**Paper:** `paper/main_is.tex`, `paper/main_is.pdf`

## Overall Recommendation

**Recommendation:** Major Revision  
**Score:** 6.2/10  
**Confidence:** 4/5

这篇论文把 stale single-column optimizer statistics repair 重新表述为 feedback-constrained underdetermined marginal completion，并把 OASIS 定位为 optimizer-facing completion：ISOMER/maximum-entropy projection 是 optimizer-agnostic completion，OASIS 学习 feedback nullspace 中更面向 optimizer 的补全。这个主线比传统“learned histogram correction”更有辨识度，且 ISOMER、STHoles、QuickSel-H、learned CE、plan-aware CE 之间的边界基本讲清楚了。当前版本已经具备 Information Systems 期刊 major-revision 潜力，但还不到直接接收的程度。

主要风险不是“方向不新”，而是若干强 claim 的精度、数学表述的严谨性、Router safety 的证据链、以及主文/补充材料数字一致性。若不修，这些问题足以把稿件从 Major Revision 拉到 Reject。

## Summary

论文提出 OASIS，一个统计层 middleware，用 query feedback 修复 stale single-column marginal statistics。它的核心观点是：feedback 只约束部分 marginal，剩余自由度形成 feedback nullspace；经典 feedback-consistency 方法用 optimizer-agnostic 规则填充这个 nullspace，而 OASIS 用 composite optimizer-facing objective 训练 prior，再经 projection 和 Router 部署。

贡献有真实价值。把 ISOMER 的 maximum-entropy 解释为 completion，把 learned prior 的收益限制在 sparse/underconstrained feedback regime，并在 dense/rank-saturated regime 与 ISOMER 收敛，是一个合理且审稿人能抓住的主线。实验覆盖 synthetic drift、DML traces、NASA telemetry proxy、composition estimators、FactorJoin、PostgreSQL planner injection 和 TPC-H sanity check，覆盖面比普通 workshop-style learned CE 论文更像期刊稿。

但当前稿件仍有几处会被严厉审稿人追问：Proposition/rank 的有限维表示仍混合了 equi-depth quantile boundary、fixed cell mass、feedback-induced partition；Router 的 residual gate 不能充分推出 plan safety；PostgreSQL sparse K sweep 虽已进正文，但只有单一代表配置且主表不展示 plan-safety 列；supplement 仍有旧表述和数字/列不一致；部分 claim 使用 “proportional / provably / reproduce fresh runtime” 仍偏强。

## Strengths

- **主线比旧式 learned correction 更清楚。** Abstract 和 Introduction 已明确提出 feedback nullspace、optimizer-facing completion、rank saturation 收敛等概念（`paper/main_is.tex:65`, `paper/main_is.tex:126`, `paper/main_is.tex:148`）。
- **相对 ISOMER/STHoles/QuickSel-H 的定位基本可辩护。** 论文承认 self-tuning histogram 传统，并把 OASIS 的差异限定为 completion rule 和 optimizer-facing objective（`paper/main_is.tex:237`, `paper/main_is.tex:240`）。
- **不是只做 selectivity proxy。** 论文有 composition family、FactorJoin、PostgreSQL planner-only injection 和 TPC-H sanity check（`paper/main_is.tex:930`, `paper/main_is.tex:951`, `paper/main_is.tex:975`, `paper/main_is.tex:1051`）。
- **失败模式有报告。** Raw prior 在 FactorJoin 和 NASA trace 上有害，projection/router 被作为必要 safety layer，而不是纯粹美化（`paper/main_is.tex:967`, `paper/main_is.tex:912`, `paper/main_is.tex:1180`）。
- **dense feedback 与 ISOMER 收敛的负结果是加分项。** PostgreSQL dense planner batch 中 OASIS 与 ISOMER near-identical，论文没有把它包装成胜利（`paper/main_is.tex:989`）。

## Major Weaknesses

### 1. Nullspace/rank proposition 仍不够严谨

论文已经从 quantile-boundary 表述改成 bucket-mass 表述，这是正确方向（`paper/main_is.tex:294`）。但 Proposition 仍有两类不严谨：

- `paper/main_is.tex:309` 的 proposition 假设所有方法都通过同一线性约束 `A m = y` enforce feedback，然后在 `paper/main_is.tex:315` 直接说 ISOMER、STHoles、QuickSel-H 和任意 prior projection 都返回同一 marginal。这个结论只对“同一个 projection operator + 同一个 finite representation”成立，不是这些原始算法天然成立。STHoles/QuickSel-H 的原始算法并不等价于同一 KL/I-projection。
- `paper/main_is.tex:294` 到 `paper/main_is.tex:307` 仍把 `B=10` equi-depth histogram 的自由度、fixed bucket-mass grid、以及 feedback-induced interval partition 放在一起。真正线性的对象应是 feedback-induced partition 上的 cell masses，维度应写成 `C-1-r`，其中 `C` 是由 stale support / feedback endpoints / projection cells 诱导出的 cell 数，而不是总是 `B-1`。

建议：把 proposition 改成“在有限 cell-mass partition 上，同一 hard projection 的 feasible set 若为 singleton，则任何初始化 prior 的 projection 相同”；把 STHoles/QuickSel-H 从 theorem 结论中移出，改到实验比较段。Rank table 也应说明它是 `B=10` histogram exposed degrees 的 operational proxy，还是严格的 `A` rank。

### 2. “Rank analysis shows gain is proportional to nullspace” 偏强

Abstract 说 gain is proportional to nullspace and vanishes when dense feedback pins the marginal（`paper/main_is.tex:80`）。实验实际支持的是 monotone/correlative pattern，而不是比例关系。Rank table 是均值 DOF（`paper/main_is.tex:551`），feedback-locality 是按距离分箱的 empirical signature（`paper/main_is.tex:717`），budget table 在 optimizer proxy 上 K=16 仍有 OASIS 优势（`paper/main_is.tex:766`），并不构成比例定律。

建议：将 “proportional to” 改成 “tracks / is concentrated in / is bounded by”。Conclusion 里的 “provably ties ISOMER only where dense feedback saturates” 也应限定为同一 finite representation 和同一 projection operator（`paper/main_is.tex:1218`）。

### 3. Router 非 oracle 但 safety 证据不足

Router 的 non-oracle 叙述是对的：它只用 feedback residual，不看 test predicates、true cardinalities、runtime 或 plan shape（`paper/main_is.tex:516`, `paper/main_is.tex:533`）。实现上也确实是 residual argmin（`experiments/oasis_accuracy_smoke.py:278`, `experiments/oasis_accuracy_smoke.py:286`）。

问题是：最小化 feedback residual 只能保证 in-window residual 不增加，不能保证 future selectivity、join regret 或 plan-shape safety。论文自己在 Limitations 承认 residual 不是 plan-shape risk signal（`paper/main_is.tex:1198`），但前文多次说 Router “supply optimizer safety” 或 “make an imperfect prior safe to compose, join, and inject into a planner”（`paper/main_is.tex:175`, `paper/main_is.tex:534`）。这中间缺少一个实证闭环：residual 分布、tie rate、错误选择率、相对 oracle router/random router/plan-aware gate 的对照。

建议：新增 Router ablation：residual router vs random among projected candidates vs always-OASIS vs always-ISOMER vs oracle-on-heldout；报告 residual ties、active constraints dropped 后 residual 的区分能力、Router 选择造成的 worst-case plan deviations。

### 4. PostgreSQL sparse K sweep 已进正文，但证据还偏薄

用户关心的 “PostgreSQL sparse K sweep” 正文已经有了（`paper/main_is.tex:1009`），且数字有 `exp2_sparse_v3_20260601` 日志支持（例如 `experiments/results/exp2_sparse_v3_20260601/k2/table_postgres_planner_stats_injection.tex:11`）。这解决了“正文完全缺少 DBMS-facing sparse evidence”的问题。

但作为 IS 期刊证据仍不足：

- 主文表只列 Row QE，不列 Fresh Plan / Recovery / New Deviations，尽管 prose 用这些指标支撑 safety（`paper/main_is.tex:1013`, `paper/main_is.tex:1027`）。生成表其实有这些列（`experiments/results/exp2_sparse_v3_20260601/k2/table_postgres_planner_stats_injection.tex:9`）。
- 该 sweep 只是一组代表配置：left-shift, 40K rows, 84 queries（`paper/main_is.tex:1011`）。dense planner batch 是 12 configurations（`paper/main_is.tex:978`），但 sparse sweep 不是。
- 仓库中仍有旧的 `exp2_sparse_sweep_20260601` 结果，其 k=2 行与正文结论相反：OASIS 6.031、Router 9.967，而非 1.96/1.95（`experiments/results/exp2_sparse_sweep_20260601/k2/table_postgres_planner_stats_injection.tex:11`）。这不一定会进入投稿包，但如果补充材料或 artifact 暴露它，会造成严重可信度问题。

建议：把 sparse sweep 扩展到至少 `2 drift directions x 2 table sizes x 3 seeds`，并在主文或 supplement 给出 Row QE + plan metrics 的完整表。

### 5. Baseline fairness 需要更强说明

论文称所有方法同样 stale stats、same feedback、same projection、same split（`paper/main_is.tex:613`），这有助于公平性。但审稿人仍会追问：

- QuickSel-H 是作者 adaptation，不是 QuickSel 原始输出；补充材料承认不做 line-by-line reproduction（`paper/appendix/system_details.tex:82`, `paper/appendix/system_details.tex:111`）。压缩到 `B-1` quantiles 再 projection 可能弱化 QuickSel。
- STHoles 只允许 up to K refinement steps（`paper/supplementary.tex:157`），是否调参到最佳、是否支持多维 holes、是否和 OASIS 的 learned prior 训练预算可比，需要更明。
- LQM 是 “recent learned query-driven baseline” 风格实现，不是公开 learned CE 系统本身（`paper/main_is.tex:609`）。不能把战胜 LQM 表述成战胜 learned CE。

建议：把 baselines 分成三类：true self-tuning baselines、adapted histogram-interface baselines、internal learned-query-driven control。对 QuickSel-H 和 LQM 的 claim 降低。

### 6. TPC-H runtime sanity check 表述已有改善，但仍要压住

正文已经明确 TPC-H 是 deliberately narrow runtime check，不声称 beat stale（`paper/main_is.tex:1054`）。Tail safety 段落也补了 worst-case（`paper/main_is.tex:1085`）。这是好修正。

仍需谨慎：

- Abstract 说 “on TPC-H reproduces fresh-statistics plans and accuracy without a table scan”（`paper/main_is.tex:87`），没有提 runtime；Conclusion 又说 reproduces fresh-statistics accuracy and runtime（`paper/main_is.tex:1224`）。建议全文统一为 “tracks fresh-statistics behavior on a six-query sanity check”。
- Table caption 说 calibrated statistics match fresh statistics on accuracy and runtime（`experiments/results/postgres_runtime_tpch_multiseed_20260601/table_tpch_runtime.tex:3`）。Time/Fresh 的 mean 是接近 1，但 worst-case Router 1.40（`paper/main_is.tex:1090`）。应把 “match” 改成 “within reported distribution; no plan-shape deviations in 18 instances”。
- 只测 6 curated date-sensitive queries、3 seeds（`paper/main_is.tex:1060`）。这个实验不能支撑 general DBMS runtime claim。

### 7. Supplement / artifact 一致性有硬伤

这类问题在期刊审稿中很伤，因为它们让人怀疑版本管理：

- `mr_supplement.tex` 的 ablation table 仍写 “Reconstruction-only (old objective)”（`paper/mr_supplement.tex:56`），主文表已改成 “Histogram reconstruction only”（`paper/main_is.tex:802`）。
- `mr_supplement.tex` 说 Router 在 `K=16` fallback to ISOMER（`paper/mr_supplement.tex:137`），但主文 feedback-budget table 的 Hybrid choice 在 K=16 仍主要选 OASIS projected，ISOMER 约 33%（`experiments/results/feedback_budget_sensitivity_v3/summary.txt:40`, `experiments/results/feedback_budget_sensitivity_v3/summary.txt:49`）。如果该 supplement 会提交，必须修。
- `supplementary.tex` 的 feedback-noise prose 使用 Router/`calibrated_hybrid` 数字 `1.375 -> 1.418`（`paper/supplementary.tex:516`），但实际 input 的 LaTeX noise table 没有 Router 列，只显示 Hybrid `1.437 -> 1.496`（`experiments/results/feedback_noise_robustness_v3/table_feedback_noise_robustness.tex:10`, `experiments/results/feedback_noise_robustness_v3/table_feedback_noise_robustness.tex:15`）。要么加 Router 列，要么改 prose。
- `information_systems_format_check.md` 称 PDF 23 页、abstract 230 words（`paper/information_systems_format_check.md:11`, `paper/information_systems_format_check.md:13`），但当前 PDF 是 36 页，粗略 abstract 计数约 266 words（`paper/main_is.tex:65`）。如果 abstract cap 是 250 words，应立即压缩。

## Novelty Assessment

**Novelty score:** 7.0/10  
**Confidence:** 4/5

相对于 ISOMER，OASIS 的 novelty 不在 feedback consistency 或 maximum entropy projection。这些已有工作已经把 query feedback 约束和最大熵/一致性估计建立起来。OASIS 的可发表 novelty 是：把最大熵解释为 optimizer-agnostic completion；把 learned prior 约束在 feedback nullspace；再用 rank/nullspace 分析解释何时 learning 有用、何时必须与 ISOMER 收敛。这个角度是可以站住的，但 “first account” 应避免过宽。

相对于 STHoles/self-tuning histograms，OASIS 不是第一个 feedback-driven histogram repair，也不是第一个 no-rescan statistics update。它的新意是 optimizer-facing learned completion + projection/router deployment layer。论文现在的 related work 基本承认了这一点（`paper/main_is.tex:1137`），建议继续压低“首创修复”的语气。

相对于 QuickSel/learned query-driven estimators，OASIS 的差别是输出 optimizer-consumable statistics 而非 point selectivity estimates。这是重要边界。但 QuickSel-H 是 adaptation，不能把结果过度推广为全面优于 QuickSel。

相对于 learned CE 和 plan-aware CE/Flow-Loss/Bao/Neo，optimizer-facing objective 本身不是全新。新意在于把 plan-aware/optimizer-facing 信号用于 statistics-layer marginal completion，并仍交给原 CBO 使用。Related Work 已经提到这些系统（`paper/main_is.tex:1169`），但还可以更明确说“我们不是替代 learned CE，而是给这些估计器提供 corrected marginals”。

## Technical Soundness

**Score:** 5.7/10  
**Confidence:** 4/5

核心方法是合理的：learned prior + feedback projection + residual router 是工程上可行的分层设计。Projection 作为 completion operator 的定位清楚（`paper/main_is.tex:491`），composite objective 包含 future-predicate error、join regret、feedback residual、reconstruction regularizer（`paper/main_is.tex:454`）。

主要 soundness gap 是理论表述和 deployment safety：

- theorem-like claim 超过了所证明的有限维条件；
- rank table 的 operational rank 与数学 `rank(A)` 不完全等价；
- Router 的 residual safety 只能保证 in-window residual，不保证 optimizer plan safety；
- “learned prior's gain is proportional to nullspace” 应改为 empirically correlated / bounded by。

## Experimental Validation

**Score:** 6.1/10  
**Confidence:** 4/5

实验覆盖面广，但 IS 期刊会更看重 external validity 和 reproducible protocol。当前强项是：主结果表给出 OASIS over ISOMER 12.8%（`experiments/results/proj_v3/table_projection_initialization.tex:18`）、locality bins（`experiments/results/proj_v3/table_feedback_locality.tex:12`）、objective ablation（`paper/main_is.tex:786`）、PostgreSQL planner batch（`experiments/results/postgres_batch_v3/table_postgres_planner_stats_injection_batch.tex:12`）和 TPC-H runtime sanity（`experiments/results/postgres_runtime_tpch_multiseed_20260601/table_tpch_runtime.tex:11`）。

短板是：多数关键收益来自 synthetic/domain-randomized drift；PostgreSQL sparse K sweep 是单 config；TPC-H 是 6 curated queries；NASA 是 HTTP append proxy 而非 DBMS query log。对 Information Systems 而言，这会触发 “does it work in real information systems?” 的追问。

## Clarity and Presentation

**Score:** 6.5/10  
**Confidence:** 4/5

文章结构清楚，但摘要偏长且密度过高。`pdftotext` 显示多个 paragraph heading 出现双句点，例如 “problem..”, “Contribution..”, “Tail safety..”，来自 `\paragraph{...}` 后手写句点（`paper/main_is.tex:126`, `paper/main_is.tex:148`, `paper/main_is.tex:1085`）。这类问题不是拒稿点，但会降低成熟度观感。

建议减少 abstract 的数字堆叠，把核心 novelty 放在一句话中；正文所有 `\paragraph{...}.` 改为不重复句点；把 “Hybrid / Router / Aggressive / Soft” 命名统一，否则读者很难判断哪个是部署默认。

## Significance and Impact

**Score:** 6.6/10  
**Confidence:** 4/5

如果主线修严谨，这篇稿件对 optimizer statistics maintenance 有意义。它不像大多数 learned CE 工作那样要求替换 optimizer，而是输出 native statistics，这对 Information Systems 的工程读者是加分点。

影响力上限受限于单列 marginal scope、缺少 production DBMS workload、以及 runtime 证据窄。当前更像一个强 IS journal paper 的雏形，而不是已经能冲 VLDB/ICDE research-track 的系统论文。

## Reproducibility

**Score:** 5.8/10  
**Confidence:** 3/5

LaTeX source 引用大量 generated tables，结果路径清楚。部分 claim audit 也显示数字经过核对。但投稿前必须清理旧结果目录、补充 Router/noise 表、给出 scripts/commands 和随机种子说明。尤其是旧 sparse sweep 目录与新 sparse v3 结果相冲突，artifact 包中不能混放。

## Questions for Authors

1. Proposition 的 `A` 是定义在 fixed `B` histogram cells 上，还是 feedback-induced partition cells 上？如果是后者，为什么 rank table 仍以 `B-1=9` 为 marginal DOF？
2. 对 STHoles 和 QuickSel-H，“same projection” 是评估后处理还是算法本身？如果是后处理，是否应称为 projected STHoles-init / QuickSel-init？
3. Router 在 hard-projected ISOMER 与 hard-projected OASIS 都满足 active constraints 时，residual 如何区分？tie rate 是多少？
4. Sparse PostgreSQL K sweep 是否能扩展到 12-configuration batch？如果不能，为什么代表配置足以支撑 5.2x headline？
5. QuickSel-H 的 `M=5`、STHoles 的 K refinement steps 和 OASIS 的 offline training budget 是否经过相同调参？
6. TPC-H 6 个 queries 的选择标准是什么？是否包含所有 date-sensitive TPC-H queries，还是 curated subset？

## Major Revision Requests

1. Rewrite Proposition/rank section using feedback-induced cell-mass partition, and narrow theorem claims to projected finite-representation methods.
2. Add multi-configuration PostgreSQL sparse K sweep, or explicitly downgrade sparse planner claim to a representative case study.
3. Add Router diagnostics: residual distributions, tie rates, choice confusion against held-out oracle, and plan-safety worst cases.
4. Fix supplement/main inconsistencies, especially `old objective`, feedback-noise Router table, sparse sweep artifacts, and abstract/page metadata.
5. Strengthen baseline fairness: QuickSel-H adaptation limits, STHoles tuning, LQM as internal control, no broad learned-CE superiority claim.

## Minor Issues

- Abstract likely exceeds 250 words under a simple count and should be shortened (`paper/main_is.tex:65`).
- Replace “proportional to the nullspace” with weaker empirical wording (`paper/main_is.tex:80`).
- Replace “every consistent method coincides” with “any method using the same finite projection coincides” (`paper/main_is.tex:142`).
- Avoid saying “full-stack TPC-H safety” in `mr_supplement.tex`; the actual TPC-H evidence is narrow (`paper/mr_supplement.tex:144`).
- Add Data Availability / COI / funding / AI-assistance declarations before final Elsevier submission (`paper/information_systems_format_check.md:18`).

## Expected Outcome

**Current direct submission to Information Systems:** likely Major Revision if reviewers are sympathetic; possible Reject if assigned to a theory-heavy optimizer/statistics reviewer because of Proposition/rank precision and limited real workload evidence.

**After necessary modifications:** solid Major Revision -> Minor Revision trajectory; with multi-config sparse planner evidence and cleaned supplement, could become Acceptable for IS.

**Distance to ICDE/VLDB:** still substantial. ICDE/VLDB would likely require a real end-to-end DBMS workload, stronger system integration, production-like refresh histories, broader runtime impact, and a cleaner theoretical statement. Current work is closer to an IS journal systems-methodology paper than a top DB conference systems paper.

