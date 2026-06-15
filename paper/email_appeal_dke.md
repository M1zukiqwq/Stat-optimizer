# Email appeal to the DKE handling editor (scope-misclassification)

> HOW TO USE: send this as a **reply to the rejection email** (keeps the thread + manuscript
> number visible), or as a new email to the handling editor (cc the EiC if you like). Plain
> text — no formatting needed. Verify the salutation 〔Dr. Woo〕 against the name on your
> decision letter before sending. Keep it short; do not attach the full manuscript unless the
> editor asks.

---

**Subject:** Scope clarification and request to reconsider — DATAK-D-26-01189

Dear Dr. 〔Woo〕,

Thank you for handling our submission "OASIS: Repairing Stale Optimizer Statistics…"
(DATAK-D-26-01189) and for the suggestion of operations-research venues. We write, with
respect, to clarify a point of scope, because we believe our earlier title gave a misleading
impression of the work's field.

The paper is a **database-systems** contribution, not an operations-research or
mathematical-optimization one. Here "optimizer" refers to the **cost-based query optimizer**
of a relational DBMS, and the work concerns maintaining the column statistics that optimizer
uses for **cardinality (selectivity) estimation**, repaired from query-execution feedback. We
can see how the earlier title—which paired "Optimizer" with "Feedback Nullspace"—invited a
mathematical-optimization reading; that was a framing error on our part. We have since
retitled the manuscript to **"OASIS: Repairing Stale and Correlation-Blind Query-Optimizer
Statistics from Query Feedback"** and reframed it so the database scope is unambiguous
throughout: the "nullspace" terminology is gone from the title and abstract, and the
contribution and evaluation—on real datasets and a live PostgreSQL query planner—are stated
in database terms (histograms, selectivity, `EXPLAIN`, plan shapes).

Given this, we would be grateful if you would consider whether the revised manuscript falls
within DKE's scope—e.g., database-system design and implementation techniques, and
data-administration/maintenance (keeping a system's optimizer statistics current between
`ANALYZE` refreshes). If you agree, we would gladly submit the revised version through
whatever channel you advise; we are also happy to send the updated manuscript for a quick
look. We of course respect your decision either way.

Thank you for your time and for reconsidering our work.

Sincerely,

Qichu Tian (corresponding author), Heng Chen
Xi'an Jiaotong University, Xi'an, China
mizukiqwq@stu.xjtu.edu.cn
