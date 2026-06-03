# Cover Letter — Submission to *Information Systems* (Elsevier)

**[DATE]**

To the Editors-in-Chief,
*Information Systems*

Dear Editors,

We are pleased to submit our manuscript, **"OASIS: Repairing Stale Optimizer
Statistics in the Feedback Nullspace,"** for consideration as a research article in
*Information Systems*.

Cost-based optimizers plan with column statistics that grow stale between `ANALYZE`
refreshes, and query feedback exposes the resulting selectivity errors. Our paper makes
the observation that repairing a histogram from such feedback is an *underdetermined
inverse problem*: a handful of observed predicates constrain the marginal only where they
fall and leave the rest free — a region we call the **feedback nullspace**. Classical
feedback-consistency methods (STHoles, ISOMER, QuickSel-H) fill this region with an
optimizer-agnostic maximum-entropy default; we reframe the deployment question as *what
to place in the nullspace* and answer it with **OASIS**, a statistics-layer middleware
that imputes the nullspace with a completion learned against the optimizer's own error,
then projects that completion onto the feedback and routes among candidates from a
deployment-visible signal so that an imperfect prior is always safe to consume.

We believe the work fits *Information Systems* for three reasons. (i) It targets a
practical, under-studied database problem — maintaining optimizer-facing statistics
*between* scheduled refreshes, without rescanning tables or modifying the optimizer.
(ii) It contributes a theory–practice loop: a rank characterization of the feedback
nullspace that predicts *when* a learned completion can help and when it must tie the
maximum-entropy projection, with experiments that confirm the predicted regime structure.
(iii) The evaluation is deliberately deployment-oriented and honest about its
boundaries: a real PostgreSQL planner-injection study, a cross-schema TPC-H runtime
sanity check, composition- and join-estimator integrations, out-of-distribution drift,
and a public telemetry trace, alongside an explicit statement of external-validity
limits.

We confirm that this manuscript is original, has not been published previously, and is
not under consideration for publication elsewhere. All authors have approved the
manuscript and agree to its submission. The authors declare no competing interests. We
have followed the journal's policies on data availability and on the disclosure of
generative-AI assistance (see the Declarations section of the manuscript).

[Optional: We suggest the following potential reviewers, who have relevant expertise and
no conflict of interest with the authors: [NAME, AFFILIATION, EMAIL] × 3–5. We would
prefer that the following individuals not review the manuscript, for reasons of conflict:
[NAMES, if any].]

Thank you for considering our submission. We look forward to your feedback.

Sincerely,

**Qichu Tian** (corresponding author), on behalf of Qichu Tian and Heng Chen
Xi'an Jiaotong University, Xi'an, China
mizukiqwq@stu.xjtu.edu.cn · [ORCID, optional]
