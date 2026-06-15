# Dropped stale supplement (2026-06-15)

These files were the supplement for the **pre-rewrite** narrative ("Feedback
Nullspace" / learned-prior-as-a-contribution / Stage-1 + Stage-2). They were
**dropped from the submission** when the main paper was re-centered to the
training-free feedback-projection story and the single-column / cross-column
dichotomy (30 pp). They are kept here (not deleted) only for possible later
salvage; nothing in `../main_is.tex` references them anymore.

Why they were inconsistent with the current main paper:

- **`supplementary.tex`** — title still *"…in the Feedback Nullspace"*; uses the
  Stage-1/Stage-2 framing; **asserts the refuted central claim** that the learned
  prior beats ISOMER (*"the learned prior earns its advantage"*, *"learned
  completion beats ISOMER at every budget"*); and `\input`s **7 cut-experiment**
  result tables (trace-grounded drift, optimizer-decision proxy, router
  diagnostics, synthetic stage1-swap, feedback locality, budget sensitivity,
  noise robustness).
- **`mr_supplement.tex`** — the major-revision *"composite-objective Stage-1
  prior"* appendix (the same refuted learned-prior story); no longer `\input` by
  the main paper.
- **`appendix/`** (`system_details.tex`, `mcv_adapter.tex` + `.py`) — `\input`
  only by `supplementary.tex`.

## If a venue later wants a methods supplement

Salvageable sections that do **not** depend on cut experiments or the old
narrative: *Scope and Error Budget*, *Baseline Fairness Discussion*,
*Predicate-Coverage Sensitivity*, *Observation Aggregation*, *PostgreSQL
Planner-Only Statistics-Injection Protocol*, *Detailed Overhead Breakdown*.
To reuse them: retitle, delete the Stage-1/2 and learned-prior-wins sections,
remove the `\input{../experiments/results/...}` lines for cut experiments, and
fix the relative `appendix/` paths.
