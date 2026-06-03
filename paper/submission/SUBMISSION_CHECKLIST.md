# Submission Checklist — *Information Systems* (Elsevier)

Manuscript: **OASIS: Repairing Stale Optimizer Statistics in the Feedback Nullspace**
Target: *Information Systems* (Elsevier), ISSN 0306-4379 — fresh submission (research article).
Submission portal: Editorial Manager via the journal's "Submit your article" link on ScienceDirect.

---

## A. Files to upload

| # | Item | Status | File |
|---|------|--------|------|
| 1 | Manuscript source (LaTeX, elsarticle) | ✅ ready | `paper/main_is.tex` + `references.bib` + `figures/` |
| 2 | Manuscript PDF (the system also builds one for review) | ✅ ready | `paper/main_is.pdf` (37 pp.) |
| 3 | Highlights | ✅ drafted | `submission/HIGHLIGHTS.txt` (5 bullets, ≤85 chars) |
| 4 | Cover letter | ✅ drafted | `submission/cover_letter.md` → paste/PDF |
| 5 | Supplementary material | ⚠️ check | `paper/supplementary.tex`/`.pdf` (uses LLNCS — see note) |
| 6 | Declaration of Interest form | ⬜ in portal | "none" (also stated in manuscript) |
| 7 | Graphical abstract | ⬜ optional | could reuse the architecture figure (Fig. 2) |

> Tip: Elsevier accepts LaTeX; the system converts your upload to a single review PDF.
> Upload the `.tex`, `.bib`, and `figures/` (and the `\input`-ed result-table `.tex`
> files under `experiments/results/...`), or upload the self-contained PDF for review and
> the source on acceptance.

---

## B. Author-supplied items still to fill (placeholders in the manuscript)

These are currently `[redacted/placeholder]` in `main_is.tex` and must be completed before
the final (non-anonymous) submission, or kept redacted if the journal review is double-blind:

1. **Author names + affiliations** — ✅ done: Qichu Tian (corresponding,
   mizukiqwq@stu.xjtu.edu.cn) and Heng Chen (hengchen@xjtu.edu.cn), Xi'an Jiaotong
   University, Xi'an, China.
2. **Data availability** — ⬜ **ONLY remaining placeholder**: the manuscript currently has
   `\url{REPO-URL-PENDING}`; replace with the public artifact-repo URL (the
   `oasis-artifact/` folder, once pushed) and optionally a Zenodo DOI.
3. **Funding** — ✅ done in manuscript: "did not receive any specific grant...".
4. **CRediT author statement** — ✅ done in manuscript (Qichu Tian: lead implementation/
   experiments/draft; Heng Chen: supervision/review/project administration).
5. **Competing interests** — ✅ done: "The authors declare no competing interests."
6. **Generative AI disclosure** — ✅ done in manuscript (wording below).

### Suggested Generative-AI disclosure (Elsevier policy: place before the references)

> During the preparation of this manuscript the author(s) used a large language model
> (Anthropic Claude) to assist with language editing, manuscript restructuring, and the
> drafting of figure- and table-generating code. No AI tool was used to design the study,
> generate or analyze experimental results, or produce scientific claims. After using
> this tool, the author(s) reviewed and edited the content as needed and take full
> responsibility for the content of the publication.

*(Adjust the scope to match what you actually used; AI must not be listed as an author.)*

---

## C. Pre-flight checks (manuscript)

- [x] Compiles clean: 0 errors, 0 undefined references/citations, 0 `??`, 37 pages.
- [x] Abstract 225 words (≤250).
- [x] 6 keywords present.
- [x] References via `elsarticle-num`; bibliography builds (1 benign BibTeX warning:
      CIDR `kipf2019learned` has no page numbers — expected for CIDR).
- [ ] **Language consistency**: pick American *or* British English (journal requires not a
      mix). Quick pass on -ize/-ise, "behavior/behaviour", etc.
- [ ] Confirm all figures are vector/≥300 dpi (Figs. 1,3,4,6,7 are vector TikZ/PDF;
      Fig. 2 architecture is vector PDF from SVG; Figs. 4-heatmap/trace are PDF — verify
      resolution).
- [x] Anonymization: real author names are in (single-blind, the IS default).

## D. Notes / decisions for you

- **Supplement** — ✅ resolved: `supplementary.tex` converted from Springer `llncs` to a
  neutral `article` class (compiles clean, 17 pp., `supplementary.pdf`); real authors
  added. Submit `supplementary.pdf` as the supplementary file. `mr_supplement.tex` is an
  orphan (referenced by nothing, superseded) — safe to delete.
- **Public artifact repo** — ✅ staged at `../../oasis-artifact/` (git-initialized, code +
  trained checkpoint + result tables + README/LICENSE/requirements; large rows and the
  NASA `.gz` excluded). Push it to a new GitHub repo, then paste its URL into the
  manuscript's Data-availability `\url{REPO-URL-PENDING}`.
- **Highlights wording** is in `HIGHLIGHTS.txt`; paste into the Editorial Manager
  highlights field (they are each ≤85 characters incl. spaces).
- **Suggested reviewers**: optional but helpful — list 3–5 non-conflicted experts in
  cardinality estimation / self-tuning histograms / learned DB components.
