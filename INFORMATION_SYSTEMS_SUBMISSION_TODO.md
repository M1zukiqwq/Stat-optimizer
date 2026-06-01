# Information Systems Submission Todo

Target: Elsevier Information Systems.

Review mode: single anonymized. Do not anonymize the manuscript or repository for
submission. Restore real authors, affiliations, acknowledgements, funding, and
artifact links before uploading.

## 0. Claim Boundary

- [ ] Keep the main claim as a statistics-calibration/system paper, not a broad
      runtime-speedup paper.
- [ ] State that Stage 2 is the core contribution: hard projection, Soft, Hybrid,
      Router, and the regime-dependent deployment story.
- [ ] State OASIS-noProj as an ablation/raw prior, not the deployed method.
- [ ] Keep TPC-H runtime wording bounded: no `>15%` median-time regressions in
      this check, and plan-to-runtime translation on plan-change queries.
- [ ] Keep NASA as public production telemetry / append-only case study, not a
      private DBMS workload trace.

## 1. Finish Running Experiments

- [ ] Finish 2-3 additional TPC-H dbgen/refresh seeds.
- [ ] For each seed, save `run_config.json`, logs, CSV/JSON rows, and table inputs.
- [ ] Aggregate TPC-H results across seeds:
  - [ ] scan Q-error by method;
  - [ ] plan-change count;
  - [ ] runtime ratio vs stale/fresh;
  - [ ] `>15%` median-time regression count;
  - [ ] per-query plan signatures for stale/fresh/Router.
- [ ] Verify no post-hoc query dropping. If a query is excluded, document the
      pre-run reason.
- [ ] Update `table_tpch_runtime.tex` from the aggregate, not a single seed.
- [ ] Update main text numbers and limitations after aggregation.

## 2. Manuscript Files

- [ ] Restore real author names and affiliations in `paper/main_is.tex`.
- [ ] Add acknowledgements if needed.
- [ ] Add funding statement if applicable.
- [ ] Add CRediT author contribution statement if the submission system asks.
- [ ] Add declaration of competing interest:
      `The authors declare no competing interests.`
- [ ] Add data/code availability statement with repository DOI.
- [ ] Decide whether to include an AI-use declaration. If included, keep it
      factual and minimal.
- [ ] Confirm abstract is no more than 250 words.
- [ ] Confirm keywords are 1-7 items.
- [ ] Keep `paper/highlights.txt` as a separate editable file:
  - [ ] 3-5 bullets;
  - [ ] each bullet no more than 85 characters;
  - [ ] no unexplained acronyms.
- [ ] Prepare a short cover letter.

## 3. Paper Polish

- [ ] Read abstract, intro, and conclusion back-to-back for one consistent story.
- [ ] Ensure terminology is fully consistent:
  - [ ] `OASIS-noProj`;
  - [ ] `OASIS`;
  - [ ] `Soft`;
  - [ ] `Hybrid`;
  - [ ] `Router`.
- [ ] Grep away old public-facing names:
  - [ ] `OASIS-Proj`;
  - [ ] `plain OASIS`;
  - [ ] `full OASIS`;
  - [ ] `Calibrated router`;
  - [ ] `OASIS-Soft`.
- [ ] Make the evaluation ladder explicit:
  - [ ] single-column;
  - [ ] Stage-2 variants;
  - [ ] OOD/DML/public trace realism;
  - [ ] composition/FactorJoin;
  - [ ] PostgreSQL planner;
  - [ ] TPC-H runtime sanity.
- [ ] Tighten every runtime sentence so it says sanity check, not benchmark.
- [ ] Tighten every public-trace sentence so it says telemetry/event-table case
      study, not DBMS production workload.
- [ ] Check that all tables mentioned in text are referenced in order.
- [ ] Check that every figure/table caption is understandable without reading
      the whole section.
- [ ] Make limitations crisp and non-defensive.

## 4. Supplement

- [ ] Ensure supplement contains enough protocol detail for every new result.
- [ ] Add final multi-seed TPC-H protocol details:
  - [ ] PostgreSQL version and config;
  - [ ] scale factor;
  - [ ] refresh/drift seed handling;
  - [ ] query templates;
  - [ ] timing protocol;
  - [ ] regression threshold definition.
- [ ] Keep public trace protocol explicit:
  - [ ] source URL;
  - [ ] parsed event count;
  - [ ] skipped lines;
  - [ ] window construction;
  - [ ] possible overlap caveat.
- [ ] Confirm supplementary material compiles independently.

## 5. Reproducibility Repository

- [ ] Create a clean public GitHub repository or release branch for submission.
- [ ] Remove private paths, secrets, machine-specific credentials, and SSH info.
- [ ] Add `README.md` with:
  - [ ] paper title and short claim;
  - [ ] directory layout;
  - [ ] expected hardware/software;
  - [ ] quick reproduction path;
  - [ ] full reproduction path.
- [ ] Add `REPRODUCE.md` mapping each paper table/figure to commands.
- [ ] Add `requirements.txt` or `environment.yml`.
- [ ] Add PostgreSQL/TPC-H setup instructions:
  - [ ] PostgreSQL version;
  - [ ] required extensions, if any;
  - [ ] TPC-H dbgen instructions;
  - [ ] why TPC-H data is generated rather than committed.
- [ ] Add NASA trace download/cache instructions and URL.
- [ ] Include pretrained `oasis_k16.json` or a documented download path.
- [ ] Include generated CSV/JSON/TEX result artifacts used by the paper.
- [ ] Add a license.
- [ ] Add citation metadata if useful (`CITATION.cff`).
- [ ] Create a release tag, e.g. `v1.0-information-systems-submission`.
- [ ] Archive the release on Zenodo, Mendeley Data, or OSF and get a DOI.
- [ ] Put the DOI in the paper's data/code availability statement.

## 6. Submission Package

- [ ] Main manuscript source (`.tex`) and compiled PDF.
- [ ] Bibliography (`.bib`).
- [ ] All figures as separate files with sufficient resolution.
- [ ] Generated table `.tex` files required by `\input`.
- [ ] Supplement source and compiled PDF.
- [ ] `paper/highlights.txt`.
- [ ] Cover letter.
- [ ] Declaration of interest file/text.
- [ ] Funding statement.
- [ ] Data/code availability statement.
- [ ] Author information:
  - [ ] emails;
  - [ ] affiliations;
  - [ ] ORCID IDs if available.
- [ ] Suggested reviewers, if the system asks.
- [ ] Opposed reviewers, if needed and justified.

## 7. Final Verification

- [ ] Run `tectonic main_is.tex`.
- [ ] Run `tectonic supplementary.tex`.
- [ ] Run `git diff --check`.
- [ ] Run Python syntax checks on new/changed experiment scripts.
- [ ] Confirm no unresolved citations or references.
- [ ] Confirm PDF first page has title, authors, abstract, keywords, and footer.
- [ ] Check `pdftotext` for stale contradictory phrases:
  - [ ] broad runtime improvement;
  - [ ] no runtime measured, except where a specific planner-only table says it;
  - [ ] old method names.
- [ ] Open PDFs and visually inspect:
  - [ ] first page;
  - [ ] all main tables;
  - [ ] TPC-H table;
  - [ ] public trace table;
  - [ ] references.
- [ ] Confirm repository DOI resolves.
- [ ] Confirm the manuscript cites the artifact DOI.

## 8. Cover Letter Points

- [ ] One-paragraph problem statement: stale optimizer marginals between
      `ANALYZE` refreshes.
- [ ] One-paragraph contribution: regime-aware feedback calibration layer.
- [ ] Emphasize fit to Information Systems:
  - [ ] data management;
  - [ ] real DBMS optimizer statistics;
  - [ ] systematic experiments;
  - [ ] reproducibility artifacts.
- [ ] Mention the bounded runtime sanity check without overselling it.
- [ ] Mention public artifact DOI.

## 9. Post-Submission

- [ ] Save the submitted PDF/source bundle.
- [ ] Save the Editorial Manager submission confirmation.
- [ ] Record manuscript ID in `progress.md`.
- [ ] Freeze the artifact release used for submission.
- [ ] Keep remote TPC-H/PostgreSQL notes for possible revision requests.
