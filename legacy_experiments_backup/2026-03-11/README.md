# Legacy Experiment Backup (2026-03-11)

These scripts were moved out of the active tree because they conflicted with the paper-facing experimental workflow.

Reasons for archival:
- multiple overlapping runners produced different numbers for the same paper tables;
- some scripts used hard-coded absolute paths to another repository;
- some plotting scripts hard-coded paper numbers instead of reading authoritative outputs;
- one JOB runner targeted Presto/Iceberg while neighboring docs/results targeted PostgreSQL;
- old automation no longer matched the current paper tables and figures.

Current cleanup direction:
- keep `experiments/mcv_adapter_validation.py` as the authoritative MCV validation path;
- replace the synthetic single-column runners with a unified paper-oriented suite;
- keep archived scripts here for auditability, but do not use them to update paper numbers.
