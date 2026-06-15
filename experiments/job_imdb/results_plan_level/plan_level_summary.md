# OASIS JOB Plan-Level Rows-Hint Case Study

Generated on remote chen against PostgreSQL 14.17 / imdb. Hints use pg_hint_plan built from upstream PG14 branch.

## variants/local positive

Cases completed: 76; errors/timeouts: 4.

| query | granularity | local PG qerr | local OASIS qerr | default ms | OASIS ms | oracle ms | OASIS same oracle | note |
|---|---:|---:|---:|---:|---:|---:|---|---|
| query00964.sql | local | 61.14 | 3.51 | 1952.2 | 25.5 | 22.5 | False | OASIS 76.7x faster |
| query00285.sql | local | 30.57 | 1.08 | 3026.4 | 2759.9 | 2759.9 | True | plan changed |
| query00153.sql | local | 3.03 | 1.03 | 1839.4 | 1814.8 | 1814.8 | True | plan changed |
| query00384.sql | local | 3.01 | 1.04 | 1834.6 | 1833.3 | 1833.3 | True | plan changed |
| query00747.sql | local | 3.03 | 1.03 | 1813.9 | 1814.3 | 1814.3 | True | plan changed |
| query00615.sql | local | 3.01 | 1.04 | 1803.5 | 1805.9 | 1805.9 | True | plan changed |
| query00516.sql | local | 3.02 | 1.04 | 1808.9 | 1839.0 | 1839.0 | True | plan changed |
| query00772.sql | local | 51.79 | 1.93 | 339.9 | 361.1 | 361.1 | True | plan changed |
| query00706.sql | local | 3.99 | 1.12 | 956.1 | 2004.1 | 2004.1 | True | OASIS 2.1x slower |

## variants/local zero

Cases completed: 4; errors/timeouts: 0.

| query | granularity | local PG qerr | local OASIS qerr | default ms | OASIS ms | oracle ms | OASIS same oracle | note |
|---|---:|---:|---:|---:|---:|---:|---|---|
| query00508.sql | local | 32523000000000.00 | 1.00 | 3443.5 | 2877.1 | 2877.1 | True | OASIS 1.2x faster |

## variants/downstream branch

Cases completed: 76; errors/timeouts: 4.

| query | granularity | local PG qerr | local OASIS qerr | default ms | OASIS ms | oracle ms | OASIS same oracle | note |
|---|---:|---:|---:|---:|---:|---:|---|---|
| query00747.sql | branch | 3.03 | 1.03 | 2047.4 | 2047.4 | 231.8 | False | oracle 8.8x faster |
| query00384.sql | branch | 3.01 | 1.04 | 1978.2 | 1978.2 | 231.5 | False | oracle 8.5x faster |
| query00516.sql | branch | 3.02 | 1.04 | 1921.0 | 1921.0 | 230.6 | False | oracle 8.3x faster |
| query00615.sql | branch | 3.01 | 1.04 | 1815.2 | 1815.2 | 230.0 | False | oracle 7.9x faster |
| query00153.sql | branch | 3.03 | 1.03 | 1956.4 | 1956.4 | 324.9 | False | oracle 6.0x faster |
| query00219.sql | branch | 3.20 | 1.00 | 230.6 | 2088.1 | 232.1 | False | OASIS 9.1x slower |
| query00772.sql | branch | 51.79 | 1.93 | 244.2 | 763.4 | 338.5 | False | OASIS 3.1x slower |

## canonical/downstream branch

Cases completed: 5; errors/timeouts: 3.

No plan changes or >1.15x runtime changes.

