# OASIS JOB Plan-Level Rows-Hint Case Study

Generated on remote chen against PostgreSQL 14.17 / imdb. Hints use pg_hint_plan built from upstream PG14 branch.

## Scan Scope

| item | count |
|---|---:|
| files_scanned | 974 |
| title_kind_candidates | 210 |
| movie_companies_candidates | 300 |
| aka_title_candidates | 0 |
| cast_info_nr_order_role_candidates | 0 |

## Clean Positive Cases

| query | pair | aliases | predicates | PG rows | OASIS rows | actual rows | PG qerr | OASIS qerr | default ms | OASIS ms | oracle ms | speedup | same oracle |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| query00964.sql | title.production_year/kind_id | kt,t | t.production_year > 2014 AND kt.kind IN ('movie') | 7.0 | 122.0 | 428.0 | 61.14 | 3.51 | 1952.2 | 25.5 | 22.5 | 76.7x | False |

## Null / Boundary / Negative Cases

Total completed evaluations across batches: 236; errors/timeouts: 17.

| query | pair | PG qerr | OASIS qerr | default ms | OASIS ms | oracle ms | same default | same oracle | note |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| query00550.sql | title.production_year/kind_id | 1.50 | 1.30 | 85.3 | 1572.8 | 1572.8 | False | True | OASIS slower |
| query00219.sql | title.production_year/kind_id | 3.20 | 1.00 | 230.6 | 2088.1 | 232.1 | False | False | OASIS slower |
| query00747.sql | title.production_year/kind_id | 3.03 | 1.03 | 2047.4 | 2047.4 | 231.8 | True | False | oracle-only faster |
| query00384.sql | title.production_year/kind_id | 3.01 | 1.04 | 1978.2 | 1978.2 | 231.5 | True | False | oracle-only faster |
| query00516.sql | title.production_year/kind_id | 3.02 | 1.04 | 1921.0 | 1921.0 | 230.6 | True | False | oracle-only faster |
| query00615.sql | title.production_year/kind_id | 3.01 | 1.04 | 1815.2 | 1815.2 | 230.0 | True | False | oracle-only faster |
| query00153.sql | title.production_year/kind_id | 3.03 | 1.03 | 1956.4 | 1956.4 | 324.9 | True | False | oracle-only faster |
| query00772.sql | title.production_year/kind_id | 51.79 | 1.93 | 244.2 | 763.4 | 338.5 | False | False | OASIS slower |
| query00706.sql | title.production_year/kind_id | 3.99 | 1.12 | 956.1 | 2004.1 | 2004.1 | False | True | OASIS slower |
| query00508.sql | title.production_year/kind_id | 32523000000000.00 | 1.00 | 3443.5 | 2877.1 | 2877.1 | False | True | positive but below/duplicate threshold |
| query00286.sql | title.production_year/kind_id | 1.35 | 1.41 | 1217.6 | 1421.3 | 1217.6 | False | False | OASIS slower |
| query00963.sql | title.production_year/kind_id | 2.99 | 1.05 | 1998.6 | 1807.7 | 1807.7 | False | True | plan changed, little runtime effect |
| query00285.sql | title.production_year/kind_id | 30.57 | 1.08 | 3026.4 | 2759.9 | 2759.9 | False | True | plan changed, little runtime effect |
| query00549.sql | title.production_year/kind_id | 3.00 | 1.05 | 1928.1 | 1812.9 | 1812.9 | False | True | plan changed, little runtime effect |
| query00772.sql | title.production_year/kind_id | 51.79 | 1.93 | 339.9 | 361.1 | 361.1 | False | True | plan changed, little runtime effect |
| query00516.sql | title.production_year/kind_id | 3.02 | 1.04 | 1808.9 | 1839.0 | 1839.0 | False | True | plan changed, little runtime effect |
| query00054.sql | title.production_year/kind_id | 2.99 | 1.05 | 1804.5 | 1832.8 | 1832.8 | False | True | plan changed, little runtime effect |
| query00153.sql | title.production_year/kind_id | 3.03 | 1.03 | 1839.4 | 1814.8 | 1814.8 | False | True | plan changed, little runtime effect |

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

## variants/title remaining local

Cases completed: 56; errors/timeouts: 3.

| query | granularity | local PG qerr | local OASIS qerr | default ms | OASIS ms | oracle ms | OASIS same oracle | note |
|---|---:|---:|---:|---:|---:|---:|---|---|
| query00963.sql | local | 2.99 | 1.05 | 1998.6 | 1807.7 | 1807.7 | True | plan changed |
| query00549.sql | local | 3.00 | 1.05 | 1928.1 | 1812.9 | 1812.9 | True | plan changed |
| query00681.sql | local | 2.99 | 1.05 | 1825.0 | 1802.8 | 1802.8 | True | plan changed |
| query00811.sql | local | 2.99 | 1.05 | 1810.1 | 1808.7 | 1808.7 | True | plan changed |
| query00286.sql | local | 1.35 | 1.41 | 1217.6 | 1421.3 | 1217.6 | False | OASIS 1.2x slower |
| query00087.sql | local | 2.99 | 1.04 | 1808.0 | 1816.4 | 1816.4 | True | plan changed |
| query00714.sql | local | 3.00 | 1.05 | 1804.1 | 1813.8 | 1813.8 | True | plan changed |
| query00186.sql | local | 3.00 | 1.05 | 1944.8 | 1966.2 | 1966.2 | True | plan changed |
| query00054.sql | local | 2.99 | 1.05 | 1804.5 | 1832.8 | 1832.8 | True | plan changed |
| query00550.sql | local | 1.50 | 1.30 | 85.3 | 1572.8 | 1572.8 | True | OASIS 18.4x slower |

## variants/movie_companies improved local

Cases completed: 19; errors/timeouts: 3.

No plan changes or >1.15x runtime changes.

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

