\set ON_ERROR_STOP on
SET synchronous_commit = off;
DROP TABLE IF EXISTS title;
CREATE TABLE title (
    id integer, title text, imdb_index character varying(12), kind_id integer,
    production_year integer, imdb_id text, phonetic_code character varying(5),
    episode_of_id integer, season_nr integer, episode_nr integer,
    series_years character varying(49), md5sum character varying(32)
);
\echo === loading title ===
\copy title FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/title.dat' WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === ANALYZE (whole db) ===
ANALYZE;
\echo === row counts ===
SELECT relname, n_live_tup FROM pg_stat_user_tables ORDER BY n_live_tup DESC;
\echo === title sanity (production_year, kind_id) ===
SELECT min(production_year), max(production_year), count(distinct kind_id) FROM title;
