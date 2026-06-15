\set ON_ERROR_STOP on
SET synchronous_commit = off;
SET maintenance_work_mem = '1GB';

TRUNCATE aka_name, aka_title, cast_info, char_name, comp_cast_type, company_name,
         company_type, complete_cast, info_type, keyword, kind_type, link_type,
         movie_companies, movie_info, movie_info_idx, movie_keyword, movie_link,
         name, person_info, role_type, title;

\echo === loading aka_name ===
\copy aka_name        FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/aka_name.dat'        WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading aka_title ===
\copy aka_title       FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/aka_title.dat'       WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading cast_info (big) ===
\copy cast_info       FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/cast_info.dat'       WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading char_name ===
\copy char_name       FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/char_name.dat'       WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading comp_cast_type ===
\copy comp_cast_type  FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/comp_cast_type.dat'  WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading company_name ===
\copy company_name    FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/company_name.dat'    WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading company_type ===
\copy company_type    FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/company_type.dat'    WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading complete_cast ===
\copy complete_cast   FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/complete_cast.dat'   WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading info_type ===
\copy info_type       FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/info_type.dat'       WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading keyword ===
\copy keyword         FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/keyword.dat'         WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading kind_type ===
\copy kind_type       FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/kind_type.dat'       WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading link_type ===
\copy link_type       FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/link_type.dat'       WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading movie_companies ===
\copy movie_companies FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/movie_companies.dat' WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading movie_info (big) ===
\copy movie_info      FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/movie_info.dat'      WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading movie_info_idx ===
\copy movie_info_idx  FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/movie_info_idx.dat'  WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading movie_keyword ===
\copy movie_keyword   FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/movie_keyword.dat'   WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading movie_link ===
\copy movie_link      FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/movie_link.dat'      WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading name ===
\copy name            FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/name.dat'            WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading person_info ===
\copy person_info     FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/person_info.dat'     WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading role_type ===
\copy role_type       FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/role_type.dat'       WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\echo === loading title (big) ===
\copy title           FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/title.dat'           WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')

\echo === ANALYZE ===
ANALYZE;
\echo === DONE ===
SELECT relname, n_live_tup FROM pg_stat_user_tables ORDER BY n_live_tup DESC;
