\set d /home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe
\copy company_type FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/company_type.dat' WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
\copy company_name FROM '/home/tianqc/experiments/build/hard_generalization_20260531/job/imdb_pipe/company_name.dat' WITH (FORMAT csv, DELIMITER '|', NULL '', QUOTE e'\b', ESCAPE e'\b')
SELECT 'company_type' AS t, count(*) AS n FROM company_type
UNION ALL SELECT 'company_name', count(*) FROM company_name;
-- spot check a row with special chars
SELECT id, name, country_code FROM company_name WHERE name LIKE '%"%' LIMIT 2;
