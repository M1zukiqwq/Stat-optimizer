SELECT COUNT(*) as movie_count,
       MIN(title) as first_title,
       MAX(title) as last_title,
       AVG(CAST(production_year AS DOUBLE)) as avg_year
FROM title
WHERE production_year BETWEEN 1970 AND 1990
