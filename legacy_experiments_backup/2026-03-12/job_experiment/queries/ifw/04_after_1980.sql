SELECT COUNT(*) as movie_count,
       AVG(CAST(production_year AS DOUBLE)) as avg_year,
       MIN(production_year) as min_year,
       MAX(production_year) as max_year
FROM title
WHERE production_year > 1980
