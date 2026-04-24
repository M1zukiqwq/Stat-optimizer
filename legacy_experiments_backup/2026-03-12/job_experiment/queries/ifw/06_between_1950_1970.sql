SELECT COUNT(*) as movie_count,
       AVG(CAST(production_year AS DOUBLE)) as avg_year
FROM title
WHERE production_year > 1950 AND production_year < 1970
