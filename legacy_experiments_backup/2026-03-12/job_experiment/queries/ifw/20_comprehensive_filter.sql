SELECT COUNT(*) as result_count,
       AVG(CAST(t.production_year AS DOUBLE)) as avg_year,
       COUNT(DISTINCT mi.info_type_id) as info_type_count
FROM title t
JOIN movie_info mi ON t.id = mi.movie_id
JOIN cast_info ci ON t.id = ci.movie_id
WHERE (t.production_year BETWEEN 1920 AND 1950 OR t.production_year BETWEEN 2000 AND 2012)
  AND mi.info_type_id IN (1, 2, 3, 4, 5)
  AND ci.role_id IN (1, 2)
  AND t.kind_id = 1
