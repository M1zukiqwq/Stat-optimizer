SELECT COUNT(*) as movie_count,
       AVG(CAST(t.production_year AS DOUBLE)) as avg_year,
       COUNT(DISTINCT mi.info_type_id) as info_type_count,
       COUNT(DISTINCT ci.person_id) as actor_count
FROM title t
JOIN movie_info mi ON t.id = mi.movie_id
JOIN cast_info ci ON t.id = ci.movie_id
WHERE t.production_year BETWEEN 1920 AND 1950
  AND mi.info_type_id IN (1, 2, 3)
  AND ci.role_id = 1
