SELECT COUNT(*) as movie_count,
       MIN(t.title) as sample_title
FROM title t
JOIN movie_info mi ON t.id = mi.movie_id
JOIN cast_info ci ON t.id = ci.movie_id
WHERE t.production_year BETWEEN 1920 AND 1950
  AND mi.info_type_id IN (1, 2, 3)
  AND ci.role_id = 1
  AND t.kind_id = 1
GROUP BY t.id
HAVING COUNT(DISTINCT ci.person_id) > 5
