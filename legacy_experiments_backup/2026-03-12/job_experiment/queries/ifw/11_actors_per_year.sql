SELECT t.production_year,
       COUNT(DISTINCT ci.person_id) as actor_count
FROM title t
JOIN cast_info ci ON t.id = ci.movie_id
WHERE t.production_year BETWEEN 1920 AND 2012
  AND ci.role_id = 1
GROUP BY t.production_year
ORDER BY t.production_year
