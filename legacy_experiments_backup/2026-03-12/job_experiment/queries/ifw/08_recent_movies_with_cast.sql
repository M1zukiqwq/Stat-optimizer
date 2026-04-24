SELECT COUNT(*) as movie_count,
       MIN(t.title) as sample_title
FROM title t
JOIN cast_info ci ON t.id = ci.movie_id
WHERE t.production_year BETWEEN 2000 AND 2012
  AND ci.role_id = 1
