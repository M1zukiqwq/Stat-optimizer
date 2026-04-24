SELECT COUNT(*) as movie_count,
       MIN(t.title) as sample_title
FROM title t
JOIN movie_companies mc ON t.id = mc.movie_id
WHERE t.production_year BETWEEN 1970 AND 1990
  AND mc.company_type_id = 1
