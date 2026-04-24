SELECT COUNT(*) as movie_count
FROM title t
JOIN movie_info mi ON t.id = mi.movie_id
WHERE t.production_year BETWEEN 2000 AND 2012
  AND mi.info_type_id = 101
  AND t.kind_id = 1
