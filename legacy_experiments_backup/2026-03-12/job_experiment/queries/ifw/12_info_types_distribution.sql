SELECT mi.info_type_id,
       COUNT(*) as info_count
FROM title t
JOIN movie_info mi ON t.id = mi.movie_id
WHERE t.production_year BETWEEN 1920 AND 2012
GROUP BY mi.info_type_id
ORDER BY info_count DESC
LIMIT 10
