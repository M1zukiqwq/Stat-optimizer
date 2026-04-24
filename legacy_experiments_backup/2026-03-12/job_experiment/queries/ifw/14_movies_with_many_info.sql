SELECT COUNT(*) as movie_count
FROM (
    SELECT t.id
    FROM title t
    JOIN movie_info mi ON t.id = mi.movie_id
    WHERE t.production_year BETWEEN 1920 AND 1950
    GROUP BY t.id
    HAVING COUNT(DISTINCT mi.info_type_id) > 5
) subq
