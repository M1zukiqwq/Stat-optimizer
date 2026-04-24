SELECT COUNT(*) as movie_count
FROM (
    SELECT t.id
    FROM title t
    JOIN cast_info ci ON t.id = ci.movie_id
    WHERE t.production_year BETWEEN 1920 AND 1950
      AND ci.role_id = 1
    GROUP BY t.id
    HAVING COUNT(DISTINCT ci.person_id) > 10
) subq
