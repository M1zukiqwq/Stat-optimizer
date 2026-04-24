SELECT COUNT(DISTINCT ci.person_id) as unique_actors
FROM (
    SELECT id
    FROM title
    WHERE production_year BETWEEN 1920 AND 1950
) t
JOIN cast_info ci ON t.id = ci.movie_id
WHERE ci.role_id = 1
