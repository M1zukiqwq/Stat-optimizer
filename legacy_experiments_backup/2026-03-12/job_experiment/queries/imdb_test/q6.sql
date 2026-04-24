-- Q6: Complex join with aggregation
SELECT 
    t.production_year,
    COUNT(DISTINCT t.id) as num_movies,
    COUNT(mk.keyword_id) as num_keywords
FROM title t 
JOIN movie_keyword mk ON t.id = mk.movie_id 
WHERE t.production_year BETWEEN 1990 AND 2010 
GROUP BY t.production_year
ORDER BY t.production_year
LIMIT 100
