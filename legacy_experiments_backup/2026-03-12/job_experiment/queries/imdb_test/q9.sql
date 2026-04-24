-- Q9: Subquery with join
SELECT t.title, t.production_year 
FROM title t 
WHERE t.id IN (
    SELECT movie_id FROM movie_info WHERE info_type_id = 1
) 
AND t.production_year > 2000 
LIMIT 100
