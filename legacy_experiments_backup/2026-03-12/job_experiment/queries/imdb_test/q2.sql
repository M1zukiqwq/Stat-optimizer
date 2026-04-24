-- Q2: Title join with movie_info
SELECT t.title, mi.info 
FROM title t 
JOIN movie_info mi ON t.id = mi.movie_id 
WHERE t.production_year = 2005 
LIMIT 100
