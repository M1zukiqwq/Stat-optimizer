-- Q7: Title with multiple joins
SELECT t.title, mi.info, mc.company_id 
FROM title t 
JOIN movie_info mi ON t.id = mi.movie_id 
JOIN movie_companies mc ON t.id = mc.movie_id 
WHERE t.production_year = 2008 
LIMIT 100
