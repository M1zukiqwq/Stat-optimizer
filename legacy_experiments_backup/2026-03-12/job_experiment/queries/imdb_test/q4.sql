-- Q4: Multi-table join
SELECT t.title, mi.info, mk.keyword_id 
FROM title t 
JOIN movie_info mi ON t.id = mi.movie_id 
JOIN movie_keyword mk ON t.id = mk.movie_id 
WHERE t.production_year = 2005 
LIMIT 100
