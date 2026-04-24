-- Q10: Three way join with filter
SELECT t.title, mi.info, mk.keyword_id, mc.company_id 
FROM title t 
JOIN movie_info mi ON t.id = mi.movie_id 
JOIN movie_keyword mk ON t.id = mk.movie_id 
JOIN movie_companies mc ON t.id = mc.movie_id 
WHERE t.production_year = 2010 
LIMIT 50
