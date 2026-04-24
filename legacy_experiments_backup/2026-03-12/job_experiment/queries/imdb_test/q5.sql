-- Q5: Title join with movie_companies
SELECT t.title, mc.company_id 
FROM title t 
JOIN movie_companies mc ON t.id = mc.movie_id 
WHERE t.production_year > 2000 
LIMIT 100
