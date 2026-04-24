-- Q3: Title join with movie_keyword
SELECT t.title, mk.keyword_id 
FROM title t 
JOIN movie_keyword mk ON t.id = mk.movie_id 
WHERE t.production_year BETWEEN 2000 AND 2010 
LIMIT 100
