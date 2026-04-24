-- Q8: Simple aggregation
SELECT production_year, COUNT(*) as cnt 
FROM title 
WHERE production_year IS NOT NULL 
GROUP BY production_year 
ORDER BY production_year DESC 
LIMIT 100
