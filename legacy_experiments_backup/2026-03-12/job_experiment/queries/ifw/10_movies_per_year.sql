SELECT production_year,
       COUNT(*) as movie_count
FROM title
WHERE production_year BETWEEN 1920 AND 2012
GROUP BY production_year
ORDER BY production_year
