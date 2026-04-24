SELECT '1920-1950' as decade,
       COUNT(*) as movie_count,
       COUNT(DISTINCT kind_id) as kind_count
FROM title
WHERE production_year BETWEEN 1920 AND 1950
UNION ALL
SELECT '1970-1990' as decade,
       COUNT(*) as movie_count,
       COUNT(DISTINCT kind_id) as kind_count
FROM title
WHERE production_year BETWEEN 1970 AND 1990
UNION ALL
SELECT '2000-2012' as decade,
       COUNT(*) as movie_count,
       COUNT(DISTINCT kind_id) as kind_count
FROM title
WHERE production_year BETWEEN 2000 AND 2012
