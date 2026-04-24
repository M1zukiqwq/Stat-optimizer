#!/usr/bin/env python3
"""生成 IMDB Filter Workload (IFW) 查询（无注释版本）"""

import argparse
from pathlib import Path

# 查询模板（移除所有注释和分号）
QUERIES = {
    "01_early_movies": """SELECT COUNT(*) as movie_count,
       MIN(title) as first_title,
       MAX(title) as last_title,
       AVG(CAST(production_year AS DOUBLE)) as avg_year
FROM title
WHERE production_year BETWEEN 1920 AND 1950""",

    "02_recent_movies": """SELECT COUNT(*) as movie_count,
       MIN(title) as first_title,
       MAX(title) as last_title,
       AVG(CAST(production_year AS DOUBLE)) as avg_year
FROM title
WHERE production_year BETWEEN 2000 AND 2012""",

    "03_middle_movies": """SELECT COUNT(*) as movie_count,
       MIN(title) as first_title,
       MAX(title) as last_title,
       AVG(CAST(production_year AS DOUBLE)) as avg_year
FROM title
WHERE production_year BETWEEN 1970 AND 1990""",

    "04_after_1980": """SELECT COUNT(*) as movie_count,
       AVG(CAST(production_year AS DOUBLE)) as avg_year,
       MIN(production_year) as min_year,
       MAX(production_year) as max_year
FROM title
WHERE production_year > 1980""",

    "05_before_1960": """SELECT COUNT(*) as movie_count,
       AVG(CAST(production_year AS DOUBLE)) as avg_year,
       MIN(production_year) as min_year,
       MAX(production_year) as max_year
FROM title
WHERE production_year < 1960""",

    "06_between_1950_1970": """SELECT COUNT(*) as movie_count,
       AVG(CAST(production_year AS DOUBLE)) as avg_year
FROM title
WHERE production_year > 1950 AND production_year < 1970""",

    "07_early_movies_with_info": """SELECT COUNT(*) as movie_count,
       MIN(t.title) as sample_title
FROM title t
JOIN movie_info mi ON t.id = mi.movie_id
WHERE t.production_year BETWEEN 1920 AND 1950
  AND mi.info_type_id IN (1, 2, 3)""",

    "08_recent_movies_with_cast": """SELECT COUNT(*) as movie_count,
       MIN(t.title) as sample_title
FROM title t
JOIN cast_info ci ON t.id = ci.movie_id
WHERE t.production_year BETWEEN 2000 AND 2012
  AND ci.role_id = 1""",

    "09_middle_movies_with_companies": """SELECT COUNT(*) as movie_count,
       MIN(t.title) as sample_title
FROM title t
JOIN movie_companies mc ON t.id = mc.movie_id
WHERE t.production_year BETWEEN 1970 AND 1990
  AND mc.company_type_id = 1""",

    "10_movies_per_year": """SELECT production_year,
       COUNT(*) as movie_count
FROM title
WHERE production_year BETWEEN 1920 AND 2012
GROUP BY production_year
ORDER BY production_year""",

    "11_actors_per_year": """SELECT t.production_year,
       COUNT(DISTINCT ci.person_id) as actor_count
FROM title t
JOIN cast_info ci ON t.id = ci.movie_id
WHERE t.production_year BETWEEN 1920 AND 2012
  AND ci.role_id = 1
GROUP BY t.production_year
ORDER BY t.production_year""",

    "12_info_types_distribution": """SELECT mi.info_type_id,
       COUNT(*) as info_count
FROM title t
JOIN movie_info mi ON t.id = mi.movie_id
WHERE t.production_year BETWEEN 1920 AND 2012
GROUP BY mi.info_type_id
ORDER BY info_count DESC
LIMIT 10""",

    "13_cast_of_early_movies": """SELECT COUNT(DISTINCT ci.person_id) as unique_actors
FROM (
    SELECT id
    FROM title
    WHERE production_year BETWEEN 1920 AND 1950
) t
JOIN cast_info ci ON t.id = ci.movie_id
WHERE ci.role_id = 1""",

    "14_movies_with_many_info": """SELECT COUNT(*) as movie_count
FROM (
    SELECT t.id
    FROM title t
    JOIN movie_info mi ON t.id = mi.movie_id
    WHERE t.production_year BETWEEN 1920 AND 1950
    GROUP BY t.id
    HAVING COUNT(DISTINCT mi.info_type_id) > 5
) subq""",

    "15_popular_early_movies": """SELECT COUNT(*) as movie_count
FROM (
    SELECT t.id
    FROM title t
    JOIN cast_info ci ON t.id = ci.movie_id
    WHERE t.production_year BETWEEN 1920 AND 1950
      AND ci.role_id = 1
    GROUP BY t.id
    HAVING COUNT(DISTINCT ci.person_id) > 10
) subq""",

    "16_early_movies_full_analysis": """SELECT COUNT(*) as movie_count,
       AVG(CAST(t.production_year AS DOUBLE)) as avg_year,
       COUNT(DISTINCT mi.info_type_id) as info_type_count,
       COUNT(DISTINCT ci.person_id) as actor_count
FROM title t
JOIN movie_info mi ON t.id = mi.movie_id
JOIN cast_info ci ON t.id = ci.movie_id
WHERE t.production_year BETWEEN 1920 AND 1950
  AND mi.info_type_id IN (1, 2, 3)
  AND ci.role_id = 1""",

    "17_recent_movies_with_ratings": """SELECT COUNT(*) as movie_count
FROM title t
JOIN movie_info mi ON t.id = mi.movie_id
WHERE t.production_year BETWEEN 2000 AND 2012
  AND mi.info_type_id = 101
  AND t.kind_id = 1""",

    "18_decade_comparison": """SELECT '1920-1950' as decade,
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
WHERE production_year BETWEEN 2000 AND 2012""",

    "19_movies_with_multiple_criteria": """SELECT COUNT(*) as movie_count,
       MIN(t.title) as sample_title
FROM title t
JOIN movie_info mi ON t.id = mi.movie_id
JOIN cast_info ci ON t.id = ci.movie_id
WHERE t.production_year BETWEEN 1920 AND 1950
  AND mi.info_type_id IN (1, 2, 3)
  AND ci.role_id = 1
  AND t.kind_id = 1
GROUP BY t.id
HAVING COUNT(DISTINCT ci.person_id) > 5""",

    "20_comprehensive_filter": """SELECT COUNT(*) as result_count,
       AVG(CAST(t.production_year AS DOUBLE)) as avg_year,
       COUNT(DISTINCT mi.info_type_id) as info_type_count
FROM title t
JOIN movie_info mi ON t.id = mi.movie_id
JOIN cast_info ci ON t.id = ci.movie_id
WHERE (t.production_year BETWEEN 1920 AND 1950 OR t.production_year BETWEEN 2000 AND 2012)
  AND mi.info_type_id IN (1, 2, 3, 4, 5)
  AND ci.role_id IN (1, 2)
  AND t.kind_id = 1""",
}


def generate_queries(output_dir: Path):
    """生成所有查询文件"""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"生成 IFW 查询到: {output_dir}")
    print(f"总共 {len(QUERIES)} 个查询\n")

    for query_id, query_sql in QUERIES.items():
        output_file = output_dir / f"{query_id}.sql"
        with open(output_file, 'w') as f:
            f.write(query_sql.strip() + '\n')
        print(f"  ✓ {query_id}.sql")

    print(f"\n✓ 完成！生成了 {len(QUERIES)} 个查询文件")


def main():
    parser = argparse.ArgumentParser(description='生成 IMDB Filter Workload 查询')
    parser.add_argument('--output', default='queries/ifw',
                       help='输出目录（默认: queries/ifw）')

    args = parser.parse_args()

    output_dir = Path(args.output)
    generate_queries(output_dir)


if __name__ == '__main__':
    main()
