#!/usr/bin/env python3
"""
生成 IMDB Filter Workload (IFW) 查询

这些查询专门设计用于测试直方图的准确性，与 JOB Benchmark 互补。
JOB 主要测试 JOIN（不使用直方图），IFW 主要测试 Filter（直接使用直方图）。
"""

import argparse
from pathlib import Path


# 查询模板
QUERIES = {
    # ========== 基础范围查询（Q1-Q6）==========
    "01_early_movies": """
SELECT COUNT(*) as movie_count,
       MIN(title) as first_title,
       MAX(title) as last_title,
       AVG(CAST(production_year AS DOUBLE)) as avg_year
FROM title
WHERE production_year BETWEEN 1920 AND 1950
""",

    "02_recent_movies": """
SELECT COUNT(*) as movie_count,
       MIN(title) as first_title,
       MAX(title) as last_title,
       AVG(CAST(production_year AS DOUBLE)) as avg_year
FROM title
WHERE production_year BETWEEN 2000 AND 2012
""",

    "03_middle_movies": """
SELECT COUNT(*) as movie_count,
       MIN(title) as first_title,
       MAX(title) as last_title,
       AVG(CAST(production_year AS DOUBLE)) as avg_year
FROM title
WHERE production_year BETWEEN 1970 AND 1990
""",

    "04_after_1980": """
SELECT COUNT(*) as movie_count,
       AVG(CAST(production_year AS DOUBLE)) as avg_year,
       MIN(production_year) as min_year,
       MAX(production_year) as max_year
FROM title
WHERE production_year > 1980
""",

    "05_before_1960": """
SELECT COUNT(*) as movie_count,
       AVG(CAST(production_year AS DOUBLE)) as avg_year,
       MIN(production_year) as min_year,
       MAX(production_year) as max_year
FROM title
WHERE production_year < 1960
""",

    "06_between_1950_1970": """
SELECT COUNT(*) as movie_count,
       AVG(CAST(production_year AS DOUBLE)) as avg_year
FROM title
WHERE production_year > 1950 AND production_year < 1970
""",

    # ========== 多列 Filter（Q7-Q9）==========
    "07_early_movies_with_info": """
-- Q7: 早期电影 + 特定信息类型
-- 测试两个直方图的联合估算
SELECT COUNT(*) as result_count,
       MIN(t.title) as sample_title
FROM title t
JOIN movie_info mi ON t.id = mi.movie_id
WHERE t.production_year BETWEEN 1920 AND 1950
  AND mi.info_type_id IN (1, 2, 3);
""",

    "08_recent_movies_with_cast": """
-- Q8: 近期电影 + 主演角色
-- 测试三列 Filter
SELECT COUNT(*) as result_count
FROM title t
JOIN cast_info ci ON t.id = ci.movie_id
WHERE t.production_year > 1990
  AND ci.role_id = 1
  AND t.kind_id = 1;
""",

    "09_middle_movies_with_companies": """
-- Q9: 中期电影 + 制作公司
-- 测试 Filter + JOIN 的组合
SELECT COUNT(*) as result_count,
       COUNT(DISTINCT mc.company_id) as company_count
FROM title t
JOIN movie_companies mc ON t.id = mc.movie_id
WHERE t.production_year BETWEEN 1970 AND 1990
  AND mc.company_type_id = 1;
""",

    # ========== 聚合查询（Q10-Q12）==========
    "10_movies_per_year": """
-- Q10: 每年的电影数量（早期）
-- 测试 Filter 后的分组聚合
SELECT production_year,
       COUNT(*) as movie_count
FROM title
WHERE production_year BETWEEN 1920 AND 1950
GROUP BY production_year
HAVING COUNT(*) > 10
ORDER BY production_year;
""",

    "11_actors_per_year": """
-- Q11: 每年的演员数量（近期）
-- 测试多表聚合
SELECT t.production_year,
       COUNT(DISTINCT ci.person_id) as actor_count
FROM title t
JOIN cast_info ci ON t.id = ci.movie_id
WHERE t.production_year BETWEEN 1990 AND 2010
  AND ci.role_id = 1
GROUP BY t.production_year
ORDER BY t.production_year;
""",

    "12_info_types_distribution": """
-- Q12: 信息类型分布（早期电影）
-- 测试 Filter + 聚合
SELECT mi.info_type_id,
       COUNT(*) as info_count,
       COUNT(DISTINCT mi.movie_id) as movie_count
FROM movie_info mi
JOIN title t ON mi.movie_id = t.id
WHERE t.production_year BETWEEN 1920 AND 1950
GROUP BY mi.info_type_id
HAVING COUNT(*) > 100
ORDER BY info_count DESC;
""",

    # ========== 子查询（Q13-Q15）==========
    "13_cast_of_early_movies": """
-- Q13: 早期电影的演员
-- 测试 IN 子查询中的 Filter
SELECT COUNT(*) as cast_count,
       COUNT(DISTINCT person_id) as unique_actors
FROM cast_info
WHERE movie_id IN (
    SELECT id FROM title
    WHERE production_year BETWEEN 1920 AND 1950
);
""",

    "14_movies_with_many_info": """
-- Q14: 有大量信息的电影
-- 测试 EXISTS 子查询
SELECT COUNT(*) as movie_count
FROM title t
WHERE production_year > 1990
  AND EXISTS (
    SELECT 1 FROM movie_info mi
    WHERE mi.movie_id = t.id
      AND mi.info_type_id IN (1, 2, 3, 4, 5)
    GROUP BY mi.movie_id
    HAVING COUNT(*) > 5
  );
""",

    "15_popular_early_movies": """
-- Q15: 受欢迎的早期电影
-- 测试复杂子查询
SELECT COUNT(*) as movie_count,
       AVG(CAST(t.production_year AS DOUBLE)) as avg_year
FROM title t
WHERE t.production_year BETWEEN 1920 AND 1950
  AND t.id IN (
    SELECT movie_id FROM cast_info
    GROUP BY movie_id
    HAVING COUNT(DISTINCT person_id) > 20
  );
""",

    # ========== 复杂查询（Q16-Q20）==========
    "16_early_movies_full_analysis": """
-- Q16: 早期电影的完整分析
-- 混合 Filter、JOIN、聚合
SELECT t.production_year,
       COUNT(DISTINCT t.id) as movie_count,
       COUNT(DISTINCT ci.person_id) as actor_count,
       COUNT(DISTINCT mc.company_id) as company_count
FROM title t
LEFT JOIN cast_info ci ON t.id = ci.movie_id AND ci.role_id = 1
LEFT JOIN movie_companies mc ON t.id = mc.movie_id
WHERE t.production_year BETWEEN 1920 AND 1950
GROUP BY t.production_year
ORDER BY t.production_year;
""",

    "17_recent_movies_with_ratings": """
-- Q17: 近期高评分电影
-- 测试多个 Filter 条件
SELECT COUNT(*) as movie_count,
       AVG(CAST(t.production_year AS DOUBLE)) as avg_year
FROM title t
JOIN movie_info_idx mii ON t.id = mii.movie_id
WHERE t.production_year BETWEEN 2000 AND 2012
  AND mii.info_type_id = 101  -- ratings
  AND t.kind_id = 1;
""",

    "18_decade_comparison": """
-- Q18: 不同年代的对比
-- 测试多个范围的 UNION
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
WHERE production_year BETWEEN 2000 AND 2012;
""",

    "19_movies_with_multiple_criteria": """
-- Q19: 多条件筛选
-- 测试复杂的 Filter 组合
SELECT COUNT(*) as movie_count,
       MIN(t.title) as sample_title
FROM title t
JOIN movie_info mi ON t.id = mi.movie_id
JOIN cast_info ci ON t.id = ci.movie_id
WHERE t.production_year BETWEEN 1920 AND 1950
  AND mi.info_type_id IN (1, 2, 3)
  AND ci.role_id = 1
  AND t.kind_id = 1
GROUP BY t.id
HAVING COUNT(DISTINCT ci.person_id) > 5;
""",

    "20_comprehensive_filter": """
-- Q20: 综合 Filter 测试
-- 测试所有漂移列的组合
SELECT COUNT(*) as result_count,
       AVG(CAST(t.production_year AS DOUBLE)) as avg_year,
       COUNT(DISTINCT mi.info_type_id) as info_type_count
FROM title t
JOIN movie_info mi ON t.id = mi.movie_id
JOIN cast_info ci ON t.id = ci.movie_id
WHERE (t.production_year BETWEEN 1920 AND 1950 OR t.production_year BETWEEN 2000 AND 2012)
  AND mi.info_type_id IN (1, 2, 3, 4, 5)
  AND ci.role_id IN (1, 2)
  AND t.kind_id = 1;
""",
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

    # 生成 README
    readme_file = output_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write("# IMDB Filter Workload (IFW)\n\n")
        f.write("这些查询专门设计用于测试直方图的准确性。\n\n")
        f.write("## 查询分类\n\n")
        f.write("- **Q1-Q3**: 基础范围查询（单表、单列）\n")
        f.write("- **Q4-Q6**: 不等式查询（测试 CDF）\n")
        f.write("- **Q7-Q9**: 多列 Filter（测试联合估算）\n")
        f.write("- **Q10-Q12**: 聚合查询（测试 Filter + 聚合）\n")
        f.write("- **Q13-Q15**: 子查询（测试嵌套 Filter）\n")
        f.write("- **Q16-Q20**: 复杂查询（混合场景）\n\n")
        f.write("## 使用方法\n\n")
        f.write("```bash\n")
        f.write("# 运行所有查询\n")
        f.write("python3 run_experiment.py \\\n")
        f.write("  --presto-host localhost:8080 \\\n")
        f.write("  --query-dir queries/ifw \\\n")
        f.write("  --strategy stale_prior \\\n")
        f.write("  --output-dir results/ifw_stale_prior\n")
        f.write("```\n")

    print(f"  ✓ README.md")


def main():
    parser = argparse.ArgumentParser(description='生成 IMDB Filter Workload 查询')
    parser.add_argument('--output', default='queries/ifw',
                       help='输出目录（默认: queries/ifw）')

    args = parser.parse_args()

    output_dir = Path(args.output)
    generate_queries(output_dir)


if __name__ == '__main__':
    main()
