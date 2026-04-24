#!/usr/bin/env python3
"""
使用 Spark 加载 IMDB 数据到 Iceberg 表
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import *
import os

# 数据目录
DATA_DIR = "/home/tianqc/presto-optimizer/presto-cdf-simulation/job_experiment/setup/imdb_data"

# 表结构定义
TABLE_SCHEMAS = {
    "title": StructType([
        StructField("id", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("imdb_index", StringType(), True),
        StructField("kind_id", IntegerType(), True),
        StructField("production_year", IntegerType(), True),
        StructField("imdb_id", IntegerType(), True),
        StructField("phonetic_code", StringType(), True),
        StructField("episode_of_id", IntegerType(), True),
        StructField("season_nr", IntegerType(), True),
        StructField("episode_nr", IntegerType(), True),
        StructField("series_years", StringType(), True),
        StructField("md5sum", StringType(), True),
    ]),
    "cast_info": StructType([
        StructField("id", IntegerType(), True),
        StructField("person_id", IntegerType(), True),
        StructField("movie_id", IntegerType(), True),
        StructField("person_role_id", IntegerType(), True),
        StructField("note", StringType(), True),
        StructField("nr_order", IntegerType(), True),
        StructField("role_id", IntegerType(), True),
    ]),
    "movie_info": StructType([
        StructField("id", IntegerType(), True),
        StructField("movie_id", IntegerType(), True),
        StructField("info_type_id", IntegerType(), True),
        StructField("info", StringType(), True),
        StructField("note", StringType(), True),
    ]),
    "movie_companies": StructType([
        StructField("id", IntegerType(), True),
        StructField("movie_id", IntegerType(), True),
        StructField("company_id", IntegerType(), True),
        StructField("company_type_id", IntegerType(), True),
        StructField("note", StringType(), True),
    ]),
    "movie_keyword": StructType([
        StructField("id", IntegerType(), True),
        StructField("movie_id", IntegerType(), True),
        StructField("keyword_id", IntegerType(), True),
    ]),
    "person_info": StructType([
        StructField("id", IntegerType(), True),
        StructField("person_id", IntegerType(), True),
        StructField("info_type_id", IntegerType(), True),
        StructField("info", StringType(), True),
        StructField("note", StringType(), True),
    ]),
    "movie_info_idx": StructType([
        StructField("id", IntegerType(), True),
        StructField("movie_id", IntegerType(), True),
        StructField("info_type_id", IntegerType(), True),
        StructField("info", StringType(), True),
        StructField("note", StringType(), True),
    ]),
    "aka_title": StructType([
        StructField("id", IntegerType(), True),
        StructField("movie_id", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("imdb_index", StringType(), True),
        StructField("kind_id", IntegerType(), True),
        StructField("production_year", IntegerType(), True),
        StructField("phonetic_code", StringType(), True),
        StructField("episode_of_id", IntegerType(), True),
        StructField("season_nr", IntegerType(), True),
        StructField("episode_nr", IntegerType(), True),
        StructField("note", StringType(), True),
        StructField("md5sum", StringType(), True),
    ]),
    "aka_name": StructType([
        StructField("id", IntegerType(), True),
        StructField("person_id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("imdb_index", StringType(), True),
        StructField("name_pcode_cf", StringType(), True),
        StructField("name_pcode_nf", StringType(), True),
        StructField("surname_pcode", StringType(), True),
        StructField("md5sum", StringType(), True),
    ]),
    "complete_cast": StructType([
        StructField("id", IntegerType(), True),
        StructField("movie_id", IntegerType(), True),
        StructField("subject_id", IntegerType(), True),
        StructField("status_id", IntegerType(), True),
    ]),
    "comp_cast_type": StructType([
        StructField("id", IntegerType(), True),
        StructField("kind", StringType(), True),
    ]),
    "company_name": StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("country_code", StringType(), True),
        StructField("imdb_id", IntegerType(), True),
        StructField("name_pcode_nf", StringType(), True),
        StructField("name_pcode_sf", StringType(), True),
        StructField("md5sum", StringType(), True),
    ]),
    "company_type": StructType([
        StructField("id", IntegerType(), True),
        StructField("kind", StringType(), True),
    ]),
    "info_type": StructType([
        StructField("id", IntegerType(), True),
        StructField("info", StringType(), True),
    ]),
    "keyword": StructType([
        StructField("id", IntegerType(), True),
        StructField("keyword", StringType(), True),
        StructField("phonetic_code", StringType(), True),
    ]),
    "kind_type": StructType([
        StructField("id", IntegerType(), True),
        StructField("kind", StringType(), True),
    ]),
    "link_type": StructType([
        StructField("id", IntegerType(), True),
        StructField("link", StringType(), True),
    ]),
    "name": StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("imdb_index", StringType(), True),
        StructField("imdb_id", IntegerType(), True),
        StructField("gender", StringType(), True),
        StructField("name_pcode_cf", StringType(), True),
        StructField("name_pcode_nf", StringType(), True),
        StructField("surname_pcode", StringType(), True),
        StructField("md5sum", StringType(), True),
    ]),
    "role_type": StructType([
        StructField("id", IntegerType(), True),
        StructField("role", StringType(), True),
    ]),
    "char_name": StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("imdb_index", StringType(), True),
        StructField("imdb_id", IntegerType(), True),
        StructField("name_pcode_nf", StringType(), True),
        StructField("surname_pcode", StringType(), True),
        StructField("md5sum", StringType(), True),
    ]),
    "movie_link": StructType([
        StructField("id", IntegerType(), True),
        StructField("movie_id", IntegerType(), True),
        StructField("linked_movie_id", IntegerType(), True),
        StructField("link_type_id", IntegerType(), True),
    ]),
}

# CSV 文件到表名的映射
TABLE_FILES = {k: f"{k}.csv" for k in TABLE_SCHEMAS.keys()}


def load_table(spark, table_name, csv_file, schema):
    """加载单个表"""
    csv_path = os.path.join(DATA_DIR, csv_file)
    
    if not os.path.exists(csv_path):
        print(f"  ✗ 文件不存在: {csv_path}")
        return False
    
    try:
        print(f"  读取 {csv_file}...", end="")
        
        # 使用 file:// 协议读取本地文件
        file_path = f"file://{csv_path}"
        
        # 读取 CSV（无表头，特殊字符处理）
        df = spark.read.csv(
            file_path,
            schema=schema,
            sep=",",
            encoding="UTF-8",
            quote='"',
            escape='\\',
            nullValue="",
            mode="PERMISSIVE"
        )
        
        count = df.count()
        print(f" {count} 行")
        
        # 写入 Iceberg 表
        print(f"  写入 iceberg.imdb.{table_name}...")
        
        # 先删除表（如果存在）
        spark.sql(f"DROP TABLE IF EXISTS hive_prod.imdb.{table_name}")
        
        # 创建新表（使用 CTAS 创建 v2 表）
        df.createOrReplaceTempView(f"temp_{table_name}")
        spark.sql(f"""
            CREATE TABLE hive_prod.imdb.{table_name}
            USING iceberg
            AS SELECT * FROM temp_{table_name}
        """)
        spark.catalog.dropTempView(f"temp_{table_name}")
        
        print(f"  ✓ {table_name} 完成")
        return True
        
    except Exception as e:
        print(f"\n  ✗ 错误: {e}")
        return False


def main():
    print("=" * 70)
    print("使用 Spark 加载 IMDB 数据到 Iceberg")
    print("=" * 70)
    
    # 创建 SparkSession（本地模式）
    spark = SparkSession.builder \
        .appName("Load IMDB to Iceberg") \
        .master("local[*]") \
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog") \
        .config("spark.sql.catalog.spark_catalog.type", "hive") \
        .config("spark.sql.catalog.hive_prod", "org.apache.iceberg.spark.SparkCatalog") \
        .config("spark.sql.catalog.hive_prod.type", "hive") \
        .config("spark.sql.catalog.hive_prod.uri", "thrift://localhost:9083") \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .getOrCreate()
    
    # 按顺序加载表（小表优先）
    load_order = [
        # 维度表（小表，先加载）
        "comp_cast_type", "company_type", "info_type", "kind_type", 
        "link_type", "role_type",
        # 中等表
        "keyword", "company_name", "char_name", "aka_name", 
        "complete_cast", "movie_link",
        # 大表（后加载）
        "name", "aka_title", "person_info", "movie_info_idx",
        "movie_companies", "movie_keyword", "movie_info", 
        "cast_info", "title",
    ]
    
    success_count = 0
    for table_name in load_order:
        if table_name not in TABLE_FILES:
            continue
            
        print(f"\n[{success_count+1}/{len(load_order)}] 加载 {table_name}...")
        csv_file = TABLE_FILES[table_name]
        schema = TABLE_SCHEMAS[table_name]
        
        if load_table(spark, table_name, csv_file, schema):
            success_count += 1
    
    print("\n" + "=" * 70)
    print(f"加载完成: {success_count}/{len(load_order)} 个表")
    print("=" * 70)
    
    spark.stop()


if __name__ == "__main__":
    main()
