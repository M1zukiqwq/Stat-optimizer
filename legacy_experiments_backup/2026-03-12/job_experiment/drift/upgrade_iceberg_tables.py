#!/usr/bin/env python3
"""
使用 Spark 升级 Iceberg 表到 v2 格式
支持行级别的 DELETE 和 UPDATE 操作
"""

from pyspark.sql import SparkSession
import sys

# 所有事实表
TABLES = [
    "cast_info",
    "movie_info",
    "movie_keyword",
    "name",
    "char_name",
    "person_info",
    "movie_companies",
    "title",
    "movie_info_idx",
    "aka_name",
    "aka_title",
    "complete_cast",
    "movie_link",
]

def create_spark_session():
    """创建 Spark Session"""
    return SparkSession.builder \
        .appName("Upgrade Iceberg Tables to v2") \
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog") \
        .config("spark.sql.catalog.spark_catalog.type", "hive") \
        .config("spark.sql.catalog.hive_prod", "org.apache.iceberg.spark.SparkCatalog") \
        .config("spark.sql.catalog.hive_prod.type", "hive") \
        .config("spark.sql.catalog.hive_prod.uri", "thrift://localhost:9083") \
        .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog") \
        .config("spark.sql.catalog.iceberg.type", "hive") \
        .config("spark.sql.catalog.iceberg.uri", "thrift://localhost:9083") \
        .enableHiveSupport() \
        .getOrCreate()

def upgrade_table(spark, catalog, schema, table_name):
    """升级单个表到 v2 格式"""
    try:
        # 检查当前版本
        current_props = spark.sql(f"SHOW TBLPROPERTIES {catalog}.{schema}.{table_name}").collect()
        current_version = None
        for row in current_props:
            if row.key == 'format-version':
                current_version = row.value
                break

        if current_version == '2':
            print(f"  ⊙ {table_name} 已经是 v2 格式，跳过")
            return True

        # 升级到 v2
        print(f"  ↑ 升级 {table_name} 从 v{current_version or '1'} 到 v2...")
        spark.sql(f"""
            ALTER TABLE {catalog}.{schema}.{table_name}
            SET TBLPROPERTIES ('format-version' = '2')
        """)

        # 验证升级
        new_props = spark.sql(f"SHOW TBLPROPERTIES {catalog}.{schema}.{table_name}").collect()
        new_version = None
        for row in new_props:
            if row.key == 'format-version':
                new_version = row.value
                break

        if new_version == '2':
            print(f"  ✓ {table_name} 升级成功 (v{current_version or '1'} → v2)")
            return True
        else:
            print(f"  ✗ {table_name} 升级失败 (当前版本: v{new_version})")
            return False

    except Exception as e:
        print(f"  ✗ {table_name} 升级失败: {e}")
        return False

def main():
    catalog = "iceberg"
    schema = "imdb"

    print("=" * 70)
    print("升级 Iceberg 表到 v2 格式 (使用 Spark)")
    print("=" * 70)
    print(f"Catalog: {catalog}")
    print(f"Schema: {schema}")
    print(f"Tables: {len(TABLES)}")
    print("=" * 70)
    print()

    # 创建 Spark Session
    print("初始化 Spark Session...")
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    print("✓ Spark Session 已创建")
    print()

    # 升级所有表
    success_count = 0
    fail_count = 0
    skip_count = 0

    for table_name in TABLES:
        result = upgrade_table(spark, catalog, schema, table_name)
        if result is True:
            success_count += 1
        elif result is None:
            skip_count += 1
        else:
            fail_count += 1
        print()

    # 输出总结
    print("=" * 70)
    print("升级完成")
    print("=" * 70)
    print(f"  成功: {success_count}/{len(TABLES)}")
    print(f"  失败: {fail_count}")
    print("=" * 70)

    spark.stop()

    if fail_count > 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
