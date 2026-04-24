#!/bin/bash
# 使用 Spark 配置 Iceberg 表以支持行级别 DELETE/UPDATE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "配置 Iceberg 表以支持行级别 DELETE/UPDATE..."
echo ""

spark-submit \
    --packages org.apache.iceberg:iceberg-spark-runtime-3.3_2.12:1.3.1 \
    --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
    --conf spark.sql.catalog.spark_catalog=org.apache.iceberg.spark.SparkSessionCatalog \
    --conf spark.sql.catalog.spark_catalog.type=hive \
    --conf spark.sql.catalog.hive_prod=org.apache.iceberg.spark.SparkCatalog \
    --conf spark.sql.catalog.hive_prod.type=hive \
    --conf spark.sql.catalog.hive_prod.uri=thrift://localhost:9083 \
    --conf spark.sql.catalog.iceberg=org.apache.iceberg.spark.SparkCatalog \
    --conf spark.sql.catalog.iceberg.type=hive \
    --conf spark.sql.catalog.iceberg.uri=thrift://localhost:9083 \
    --driver-memory 2g \
    "$SCRIPT_DIR/enable_row_level_deletes.py"
