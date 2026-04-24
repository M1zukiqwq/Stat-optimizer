#!/bin/bash
# 使用 Spark 加载 IMDB 数据到 Iceberg

export HADOOP_USER_NAME=tianqc

cd /home/tianqc/presto-optimizer/presto-cdf-simulation/job_experiment/setup

echo "启动 Spark 加载 IMDB 数据..."

spark-submit \
    --packages org.apache.iceberg:iceberg-spark-runtime-3.3_2.12:1.3.1 \
    --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
    --conf spark.sql.catalog.spark_catalog=org.apache.iceberg.spark.SparkSessionCatalog \
    --conf spark.sql.catalog.spark_catalog.type=hive \
    --conf spark.sql.catalog.hive_prod=org.apache.iceberg.spark.SparkCatalog \
    --conf spark.sql.catalog.hive_prod.type=hive \
    --conf spark.sql.catalog.hive_prod.uri=thrift://localhost:9083 \
    --driver-memory 4g \
    --executor-memory 8g \
    --executor-cores 4 \
    load_imdb_spark.py

echo "完成"
