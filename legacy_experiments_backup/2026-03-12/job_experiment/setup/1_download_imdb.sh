#!/bin/bash
# 下载 IMDB 数据集

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/imdb_data"

echo "Downloading IMDB dataset for JOB benchmark..."

# 创建数据目录
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# 下载数据集（~3.6GB）
if [ ! -f "imdb.tgz" ]; then
    echo "Downloading imdb.tgz..."
    wget https://bonsai.cedardb.com/job/imdb.tgz
else
    echo "imdb.tgz already exists, skipping download"
fi

# 解压
if [ ! -d "imdb" ]; then
    echo "Extracting..."
    tar -xzf imdb.tgz
else
    echo "imdb directory already exists, skipping extraction"
fi

echo "✓ IMDB dataset ready at $DATA_DIR/imdb"
echo ""
echo "Dataset contains:"
ls -lh imdb/*.csv | awk '{print "  " $9 " (" $5 ")"}'
