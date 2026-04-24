#!/bin/bash
# PostgreSQL 实验快速启动脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PostgreSQL JOB Benchmark 实验启动脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查环境变量
if [ -z "$PG_HOME" ]; then
    echo -e "${RED}错误: PG_HOME 未设置${NC}"
    echo "请先设置环境变量："
    echo "  export PG_HOME=\$HOME/postgres_workspace/pg_install"
    exit 1
fi

if [ -z "$PG_DATA" ]; then
    echo -e "${RED}错误: PG_DATA 未设置${NC}"
    echo "请先设置环境变量："
    echo "  export PG_DATA=\$HOME/postgres_workspace/pg_data"
    exit 1
fi

# 检查 PostgreSQL 是否运行
if ! $PG_HOME/bin/pg_ctl -D $PG_DATA status > /dev/null 2>&1; then
    echo -e "${YELLOW}PostgreSQL 未运行，正在启动...${NC}"
    $PG_HOME/bin/pg_ctl -D $PG_DATA -l $PG_DATA/logfile start
    sleep 2
fi

echo -e "${GREEN}✓ PostgreSQL 运行中${NC}"

# 检查数据库是否存在
if ! $PG_HOME/bin/psql -lqt | cut -d \| -f 1 | grep -qw imdb; then
    echo -e "${YELLOW}数据库 'imdb' 不存在，正在创建...${NC}"
    $PG_HOME/bin/createdb imdb
    echo -e "${GREEN}✓ 数据库创建成功${NC}"
fi

# 检查 Python 虚拟环境
VENV_DIR="$HOME/postgres_workspace/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Python 虚拟环境不存在，正在创建...${NC}"
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip install psycopg2-binary numpy pandas
    echo -e "${GREEN}✓ 虚拟环境创建成功${NC}"
else
    source $VENV_DIR/bin/activate
    echo -e "${GREEN}✓ 虚拟环境已激活${NC}"
fi

# 检查查询目录
QUERY_DIR="$HOME/postgres_workspace/join-order-benchmark/queries"
if [ ! -d "$QUERY_DIR" ]; then
    echo -e "${RED}错误: 查询目录不存在: $QUERY_DIR${NC}"
    echo "请先下载 JOB 查询："
    echo "  cd ~/postgres_workspace"
    echo "  git clone https://github.com/gregrahn/join-order-benchmark.git"
    exit 1
fi

echo -e "${GREEN}✓ 查询目录存在: $QUERY_DIR${NC}"

# 显示菜单
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}请选择实验步骤：${NC}"
echo -e "${GREEN}========================================${NC}"
echo "1. 运行 Baseline 实验（无漂移）"
echo "2. 注入数据漂移（15 轮，2% 每轮）"
echo "3. 运行 Stale Prior 实验（有漂移，不更新统计）"
echo "4. 运行 Full ANALYZE 实验（有漂移，重新收集统计）"
echo "5. 分析结果"
echo "6. 查看数据库状态"
echo "7. 退出"
echo ""

read -p "请输入选项 (1-7): " choice

case $choice in
    1)
        echo -e "${GREEN}运行 Baseline 实验...${NC}"
        python run_postgres_experiment.py \
            --db-name imdb \
            --query-dir $QUERY_DIR \
            --strategy baseline \
            --output-dir results/baseline
        ;;
    2)
        echo -e "${YELLOW}警告: 此操作将修改数据库！${NC}"
        read -p "确认继续? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            echo -e "${GREEN}注入数据漂移...${NC}"
            python inject_drift_postgres.py \
                --db-name imdb \
                --drift-rounds 15 \
                --drift-ratio 0.02
        else
            echo "已取消"
        fi
        ;;
    3)
        echo -e "${GREEN}运行 Stale Prior 实验...${NC}"
        python run_postgres_experiment.py \
            --db-name imdb \
            --query-dir $QUERY_DIR \
            --strategy stale_prior \
            --output-dir results/stale_prior
        ;;
    4)
        echo -e "${GREEN}重新收集统计信息...${NC}"
        $PG_HOME/bin/psql -d imdb -c "ANALYZE VERBOSE;"
        echo -e "${GREEN}运行 Full ANALYZE 实验...${NC}"
        python run_postgres_experiment.py \
            --db-name imdb \
            --query-dir $QUERY_DIR \
            --strategy full_analyze \
            --output-dir results/full_analyze
        ;;
    5)
        echo -e "${GREEN}分析结果...${NC}"
        if [ ! -f "results/baseline/baseline_results.json" ]; then
            echo -e "${RED}错误: Baseline 结果不存在${NC}"
            exit 1
        fi
        if [ ! -f "results/stale_prior/stale_prior_results.json" ]; then
            echo -e "${RED}错误: Stale Prior 结果不存在${NC}"
            exit 1
        fi
        if [ ! -f "results/full_analyze/full_analyze_results.json" ]; then
            echo -e "${RED}错误: Full ANALYZE 结果不存在${NC}"
            exit 1
        fi
        python analyze_postgres_results.py \
            --baseline results/baseline/baseline_results.json \
            --stale results/stale_prior/stale_prior_results.json \
            --full-analyze results/full_analyze/full_analyze_results.json \
            --output results/comparison.json \
            --latex results/latex_tables.tex
        ;;
    6)
        echo -e "${GREEN}数据库状态：${NC}"
        $PG_HOME/bin/psql -d imdb -c "
        SELECT
            schemaname,
            tablename,
            n_live_tup as row_count,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        LIMIT 10;
        "
        ;;
    7)
        echo -e "${GREEN}退出${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}操作完成！${NC}"
echo -e "${GREEN}========================================${NC}"
