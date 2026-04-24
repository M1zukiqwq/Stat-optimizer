#!/bin/bash
# ANALYZE 辅助脚本 - 调试版本
# 添加详细的调试输出

# 配置
PRESTO_CLI="${PRESTO_CLI:-/home/tianqc/presto-server-0.296/presto-cli}"
HOST="localhost:8080"
CATALOG="iceberg"
SCHEMA="imdb"
KLL_PARAMETER=1024

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 表列表
ALL_FACT_TABLES=(
    "cast_info"
    "movie_info"
    "movie_keyword"
    "name"
    "char_name"
    "person_info"
    "movie_companies"
    "title"
    "movie_info_idx"
    "aka_name"
    "aka_title"
    "complete_cast"
    "movie_link"
)

CORE_TABLES=(
    "title"
    "cast_info"
    "movie_info"
    "movie_companies"
    "name"
    "movie_keyword"
)

# 执行 ANALYZE
analyze_table() {
    local table="$1"
    local index="$2"
    local total="$3"

    echo ""
    echo -e "${BLUE}[$index/$total]${NC} ${GREEN}ANALYZE $table${NC}"
    echo "  执行命令: $PRESTO_CLI --server $HOST --catalog $CATALOG --schema $SCHEMA"
    echo "  SQL: SET SESSION iceberg.statistics_kll_sketch_k_parameter = $KLL_PARAMETER; ANALYZE $table;"

    local start_time=$(date +%s)

    result=$($PRESTO_CLI --server $HOST --catalog $CATALOG --schema $SCHEMA \
        --execute "SET SESSION iceberg.statistics_kll_sketch_k_parameter = $KLL_PARAMETER; ANALYZE $table;" 2>&1)

    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo "  退出码: $exit_code"
    echo "  耗时: ${duration}s"

    # 显示输出（前5行）
    if [ -n "$result" ]; then
        echo "  输出:"
        echo "$result" | head -5 | sed 's/^/    /'
    fi

    # 检查是否成功
    if [ $exit_code -eq 0 ] && ! echo "$result" | grep -qi "error\|failed\|exception"; then
        echo -e "  ${GREEN}✓ 成功${NC}"
        return 0
    else
        echo -e "  ${YELLOW}✗ 失败${NC}"
        return 1
    fi
}

# 主函数
main() {
    local mode="${1:-all}"

    # 选择表列表
    local tables=()
    case $mode in
        --all|all)
            tables=("${ALL_FACT_TABLES[@]}")
            echo "=========================================="
            echo "分析所有 13 张事实表（调试模式）"
            echo "=========================================="
            ;;
        --core|core)
            tables=("${CORE_TABLES[@]}")
            echo "=========================================="
            echo "分析核心 6 张表（调试模式）"
            echo "=========================================="
            ;;
        *)
            echo "用法: $0 [--all|--core]"
            exit 1
            ;;
    esac

    echo "配置:"
    echo "  Presto CLI: $PRESTO_CLI"
    echo "  Presto: $HOST"
    echo "  Catalog: $CATALOG"
    echo "  Schema: $SCHEMA"
    echo "  KLL 参数: $KLL_PARAMETER"
    echo "  表数量: ${#tables[@]}"
    echo "=========================================="

    # 检查 Presto CLI 是否存在
    if [ ! -f "$PRESTO_CLI" ]; then
        echo -e "${YELLOW}警告: Presto CLI 不存在: $PRESTO_CLI${NC}"
        echo "请设置正确的 PRESTO_CLI 环境变量"
        exit 1
    fi

    # 执行 ANALYZE
    local success_count=0
    local fail_count=0
    local start_time=$(date +%s)
    local index=1

    for table in "${tables[@]}"; do
        echo ""
        echo "========================================"

        if analyze_table "$table" "$index" "${#tables[@]}"; then
            success_count=$((success_count + 1))
        else
            fail_count=$((fail_count + 1))

            # 询问是否继续
            echo ""
            read -p "是否继续下一个表？(y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "用户中止"
                break
            fi
        fi

        index=$((index + 1))
    done

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # 输出总结
    echo ""
    echo "=========================================="
    echo "ANALYZE 完成"
    echo "=========================================="
    echo "  成功: $success_count/${#tables[@]}"
    echo "  失败: $fail_count"
    echo "  总耗时: $((duration / 60)) 分钟 $((duration % 60)) 秒"
    echo "=========================================="

    if [ $fail_count -gt 0 ]; then
        exit 1
    fi
}

# 运行主函数
main "$@"
