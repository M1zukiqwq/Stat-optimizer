#!/bin/bash
# ANALYZE 辅助脚本 - 统一管理统计信息生成
# 确保与漂移表列表一致

# 注意：不使用 set -e，因为我们需要手动处理错误

# 配置
PRESTO_CLI="${PRESTO_CLI:-/home/tianqc/presto-server-0.296/presto-cli}"
HOST="localhost:8080"
CATALOG="iceberg"
SCHEMA="imdb"
KLL_PARAMETER=1024

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 表列表定义
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

# 使用说明
usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --all       分析所有 13 张事实表（默认）"
    echo "  --core      只分析核心 6 张表"
    echo "  --tables    自定义表列表（空格分隔）"
    echo "  --help      显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --all                                    # 分析所有事实表"
    echo "  $0 --core                                   # 只分析核心表"
    echo "  $0 --tables \"title cast_info movie_info\"   # 自定义表列表"
    exit 1
}

# 执行 ANALYZE
analyze_table() {
    local table="$1"

    echo -e "${GREEN}[INFO]${NC} ANALYZE $table..."

    result=$($PRESTO_CLI --server $HOST --catalog $CATALOG --schema $SCHEMA \
        --execute "SET SESSION optimizer_use_histograms = true; SET SESSION iceberg.statistics_kll_sketch_k_parameter = $KLL_PARAMETER; ANALYZE $table;" 2>&1)

    local exit_code=$?

    # 检查是否有错误信息
    if [ $exit_code -eq 0 ] && ! echo "$result" | grep -qi "error\|failed\|exception"; then
        echo -e "${GREEN}  ✓ $table 完成${NC}"
        return 0
    else
        echo -e "${YELLOW}  ✗ $table 失败${NC}"
        if [ -n "$result" ]; then
            echo "     错误信息: $result" | head -3
        fi
        return 1
    fi
}

# 主函数
main() {
    local mode="all"
    local custom_tables=""

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                mode="all"
                shift
                ;;
            --core)
                mode="core"
                shift
                ;;
            --tables)
                mode="custom"
                custom_tables="$2"
                shift 2
                ;;
            --help)
                usage
                ;;
            *)
                echo "未知选项: $1"
                usage
                ;;
        esac
    done

    # 选择表列表
    local tables=()
    case $mode in
        all)
            tables=("${ALL_FACT_TABLES[@]}")
            echo "=========================================="
            echo "分析所有 13 张事实表"
            echo "=========================================="
            ;;
        core)
            tables=("${CORE_TABLES[@]}")
            echo "=========================================="
            echo "分析核心 6 张表"
            echo "=========================================="
            ;;
        custom)
            IFS=' ' read -ra tables <<< "$custom_tables"
            echo "=========================================="
            echo "分析自定义表列表"
            echo "=========================================="
            ;;
    esac

    echo "配置:"
    echo "  Presto: $HOST"
    echo "  Catalog: $CATALOG"
    echo "  Schema: $SCHEMA"
    echo "  KLL 参数: $KLL_PARAMETER"
    echo "  表数量: ${#tables[@]}"
    echo "=========================================="
    echo ""

    # 执行 ANALYZE
    local success_count=0
    local fail_count=0
    local start_time=$(date +%s)

    for table in "${tables[@]}"; do
        if analyze_table "$table"; then
            success_count=$((success_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi
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
    echo "  耗时: $((duration / 60)) 分钟 $((duration % 60)) 秒"
    echo "=========================================="

    if [ $fail_count -gt 0 ]; then
        exit 1
    fi
}

# 运行主函数
main "$@"
