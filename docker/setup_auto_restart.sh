#!/bin/bash

# ChemEnzyRetroPlanner 自动重启定时任务安装脚本
# 功能：配置 crontab 定时任务，每周日凌晨3点重启容器
# 作者：自动生成
# 日期：2025-10-25

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_RESTART_SCRIPT="${SCRIPT_DIR}/auto_restart.sh"
CRON_LOG_DIR="${SCRIPT_DIR}/logs"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否有执行权限
check_permissions() {
    if [ ! -x "${AUTO_RESTART_SCRIPT}" ]; then
        print_info "为 auto_restart.sh 添加执行权限..."
        chmod +x "${AUTO_RESTART_SCRIPT}"
    fi
}

# 创建日志目录
create_log_dir() {
    if [ ! -d "${CRON_LOG_DIR}" ]; then
        print_info "创建日志目录：${CRON_LOG_DIR}"
        mkdir -p "${CRON_LOG_DIR}"
    fi
}

# 检查现有的 crontab 任务
check_existing_cron() {
    if crontab -l 2>/dev/null | grep -q "auto_restart.sh"; then
        print_warn "检测到已存在的自动重启定时任务"
        return 0
    fi
    return 1
}

# 安装 crontab 任务
install_cron() {
    print_info "开始安装定时任务..."
    
    # 获取当前 crontab
    local temp_cron=$(mktemp)
    crontab -l > "${temp_cron}" 2>/dev/null || true
    
    # 添加新任务（每周日凌晨3点执行）
    # 格式：分 时 日 月 周
    # 0 3 * * 0 表示每周日凌晨3点
    echo "" >> "${temp_cron}"
    echo "# ChemEnzyRetroPlanner 自动重启任务 - 每周日凌晨3点执行" >> "${temp_cron}"
    echo "0 3 * * 0 cd ${SCRIPT_DIR} && bash ${AUTO_RESTART_SCRIPT} >> ${CRON_LOG_DIR}/cron_auto_restart.log 2>&1" >> "${temp_cron}"
    
    # 安装新的 crontab
    if crontab "${temp_cron}"; then
        print_info "定时任务安装成功！"
        rm -f "${temp_cron}"
        return 0
    else
        print_error "定时任务安装失败"
        rm -f "${temp_cron}"
        return 1
    fi
}

# 显示当前 crontab 任务
show_crontab() {
    print_info "当前的 crontab 任务："
    echo "----------------------------------------"
    crontab -l 2>/dev/null | grep -A 1 "ChemEnzyRetroPlanner" || echo "未找到相关任务"
    echo "----------------------------------------"
}

# 移除定时任务
uninstall_cron() {
    print_warn "移除自动重启定时任务..."
    
    local temp_cron=$(mktemp)
    crontab -l > "${temp_cron}" 2>/dev/null || true
    
    # 删除相关行
    sed -i '/ChemEnzyRetroPlanner 自动重启任务/d' "${temp_cron}"
    sed -i '/auto_restart.sh/d' "${temp_cron}"
    
    # 重新安装 crontab
    if crontab "${temp_cron}"; then
        print_info "定时任务已移除"
        rm -f "${temp_cron}"
        return 0
    else
        print_error "移除定时任务失败"
        rm -f "${temp_cron}"
        return 1
    fi
}

# 测试运行
test_run() {
    print_info "测试运行自动重启脚本..."
    print_warn "这将停止并重启所有容器，是否继续？(y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        bash "${AUTO_RESTART_SCRIPT}"
    else
        print_info "测试已取消"
    fi
}

# 显示帮助信息
show_help() {
    cat << EOF
ChemEnzyRetroPlanner 自动重启定时任务管理脚本

用法：
    bash $0 [选项]

选项：
    install     安装定时任务（每周日凌晨3点自动重启）
    uninstall   移除定时任务
    status      显示当前定时任务状态
    test        测试运行一次自动重启脚本
    help        显示此帮助信息

示例：
    bash $0 install    # 安装定时任务
    bash $0 status     # 查看状态
    bash $0 test       # 测试运行

定时任务说明：
    - 执行时间：每周日凌晨 3:00
    - 操作流程：停止容器 → 等待10秒 → 启动容器
    - 日志位置：${CRON_LOG_DIR}/
    - 日志保留：30天

EOF
}

# 主函数
main() {
    case "${1:-}" in
        install)
            check_permissions
            create_log_dir
            if check_existing_cron; then
                print_warn "是否要重新安装？这将覆盖现有任务。(y/N)"
                read -r response
                if [[ "$response" =~ ^[Yy]$ ]]; then
                    uninstall_cron
                    install_cron
                else
                    print_info "保持现有任务不变"
                fi
            else
                install_cron
            fi
            show_crontab
            ;;
        uninstall)
            uninstall_cron
            show_crontab
            ;;
        status)
            show_crontab
            print_info "自动重启脚本路径：${AUTO_RESTART_SCRIPT}"
            print_info "日志目录：${CRON_LOG_DIR}"
            if [ -f "${CRON_LOG_DIR}/cron_auto_restart.log" ]; then
                print_info "最近的执行记录："
                tail -n 20 "${CRON_LOG_DIR}/cron_auto_restart.log"
            fi
            ;;
        test)
            check_permissions
            create_log_dir
            test_run
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知选项：${1:-}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"
