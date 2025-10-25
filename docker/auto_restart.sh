#!/bin/bash

# ChemEnzyRetroPlanner 自动重启脚本
# 功能：每周定时停止并重启Docker容器
# 作者：自动生成
# 日期：2025-10-25

# 设置日志文件
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/auto_restart_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 日志函数
log() {
    local level=$1
    shift
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "${LOG_FILE}"
}

# 错误处理函数
error_exit() {
    log "ERROR" "$1"
    exit 1
}

# 检查是否在docker目录下
check_directory() {
    if [ ! -f "${SCRIPT_DIR}/stop_container.sh" ] || [ ! -f "${SCRIPT_DIR}/run_container.sh" ]; then
        error_exit "必须在docker目录下运行此脚本！当前目录：${SCRIPT_DIR}"
    fi
    log "INFO" "目录检查通过：${SCRIPT_DIR}"
}

# 检查脚本是否具有执行权限
check_permissions() {
    if [ ! -x "${SCRIPT_DIR}/stop_container.sh" ]; then
        log "WARN" "stop_container.sh 没有执行权限，正在添加..."
        chmod +x "${SCRIPT_DIR}/stop_container.sh" || error_exit "无法添加执行权限"
    fi
    
    if [ ! -x "${SCRIPT_DIR}/run_container.sh" ]; then
        log "WARN" "run_container.sh 没有执行权限，正在添加..."
        chmod +x "${SCRIPT_DIR}/run_container.sh" || error_exit "无法添加执行权限"
    fi
    
    log "INFO" "权限检查通过"
}

# 检查Docker服务状态
check_docker() {
    if ! command -v docker &> /dev/null; then
        error_exit "Docker 未安装或不在PATH中"
    fi
    
    if ! docker info &> /dev/null; then
        error_exit "Docker 服务未运行或当前用户无权限访问"
    fi
    
    log "INFO" "Docker 服务运行正常"
}

# 停止容器
stop_containers() {
    log "INFO" "开始停止容器..."
    
    cd "${SCRIPT_DIR}" || error_exit "无法切换到docker目录"
    
    # 执行停止脚本
    if bash ./stop_container.sh >> "${LOG_FILE}" 2>&1; then
        log "INFO" "容器停止成功"
        return 0
    else
        log "ERROR" "容器停止失败，退出码：$?"
        return 1
    fi
}

# 启动容器
start_containers() {
    log "INFO" "开始启动容器..."
    
    cd "${SCRIPT_DIR}" || error_exit "无法切换到docker目录"
    
    # 执行启动脚本
    if bash ./run_container.sh >> "${LOG_FILE}" 2>&1; then
        log "INFO" "容器启动成功"
        return 0
    else
        log "ERROR" "容器启动失败，退出码：$?"
        return 1
    fi
}

# 验证容器状态
verify_containers() {
    log "INFO" "验证容器状态..."
    
    local expected_containers=("retro_template_relevance" "parrot_serve_container" "retroplanner_container")
    local all_running=true
    
    for container in "${expected_containers[@]}"; do
        if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            log "INFO" "容器 ${container} 正在运行"
        else
            log "WARN" "容器 ${container} 未在运行"
            all_running=false
        fi
    done
    
    if [ "$all_running" = true ]; then
        log "INFO" "所有容器验证通过"
        return 0
    else
        log "WARN" "部分容器未运行，但继续执行"
        return 0
    fi
}

# 主函数
main() {
    log "INFO" "=========================================="
    log "INFO" "开始执行自动重启流程"
    log "INFO" "=========================================="
    
    # 前置检查
    check_directory
    check_permissions
    check_docker
    
    # 停止容器
    if ! stop_containers; then
        log "WARN" "停止容器时出现问题，但继续执行"
    fi
    
    # 等待10秒
    log "INFO" "等待 10 秒..."
    sleep 10
    
    # 启动容器
    if ! start_containers; then
        error_exit "启动容器失败，请检查日志：${LOG_FILE}"
    fi
    
    # 额外等待30秒让容器完全启动
    log "INFO" "等待 30 秒让容器完全启动..."
    sleep 30
    
    # 验证容器状态
    verify_containers
    
    log "INFO" "=========================================="
    log "INFO" "自动重启流程完成"
    log "INFO" "日志文件：${LOG_FILE}"
    log "INFO" "=========================================="
    
    # 清理旧日志（保留最近30天的日志）
    find "${LOG_DIR}" -name "auto_restart_*.log" -type f -mtime +30 -delete 2>/dev/null
    
    exit 0
}

# 捕获错误信号
trap 'log "ERROR" "脚本被中断"; exit 1' INT TERM

# 执行主函数
main
