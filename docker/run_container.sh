#!/bin/bash

set -euo pipefail

# 默认执行全部逻辑
RUN_WEB_APP=true
RECREATE=false

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --only-run-backend)
      RUN_WEB_APP=false
      shift
      ;;
    --recreate)
      RECREATE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
export PROJECT_ROOT="$PROJECT_ROOT"
export TEMPLATE_RELEVANCE_ROOT=${PROJECT_ROOT}/retro_planner/packages/template_relevance
export PARROT_ROOT=${PROJECT_ROOT}/retro_planner/packages/parrot

# 设置默认的镜像仓库，如果没有指定环境变量则使用此默认值
export ASKCOS_REGISTRY=${ASKCOS_REGISTRY:-registry.gitlab.com/mlpds_mit/askcosv2/askcos2_core}

# 检查是否存在 retroplanner_image 镜像，如果不存在则构建
if ! docker image inspect wangxiaorui/retroplanner_image:latest > /dev/null 2>&1; then
    echo "Building retroplanner_image..."
    ENV_PACK_FOLDER="$(conda info --base)/envs"
    cp "${ENV_PACK_FOLDER}/retro_planner_env.tar.gz" ./retro_planner_env_py38.tar.gz
    docker build -t wangxiaorui/retroplanner_image:latest -f ./Dockerfile .
    rm retro_planner_env_py38.tar.gz
fi

# 启动 Docker Compose 服务
if docker compose version > /dev/null 2>&1; then
    COMPOSE=(docker compose)
else
    COMPOSE=(docker-compose)
fi
COMPOSE_ARGS=(up -d)
if [ "$RECREATE" = true ]; then
    COMPOSE_ARGS+=(--force-recreate)
fi
"${COMPOSE[@]}" "${COMPOSE_ARGS[@]}"

# Detect lost GPU access before starting Flask and accepting requests.
for container in retro_template_relevance parrot_serve_container retroplanner_container; do
    if ! docker exec "$container" nvidia-smi --query-gpu=name --format=csv,noheader; then
        echo "GPU access failed in ${container}. Check host nvidia-smi, then rerun with --recreate." >&2
        exit 1
    fi
done
if ! docker exec retroplanner_container /opt/conda/envs/retro_planner_env_py38/bin/python -c \
    'import torch; x = torch.ones(1, device="cuda:0"); torch.cuda.synchronize(); print("CUDA allocation OK:", torch.cuda.get_device_name(0))'; then
    echo "CUDA initialization failed in retroplanner_container; web services were not started." >&2
    exit 1
fi

# 执行最后一步逻辑
if [ "$RUN_WEB_APP" = true ]; then
    docker exec retroplanner_container /bin/bash -c "source ~/.bashrc && source activate /opt/conda/envs/retro_planner_env_py38/ && cd /retro_planner && bash setup.sh && cd /retro_planner/webapp && bash ./run_web_server.sh"
else
    echo "Skipping the last step as per user request."
fi
