#!/bin/bash

# 默认执行全部逻辑
RUN_WEB_APP=true

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --only-run-backend)
      RUN_WEB_APP=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

export ENV_PACK_FOLDER=$(conda info --base)/envs
CURRENT_DIR=$(pwd)
PROJECT_ROOT=$(dirname "$CURRENT_DIR")
export PROJECT_ROOT="$PROJECT_ROOT"
export TEMPLATE_RELEVANCE_ROOT=${PROJECT_ROOT}/retro_planner/packages/template_relevance
export PARROT_ROOT=${PROJECT_ROOT}/retro_planner/packages/parrot

# 设置默认的镜像仓库，如果没有指定环境变量则使用此默认值
ASKCOS_REGISTRY=${ASKCOS_REGISTRY:-registry.gitlab.com/mlpds_mit/askcosv2/askcos2_core}

# 检查是否存在 retroplanner_image 镜像，如果不存在则构建
if ! docker images | grep -q "retroplanner_image"; then
    echo "Building retroplanner_image..."
    cp ${ENV_PACK_FOLDER}/retro_planner_env.tar.gz ./retro_planner_env_py38.tar.gz
    docker build -t retroplanner_image:latest -f ./Dockerfile .
    rm retro_planner_env_py38.tar.gz
fi

# 启动 Docker Compose 服务
docker-compose up -d

# 执行最后一步逻辑
if [ "$RUN_WEB_APP" = true ]; then
    docker exec retroplanner_container /bin/bash -c "source ~/.bashrc && source activate /opt/conda/envs/retro_planner_env_py38/ && cd /retro_planner && bash setup.sh && cd /retro_planner/webapp && bash ./run_web_server.sh"
else
    echo "Skipping the last step as per user request."
fi
