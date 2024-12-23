#!/bin/bash
export ENV_PACK_FOLDER=$(conda info --base)/envs
CURRENT_DIR=$(pwd)
PROJECT_ROOT=$(dirname "$CURRENT_DIR")
export PROJECT_ROOT="$PROJECT_ROOT"
export TEMPLATE_RELEVANCE_ROOT=${PROJECT_ROOT}/retro_planner/packages/template_relevance
export PARROT_ROOT=${PROJECT_ROOT}/retro_planner/packages/parrot

# 设置默认的镜像仓库，如果没有指定环境变量则使用此默认值
ASKCOS_REGISTRY=${ASKCOS_REGISTRY:-registry.gitlab.com/mlpds_mit/askcosv2/askcos2_core}

SINGULARITY_NETWORK="retroplanner_network"

echo "Creating network: $SINGULARITY_NETWORK"


# 启动 parrot_serve_container 实例
echo "Starting instance: parrot_serve_container"
singularity instance start \
  --bind "${PARROT_ROOT}/mars:/app/parrot/mars" \
  --bind "${PARROT_ROOT}:/app/parrot" \
  ./singularity_images/parrot_image.sif \
  parrot_serve_container

singularity exec --nv --env PATH=/opt/conda/parrot_env/bin:$PATH instance://parrot_serve_container \
  torchserve --start --foreground --ncs --model-store=/app/parrot/mars \
  --models USPTO_condition=USPTO_condition.mar --ts-config /app/parrot/config.properties

# curl http://localhost:9510/predictions/USPTO_condition     --header "Content-Type: application/json"     --request POST     --data '["CCCC>>CCCC"]'