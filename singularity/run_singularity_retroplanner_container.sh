#!/bin/bash
export ENV_PACK_FOLDER=$(conda info --base)/envs
CURRENT_DIR=$(pwd)
PROJECT_ROOT=$(dirname "$CURRENT_DIR")
export PROJECT_ROOT="$PROJECT_ROOT"
export TEMPLATE_RELEVANCE_ROOT=${PROJECT_ROOT}/retro_planner/packages/template_relevance
export PARROT_ROOT=${PROJECT_ROOT}/retro_planner/packages/parrot

# 设置默认的镜像仓库，如果没有指定环境变量则使用此默认值
ASKCOS_REGISTRY=${ASKCOS_REGISTRY:-registry.gitlab.com/mlpds_mit/askcosv2/askcos2_core}


# 启动 retroplanner_container 实例
echo "Starting instance: retroplanner_container"
singularity instance start \
  --nv \
  --bind "${PROJECT_ROOT}:/retro_planner" \
  --bind "$HOME/.cache/torch/hub/checkpoints:/home/retro_planner/.cache/torch/hub/checkpoints" \
  ./singularity_images/retroplanner_image.sif \
  retroplanner_container


singularity exec --nv instance://retroplanner_container /bin/bash -c "source ~/.bashrc && source activate /opt/conda/envs/retro_planner_env_py38/ && cd /retro_planner && bash setup.sh && cd /retro_planner/webapp && bash ./run_web_server.sh"