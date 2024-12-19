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

# 启动 retro_template_relevance 实例
echo "Starting instance: retro_template_relevance"
singularity instance start \
  --bind "${TEMPLATE_RELEVANCE_ROOT}:/app/template_relevance" \
  --bind "${TEMPLATE_RELEVANCE_ROOT}/mars:/app/template_relevance/mars" \
  ./singularity_images/template_relevance_1.0-gpu.sif \
  retro_template_relevance

nohup singularity exec --nv --env PATH=/opt/conda/bin:$PATH instance://retro_template_relevance \
  torchserve --start --foreground --ncs --model-store=/app/template_relevance/mars \
  --models bkms_metabolic=bkms_metabolic.mar cas=cas.mar pistachio=pistachio.mar \
  pistachio_ringbreaker=pistachio_ringbreaker.mar reaxys=reaxys.mar \
  reaxys_biocatalysis=reaxys_biocatalysis.mar --ts-config /app/template_relevance/config.properties &
# curl http://localhost:9410/predictions/reaxys     --header "Content-Type: application/json"     --request POST     --data '{"smiles": ["[Br:1][CH2:2]/[CH:3]=[CH:4]/[C:5](=[O:6])[O:7][Si:8]([CH3:9])([CH3:10])[CH3:11]", "CC(C)(C)OC(=O)N1CCC(OCCO)CC1"]}'

# 启动 parrot_serve_container 实例
echo "Starting instance: parrot_serve_container"
singularity instance start \
  --bind "${PARROT_ROOT}/mars:/app/parrot/mars" \
  --bind "${PARROT_ROOT}:/app/parrot" \
  ./singularity_images/parrot_image.sif \
  parrot_serve_container

nohup singularity exec --nv --env PATH=/opt/conda/parrot_env/bin:$PATH instance://parrot_serve_container \
  torchserve --start --foreground --ncs --model-store=/app/parrot/mars \
  --models USPTO_condition=USPTO_condition.mar --ts-config /app/parrot/config.properties &
# curl http://localhost:9510/predictions/USPTO_condition     --header "Content-Type: application/json"     --request POST     --data '["CCCC>>CCCC"]'

# 启动 retroplanner_container 实例
echo "Starting instance: retroplanner_container"
singularity instance start \
  --nv \
  --bind "${POJECT_ROOT}:/retro_planner" \
  --bind "${POJECT_ROOT}/retro_planner/packages/organic_enzyme_rxn_classifier/organic_enzyme_rxn_classifier:/retro_planner/retro_planner/packages/organic_enzyme_rxn_classifier/organic_enzyme_rxn_classifier" \
  --bind "$HOME/.cache/torch/hub/checkpoints:/home/retro_planner/.cache/torch/hub/checkpoints" \
  ./singularity_images/retroplanner_image.sif \
  retroplanner_container

# singularity exec --nv instance://retroplanner_container /bin/bash &

# echo "All instances started successfully!"

# 查看实例状态
singularity instance list
