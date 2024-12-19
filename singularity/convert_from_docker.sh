#!/bin/bash

# 定义 Docker 镜像列表
declare -A IMAGES
IMAGES=(
  ["parrot_image:latest"]="parrot_image.sif"
  ["retroplanner_image:latest"]="retroplanner_image.sif"
  ["registry.gitlab.com/mlpds_mit/askcosv2/askcos2_core/retro/template_relevance:1.0-gpu"]="template_relevance.sif"
)

# 定义目标存储文件夹
OUTPUT_DIR="./singularity_images"
mkdir -p $OUTPUT_DIR

# 创建一个临时目录存放中间的 TAR 文件
TMP_DIR="./docker_to_singularity_temp"
mkdir -p $TMP_DIR

echo "开始转换 Docker 镜像为 Singularity 镜像..."

# 遍历镜像列表并逐一转换
for docker_image in "${!IMAGES[@]}"; do
  singularity_image="${IMAGES[$docker_image]}"
  tar_file="${TMP_DIR}/${docker_image//[:\/]/_}.tar"
  output_image="${OUTPUT_DIR}/${singularity_image}"

  echo "正在处理镜像：$docker_image"
  
  # 保存 Docker 镜像为 TAR 文件
  echo "导出 Docker 镜像到 TAR 文件：$tar_file"
  docker save -o "$tar_file" "$docker_image"
  
  # 将 TAR 文件转换为 Singularity 镜像
  echo "转换为 Singularity 镜像：$output_image"
  singularity build "$output_image" "docker-archive://$tar_file"
  
  echo "镜像 $docker_image 转换完成为 $output_image"
done

# 清理临时目录
echo "清理临时文件..."
rm -rf $TMP_DIR

echo "所有镜像转换完成！Singularity 镜像已存储在文件夹：$OUTPUT_DIR"
