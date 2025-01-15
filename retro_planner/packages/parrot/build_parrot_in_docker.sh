#!/bin/bash


export CONDA_HOME=$(conda info --base)
# 检查并下载 parrot_env.tar.gz
if [ ! -f $CONDA_HOME/envs/parrot_env.tar.gz ]; then
    echo "Downloading parrot_env.tar.gz..."
    curl -L -o $CONDA_HOME/envs/parrot_env.tar.gz "https://huggingface.co/xiaoruiwang/ChemEnzyRetroPlanner_metadata/resolve/main/parrot_env.tar.gz?download=true"
else
    echo "parrot_env.tar.gz already exists, skipping download."
fi

# 检查文件是否存在，如果不存在则执行复制操作
if [ ! -f ./parrot_env.tar.gz ]; then
    echo "File does not exist in the current directory, copying it..."
    cp $CONDA_HOME/envs/parrot_env.tar.gz ./
else
    echo "File already exists in the current directory, skipping copy."
fi

# 构建 Docker 镜像
docker build -t wangxiaorui/parrot_image:latest -f Dockerfile_gpu .
rm parrot_env.tar.gz
