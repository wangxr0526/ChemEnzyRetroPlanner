# 使用较小的基础镜像，如需要完整的开发环境则保持 devel 镜像
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04

# 安装必要的软件包并清理无用缓存
RUN apt-get update && apt-get install -y --no-install-recommends wget libxrender1 libxext6 redis-server curl build-essential && \
    rm -rf /var/lib/apt/lists/*


# # 下载并安装 Miniconda，安装后删除安装脚本
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/Miniconda3.sh && \
#     bash /tmp/Miniconda3.sh -b -p /opt/conda && \
#     rm /tmp/Miniconda3.sh

# # 设置环境变量
# ENV PATH=/opt/conda/bin:$PATH

# 创建新用户 retro_planner 并添加到 sudo 组   (rootless情况下使用)
# RUN useradd -m -s /bin/bash retro_planner && \
#     echo "retro_planner ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 解压传入的 tar.gz 文件，并删除压缩包
COPY retro_planner_env_py38.tar.gz /opt/conda/envs/retro_planner_env_py38.tar.gz
RUN mkdir -p /opt/conda/envs/retro_planner_env_py38 && \
    tar -xvf /opt/conda/envs/retro_planner_env_py38.tar.gz -C /opt/conda/envs/retro_planner_env_py38 && \
    rm /opt/conda/envs/retro_planner_env_py38.tar.gz 

# 设置 Python 环境变量为默认的 Python
ENV PATH=/opt/conda/envs/retro_planner_env_py38/bin:$PATH

# 安装必要的 Python 包
RUN pip install --no-cache-dir flask==2.2.5 flask_cors celery numpy==1.23.5 redis -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 设置工作目录
WORKDIR /retro_planner

# 切换默认用户 (rootless情况下使用)
# USER retro_planner
# RUN mkdir /home/retro_planner/.cache
RUN mkdir /root/.cache


# 设置容器启动时默认命令
CMD ["bash"]
