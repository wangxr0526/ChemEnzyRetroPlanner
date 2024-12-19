
# 检查文件是否存在，如果不存在则执行复制操作
if [ ! -f ./parrot_env.tar.gz ]; then
    echo "File does not exist in the current directory, copying it..."
    cp ~/data/ubuntu_work_beta/env_pack/parrot_env.tar.gz ./
else
    echo "File already exists in the current directory, skipping copy."
fi

# 构建 Docker 镜像
docker build -t parrot_image:latest -f Dockerfile_gpu .
rm parrot_env.tar.gz
