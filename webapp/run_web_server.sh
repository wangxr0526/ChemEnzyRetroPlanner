#!/bin/bash

mkdir -p /retro_planner/webapp/logs
# 启动 Redis 服务器
nohup redis-server > /retro_planner/webapp/logs/redis.log 2>&1 &

# 启动主应用
nohup /opt/conda/envs/retro_planner_env_py38/bin/python app.py run --no-debugger --no-reload > /retro_planner/webapp/logs/app.log 2>&1 &

# 启动 Celery Worker
nohup /opt/conda/envs/retro_planner_env_py38/bin/python celery_worker_start.py -A app.celery worker --loglevel=info > /retro_planner/webapp/logs/celery.log 2>&1 &

# 确保后台任务启动
echo "Services started successfully."
