import os

from app import celery  # 导入你的 Celery 实例

if __name__ == "__main__":
    concurrency = int(os.environ.get("CELERY_WORKER_CONCURRENCY", "1"))
    hostname = os.environ.get("CELERY_WORKER_HOSTNAME", "").strip()

    # argv 参数用于提供 Celery 命令行选项，模仿在命令行中启动 worker 的方式
    argv = [
        'worker',  # 启动一个 worker
        '--loglevel=info',  # 设置日志级别为 info
        f'--concurrency={concurrency}',  # 可选参数，设置 worker 的并发数量
        '--pool=solo'  # 使用 solo 池，避免 CUDA 初始化问题
    ]
    if hostname:
        argv.append(f'--hostname={hostname}')
    celery.worker_main(argv)
