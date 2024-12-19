from app import celery  # 导入你的 Celery 实例

if __name__ == "__main__":
    # argv 参数用于提供 Celery 命令行选项，模仿在命令行中启动 worker 的方式
    argv = [
        'worker',  # 启动一个 worker
        '--loglevel=info',  # 设置日志级别为 info
        '--concurrency=1',  # 可选参数，设置 worker 的并发数量
        '--pool=solo'  # 使用 solo 池，避免 CUDA 初始化问题
    ]
    celery.worker_main(argv)
