# -*- coding: utf-8 -*-
"""根目录快捷入口：python server.py 即可启动。

监听地址复用 src/config.py 的 HOST/PORT（默认只绑回环），
不要在这里写死对外地址 —— 这个服务没有鉴权。
"""
import uvicorn

from src.config import HOST, PORT

if __name__ == "__main__":
    from src.server import app

    uvicorn.run(app, host=HOST, port=PORT)
