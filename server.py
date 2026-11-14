# -*- coding: utf-8 -*-
"""根目录快捷入口：python server.py 即可启动。

监听地址复用 src/config.py 的 HOST/PORT（默认只绑回环），
不要在这里写死对外地址 —— 这个服务没有鉴权。

**对外监听必须设 AUTH_TOKEN**：检查在 `src/security.ensure_startup_safe`，
两个启动入口共用同一个出口，不通过就直接退出（退出码 2）。
"""
import uvicorn

from src.config import AUTH_TOKEN, HOST, PORT
from src.security import ensure_startup_safe

if __name__ == "__main__":
    ensure_startup_safe(HOST, AUTH_TOKEN)
    from src.server import app

    uvicorn.run(app, host=HOST, port=PORT)
