# -*- coding: utf-8 -*-
"""pytest 公共配置：把项目根目录加进 sys.path，并提供不依赖真实 API Key 的环境变量。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 必须在 import src.config 之前设置，否则 OpenAI 客户端初始化会失败
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("BASE_URL", "https://example.invalid")

import pytest


@pytest.fixture(autouse=True)
def _isolate_upload_dir(tmp_path, monkeypatch):
    """所有测试都不许把上传原件写进真实的 uploads/。

    原文件预览要落盘原件，漏掉这层隔离就会在开发者工作区留下用户数据。
    """
    import src.server as server

    monkeypatch.setattr(server, "UPLOAD_DIR", str(tmp_path / "uploads"), raising=False)
