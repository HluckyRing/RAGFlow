# -*- coding: utf-8 -*-
"""`scripts/update_deploy.py`（Docker 更新部署）的回归。

这脚本治的是**静默陷阱**：`docker compose up -d` 不会因为源码变了就重建镜像，于是「git pull 完
就以为更新好了」其实还在跑旧代码。所以这里钉住的不是「脚本能跑」，而是几个会出事的性质：

- `.env` 没有 `AUTH_TOKEN` 就必须**在前置检查阶段**拦下来（容器必然对外监听，没有它拒绝启动）；
- 工作区脏不许自动 stash / 覆盖（本地改动不能被静默丢掉）；
- **绝不出现 `down -v`**（那是连数据卷一起删）；
- 健康检查要能区分 `running` / `exited` / `unhealthy`，容器退出要立刻报错而不是等超时。
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "update_deploy.py"


def _module():
    spec = importlib.util.spec_from_file_location("update_deploy_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_exists_and_is_importable():
    assert SCRIPT.exists(), "Docker 更新脚本应当随仓库提供"
    module = _module()
    assert callable(module.main)


@pytest.mark.parametrize("text,expected", [
    ("AUTH_TOKEN=abc123\n", True),
    ('AUTH_TOKEN="abc123"\n', True),
    ("  AUTH_TOKEN = x  \n", True),
    ("# AUTH_TOKEN=abc\n", False),          # 注释掉的不算
    ("AUTH_TOKEN=\n", False),               # 空值不算
    ("AUTH_TOKEN=   \n", False),
    ("API_KEY=sk-xxx\n", False),
    ("", False),
])
def test_auth_token_detection(text, expected):
    """容器必然以 0.0.0.0 监听，没有令牌就起不来 —— 前置检查必须先拦住。"""
    assert _module().has_auth_token(text) is expected


@pytest.mark.parametrize("payload,expected", [
    ([{"State": {"Status": "running", "Health": {"Status": "healthy"}}}], ("running", "healthy")),
    ([{"State": {"Status": "running", "Health": {"Status": "starting"}}}], ("running", "starting")),
    ([{"State": {"Status": "exited"}}], ("exited", "none")),
    ([{"State": {}}], ("", "none")),
    ([], ("", "none")),
])
def test_health_state_parsing(payload, expected):
    assert _module().health_state(payload) == expected


def test_script_never_deletes_the_data_volume():
    """`docker compose down -v` 会把会话/上传/向量库一起删掉 —— 更新脚本绝不能出现它。"""
    text = SCRIPT.read_text(encoding="utf-8")
    code = "\n".join(line for line in text.splitlines()
                     if not line.strip().startswith("#"))       # 注释里解释「别用」不算
    assert "down" not in code or "-v" not in code, "更新脚本不许出现 `down -v`（会连数据卷一起删）"
    assert "up" in code and "--build" in code, "更新必须重建镜像（compose 不会自动重建）"


def test_script_refuses_when_worktree_is_dirty():
    """工作区脏要中止：脚本不替用户 stash/提交，本地改动不能静默丢。"""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "git\", \"status\", \"--porcelain\"" in text or "status\", \"--porcelain" in text, \
        "更新前必须检查工作区是否干净"
    assert "先提交或 git stash" in text, "要用人话说清为什么停下"
