# -*- coding: utf-8 -*-
"""安全相关静态守卫（P1-21 / P2）。

守的是「把不安全默认值写回代码」这类回归：默认监听全网段、以及 archived 脚本里
再次把全局 HTTPS 上下文换成不校验证书的实现。都是源码级断言，不依赖运行时。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_startup_does_not_bind_all_interfaces_by_default():
    """P1-21：默认只监听回环；要暴露到局域网必须显式设置 HOST。"""
    for rel in ("src/server.py", "server.py"):
        text = _read(rel)
        assert "0.0.0.0" not in text, f"{rel} 里写死了 0.0.0.0"


def test_root_launcher_reuses_configured_host_and_port():
    text = _read("server.py")
    assert "HOST" in text and "PORT" in text, "根目录启动脚本必须复用 config 里的 HOST/PORT"


def test_src_launcher_reuses_configured_host_and_port():
    text = _read("src/server.py")
    assert "host=HOST" in text and "port=PORT" in text, "src/server.py 必须用配置里的 HOST/PORT 启动"


def test_legacy_scripts_do_not_disable_tls_verification():
    """P2：legacy/ 里曾用 ssl._create_unverified_context 替换全局 HTTPS 上下文。"""
    for path in sorted((ROOT / "legacy").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "_create_unverified_context" not in text, f"{path.name} 仍在关闭 TLS 校验"
        assert "ssl._create_default_https_context" not in text, f"{path.name} 仍在替换全局 HTTPS 上下文"
