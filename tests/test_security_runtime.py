# -*- coding: utf-8 -*-
"""启动安全闸 + /api/chat 限流的运行时回归（2026-11-12 第十七轮）。

这两条治的是「不安全配置可以悄悄发生」：

- 这个服务**没有鉴权**，唯一的护栏是「默认只绑回环」。但照文档改成 `HOST=0.0.0.0` 时
  护栏就没了，而且不报错 —— 所以启动自检必须 fail-closed：对外监听 + 没有 `AUTH_TOKEN`
  直接拒绝启动（退出码 2），本机回环 + 空 token 仍保持零摩擦。
- `/api/chat` 每问一次都要花钱调大模型，所以要有按客户端 IP 的滑动窗口限流；
  按 `session_id` 计数没有意义（那是客户端生成的，换一个就绕过）。

`tests/test_security.py` 那批是源码级静态断言（不许写死 0.0.0.0 之类），这里补的是**行为**。
"""
import pytest

import src.server as server
from src.security import (
    SlidingWindowLimiter,
    ensure_startup_safe,
    is_loopback,
    startup_error,
)


# ── 启动自检 ──

@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", " 127.0.0.1 ", "LOCALHOST"])
def test_loopback_hosts_are_recognised(host):
    assert is_loopback(host) is True


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "10.0.0.1", "", None])
def test_non_loopback_hosts_are_not(host):
    assert is_loopback(host) is False


def test_loopback_without_token_may_start():
    """本机单人用：保持改造前的零摩擦，不需要 token。"""
    assert startup_error("127.0.0.1", "") is None


def test_external_bind_without_token_is_refused():
    message = startup_error("0.0.0.0", "")
    assert message and "拒绝启动" in message
    assert "AUTH_TOKEN" in message
    assert "127.0.0.1" in message, "要告诉用户最省事的那个选项"


def test_external_bind_with_token_may_start():
    assert startup_error("0.0.0.0", "s3cret-token") is None


def test_whitespace_only_token_counts_as_missing():
    assert startup_error("0.0.0.0", "   ") is not None


def test_ensure_startup_safe_exits_with_code_two():
    with pytest.raises(SystemExit) as excinfo:
        ensure_startup_safe("0.0.0.0", "")
    assert excinfo.value.code != 0
    assert ensure_startup_safe("127.0.0.1", "") is None


# ── 滑动窗口限流 ──

def test_limiter_allows_up_to_limit_then_blocks():
    limiter = SlidingWindowLimiter(limit=3, window_seconds=60)
    assert [limiter.allow("ip", now=t) for t in (0, 1, 2)] == [True, True, True]
    assert limiter.allow("ip", now=3) is False


def test_limiter_window_slides():
    """窗口滑过之后必须重新放行 —— 否则「限流」会变成永久封禁。"""
    limiter = SlidingWindowLimiter(limit=2, window_seconds=10)
    assert limiter.allow("ip", now=0) is True
    assert limiter.allow("ip", now=1) is True
    assert limiter.allow("ip", now=2) is False
    assert limiter.allow("ip", now=11) is True, "t=0 那次已滑出窗口"
    assert limiter.allow("ip", now=11.5) is True, "t=1 那次也滑出了"
    assert limiter.allow("ip", now=12) is False, "此刻窗口里已有两次"


def test_limiter_is_per_key():
    limiter = SlidingWindowLimiter(limit=1, window_seconds=60)
    assert limiter.allow("a", now=0) is True
    assert limiter.allow("b", now=0) is True, "不同 IP 不该互相影响"
    assert limiter.allow("a", now=0) is False


def test_limiter_zero_means_unlimited():
    limiter = SlidingWindowLimiter(limit=0)
    assert all(limiter.allow("ip", now=t) for t in range(100))


def test_retry_after_tells_how_long_to_wait():
    limiter = SlidingWindowLimiter(limit=1, window_seconds=10)
    assert limiter.retry_after("ip", now=0) == 0
    limiter.allow("ip", now=0)
    assert limiter.retry_after("ip", now=4) == pytest.approx(6.0)
    assert limiter.retry_after("ip", now=20) == 0


# ── 接线：/api/chat 返回 429 ──

@pytest.fixture
def client(tmp_path, monkeypatch):
    import src.state as state
    from fastapi.testclient import TestClient

    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", tmp_path / "kb_state.json")
    state._cache.clear()
    state._locks.clear()
    server.sessions.clear()
    monkeypatch.setattr(server, "init_vector_store", lambda name=None: (None, False))
    yield TestClient(server.app)
    server.sessions.clear()
    state._cache.clear()
    state._locks.clear()


def test_chat_returns_429_after_the_limit(client, monkeypatch):
    monkeypatch.setattr(server, "_chat_limiter", SlidingWindowLimiter(limit=1, window_seconds=60))
    payload = {"question": "有什么内容", "session_id": "s1", "conv_id": "不存在的对话"}

    first = client.post("/api/chat", json=payload)
    assert first.status_code == 400, "第一次不被限流，走到「对话不存在」"

    second = client.post("/api/chat", json=payload)
    assert second.status_code == 429
    assert "Retry-After" in second.headers
    assert "请求过于频繁" in second.json()["error"]


def test_chat_limit_is_checked_before_touching_the_model(client, monkeypatch):
    """被限流时不许再跑检索/落盘 —— 否则「限流」只挡住了回答，没挡住成本。"""
    called = []
    monkeypatch.setattr(server, "_chat_limiter", SlidingWindowLimiter(limit=0, window_seconds=60))
    monkeypatch.setattr(server, "retrieve_and_build_context",
                        lambda *a, **k: called.append(1) or ("", []))
    monkeypatch.setattr(server, "_chat_limiter", SlidingWindowLimiter(limit=1, window_seconds=60))

    payload = {"question": "有什么内容", "session_id": "s1", "conv_id": "不存在的对话"}
    client.post("/api/chat", json=payload)
    client.post("/api/chat", json=payload)
    assert called == [], "对话不存在时也没走到检索；限流分支更不该走到"
