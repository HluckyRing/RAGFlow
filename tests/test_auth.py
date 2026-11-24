# -*- coding: utf-8 -*-
"""AUTH_TOKEN 逐请求校验 + CSRF（待办 ⑦ 第一刀，2026-11-24）。

之前的洞：`AUTH_TOKEN` 只在启动时被检查，请求到达时一次都没校验过 —— 设了 token 的对外部署
实际上对任何人开放。这组测试钉住四件事：
1. 设了 token：无凭证 401（带 `WWW-Authenticate`）、错误凭证 401、正确凭证 200；
2. 没设 token：行为与改造前一致（不要求凭证）；
3. 写操作 `Origin` 与 Host 不同源 → 403，同源或缺席 → 放行；
4. 错误响应里**不许泄漏 token**，比较用 `hmac.compare_digest`。
"""
import base64

import pytest
from fastapi.testclient import TestClient

import src.config as config
import src.security as security
import src.server as server


def _auth(password, username="ragflow"):
    raw = f"{username}:{password}".encode("utf-8")
    return {"Authorization": "Basic " + base64.b64encode(raw).decode("ascii")}


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """这组用例只测鉴权/CSRF，**绝不许碰真实 state/ 与 chroma_db/**。

    2026-11-24 实测踩到：`/api/conversations` 会按 sid 建状态文件、还会建向量集合 ——
    不隔离的话跑一次测试就在开发者工作区留下 17 个空会话 + 5 个空集合。
    """
    import src.config as config
    import src.retrieval as retrieval
    import src.state as state

    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "VECTOR_DB_PATH", str(tmp_path / "chroma"), raising=False)
    monkeypatch.setattr(retrieval, "VECTOR_DB_PATH", str(tmp_path / "chroma"), raising=False)
    monkeypatch.setattr(retrieval, "_chroma_client", None, raising=False)
    server.sessions.clear()
    yield
    server.sessions.clear()
    monkeypatch.setattr(retrieval, "_chroma_client", None, raising=False)


@pytest.fixture()
def client():
    return TestClient(server.app)


def test_without_a_token_nothing_is_required(client, monkeypatch):
    """默认回环单机：没设 AUTH_TOKEN → 不要求凭证，与改造前逐位一致。"""
    monkeypatch.setattr(config, "AUTH_TOKEN", "")
    assert client.get("/api/session").status_code == 200


def test_with_a_token_requests_need_credentials(client, monkeypatch):
    monkeypatch.setattr(config, "AUTH_TOKEN", "s3cret-token")
    response = client.get("/api/session")
    assert response.status_code == 401
    assert response.headers.get("www-authenticate", "").startswith("Basic"), \
        "401 必须带 WWW-Authenticate，浏览器才会弹框"


def test_wrong_or_malformed_credentials_are_rejected(client, monkeypatch):
    monkeypatch.setattr(config, "AUTH_TOKEN", "s3cret-token")
    for headers in (_auth("wrong"),
                    {"Authorization": "Basic !!!not-base64!!!"},
                    {"Authorization": "Bearer s3cret-token"},
                    {"Authorization": "Basic " + base64.b64encode(b"no-colon").decode("ascii")}):
        assert client.get("/api/session", headers=headers).status_code == 401


def test_the_username_is_irrelevant(client, monkeypatch):
    """用户名随便填（单用户工具）：只有密码 = AUTH_TOKEN 才算数。"""
    monkeypatch.setattr(config, "AUTH_TOKEN", "s3cret-token")
    for username in ("ragflow", "anyone", ""):
        assert client.get("/api/session",
                          headers=_auth("s3cret-token", username)).status_code == 200


def test_correct_credentials_are_accepted(client, monkeypatch):
    monkeypatch.setattr(config, "AUTH_TOKEN", "s3cret-token")
    response = client.get("/api/session", headers=_auth("s3cret-token"))
    assert response.status_code == 200
    assert response.json()["session_id"]


def test_the_error_body_never_leaks_the_token(client, monkeypatch):
    monkeypatch.setattr(config, "AUTH_TOKEN", "s3cret-token")
    body = client.get("/api/session").text
    assert "s3cret-token" not in body


def test_static_page_is_guarded_too(client, monkeypatch):
    """首页也是内容（对话历史由前端从 API 拉），一律要求凭证。"""
    monkeypatch.setattr(config, "AUTH_TOKEN", "s3cret-token")
    assert client.get("/").status_code == 401
    assert client.get("/", headers=_auth("s3cret-token")).status_code == 200


# ── CSRF：写操作的 Origin 必须与 Host 同源 ──────────────────────

def test_cross_site_write_is_refused(client):
    response = client.post("/api/conversations",
                           json={"session_id": "abc"},
                           headers={"Origin": "http://evil.example"})
    assert response.status_code == 403, "Basic 凭证是浏览器自动带的，必须靠 Origin 拒掉跨站写"


def test_same_origin_write_is_allowed(client):
    response = client.post("/api/conversations", json={"session_id": "abc"},
                           headers={"Origin": "http://testserver"})
    assert response.status_code == 200


def test_write_without_origin_is_allowed(client):
    """curl / 脚本不带 Origin —— 匿名请求已经被 Basic 挡住，这里不能误伤。"""
    assert client.post("/api/conversations", json={"session_id": "abc"}).status_code == 200


def test_reads_are_not_origin_checked(client):
    """GET 不做 Origin 检查：跨站读取靠 Basic 凭证与 CORS 默认禁止来挡。"""
    assert client.get("/api/session", headers={"Origin": "http://evil.example"}).status_code == 200


def test_origin_check_ignores_the_path():
    assert security.origin_allowed("http://127.0.0.1:8080/whatever", "127.0.0.1:8080", "POST")
    assert security.origin_allowed("http://127.0.0.1:8081", "127.0.0.1:8080", "POST") is False
    assert security.origin_allowed("http://127.0.0.1:8080", "", "POST") is False


# ── 凭证比较必须时序安全 ──────────────────────────────────────

def test_password_comparison_is_constant_time(monkeypatch):
    """行为检查（不是读源码）：比较必须真的走 `hmac.compare_digest`。

    读源码那种写法会被函数 docstring 里的同名文字骗过 —— 第一版就是这么假绿的。
    """
    import src.security as security_module

    calls = []
    real = security_module.hmac.compare_digest

    def spy(left, right):
        calls.append((left, right))
        return real(left, right)

    monkeypatch.setattr(security_module.hmac, "compare_digest", spy)
    assert security_module.basic_auth_ok(_auth("s3cret-token")["Authorization"],
                                         "s3cret-token")
    assert calls, "密码比较必须走 hmac.compare_digest（时序安全），不许用 =="
