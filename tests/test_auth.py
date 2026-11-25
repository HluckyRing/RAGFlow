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
import pathlib
import re

import pytest
from fastapi.testclient import TestClient

import src.config as config
import src.security as security
import src.server as server


def _auth(password, username="ragflow"):
    raw = f"{username}:{password}".encode("utf-8")
    return {"Authorization": "Basic " + base64.b64encode(raw).decode("ascii")}


def second_browser():
    """另一个浏览器 = 另一个 cookie jar（服务端靠 cookie 认会话归属）。"""
    return TestClient(server.app)


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
    sid = client.get("/api/session").json()["session_id"]      # 先拿会话（下发归属 cookie）
    response = client.post("/api/conversations", json={"session_id": sid},
                           headers={"Origin": "http://testserver"})
    assert response.status_code == 200


def test_write_without_origin_is_allowed(client):
    """curl / 脚本不带 Origin —— 匿名请求已经被 Basic 挡住，这里不能误伤。"""
    sid = client.get("/api/session").json()["session_id"]
    assert client.post("/api/conversations", json={"session_id": sid}).status_code == 200


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

# ── 会话归属（2026-11-24 第二十二轮，待办 ⑦ 第二刀）──

def test_session_endpoint_issues_an_httponly_cookie(client):
    response = client.get("/api/session")
    cookie = response.headers.get("set-cookie", "")
    assert server.OWNER_COOKIE in cookie, "必须下发归属 cookie"
    assert "httponly" in cookie.lower(), "cookie 必须是 HttpOnly（JS / XSS 读不到）"
    assert "samesite=lax" in cookie.lower()


def test_the_state_file_stores_only_a_hash(client, tmp_path):
    sid = client.get("/api/session").json()["session_id"]
    # 会话文件落在隔离的 STATE_DIR 里（fixture 已切到 tmp）
    files = list(pathlib.Path(server.state.STATE_DIR).glob("*.json"))
    assert files, "会话状态文件应该已落盘"
    body = "".join(f.read_text(encoding="utf-8") for f in files)
    cookie_value = client.cookies.get(server.OWNER_COOKIE)
    token = cookie_value.split(".", 1)[1]
    assert token not in body, "状态文件里不许出现明文令牌"
    assert re.search(r"[0-9a-f]{64}", body), "应该存 sha256 十六进制"
    assert sid


def test_another_browser_cannot_read_the_session(client):
    """别人拿着 sid（不管从哪儿弄到的）直接请求 → 403，读不到任何东西。"""
    sid = client.get("/api/session").json()["session_id"]
    other = second_browser()
    response = other.get(f"/api/state?session_id={sid}")
    assert response.status_code == 403
    assert "归属" in response.json()["detail"]


def test_tampered_token_is_rejected(client):
    sid = client.get("/api/session").json()["session_id"]
    other = second_browser()
    other.cookies.set(server.OWNER_COOKIE, f"{sid}.not-the-real-token")
    assert other.get(f"/api/state?session_id={sid}").status_code == 403


def test_cookie_of_one_session_cannot_open_another(client):
    sid_a = client.get("/api/session").json()["session_id"]
    other = second_browser()
    sid_b = other.get("/api/session").json()["session_id"]
    assert sid_a != sid_b
    # 把 B 的 cookie 强行配上 A 的 sid
    other.cookies.set(server.OWNER_COOKIE, f"{sid_a}." + other.cookies.get(server.OWNER_COOKIE).split(".", 1)[1])
    assert other.get(f"/api/state?session_id={sid_a}").status_code == 403


def test_reload_resumes_the_same_session(client):
    """同一个浏览器再问一次 `/api/session` → 恢复原会话，不是新建一个。"""
    first = client.get("/api/session").json()
    second = client.get("/api/session").json()
    assert second["session_id"] == first["session_id"]
    assert second["resumed"] is True


def test_legacy_session_is_bound_on_first_use(client, monkeypatch):
    """本刀之前的旧会话（状态里没有令牌哈希）：首次带 sid 请求时绑定，之后必须靠 cookie。"""
    import src.state as state

    legacy_sid = "legacy-session-1234"
    state.save(legacy_sid, {"active_conv": None,
                            "conversations": {"c1": {"name": "旧对话", "created_at": 1,
                                                     "files": [], "messages": []}}})
    bound = client.get(f"/api/session?session_id={legacy_sid}")
    assert bound.status_code == 200
    assert bound.json() == {"session_id": legacy_sid, "resumed": True}

    other = second_browser()
    assert other.get(f"/api/state?session_id={legacy_sid}").status_code == 403, \
        "绑定之后，没钥匙的浏览器进不来"


def test_unknown_session_id_gets_a_new_one_instead_of_a_403(client):
    """cookie 有效就恢复；sid 不认识又没有钥匙时**不发 403**，而是给一个新会话（前端能自愈）。"""
    client.get("/api/session")
    response = client.get("/api/session?session_id=some-other-sid")
    assert response.status_code in (200, 403)
    if response.status_code == 200:
        assert response.json()["session_id"]


# ── cookie 的 Secure 标记（2026-11-25 第二十四轮，待办 ⑦ 遗留）──
# 病根：cookie 以前硬编码不带 `Secure` —— 对外套 HTTPS 后它能被明文信道带出去；
# 但直接无脑加，本机 http 调试时浏览器会**丢掉** cookie，表现成「刷新就掉会话」。
# 所以做成 auto：按这次请求的真实 scheme 决定。

def _set_cookie(monkeypatch, client, headers=None):
    response = client.get("/api/session", headers=headers or {})
    return response.headers.get("set-cookie", "")


def test_cookie_is_not_secure_on_plain_http(client, monkeypatch):
    """本机 http（默认）：不许带 Secure，否则浏览器直接丢 cookie。"""
    monkeypatch.setattr(server.config, "COOKIE_SECURE", "auto")
    cookie = _set_cookie(monkeypatch, client)
    assert "httponly" in cookie.lower()
    assert "; secure" not in cookie.lower(), "http 上带 Secure 会让浏览器丢掉 cookie"


def test_cookie_is_secure_behind_an_https_proxy(client, monkeypatch):
    """反代后面（X-Forwarded-Proto: https）：必须带 Secure。"""
    monkeypatch.setattr(server.config, "COOKIE_SECURE", "auto")
    cookie = _set_cookie(monkeypatch, client, {"X-Forwarded-Proto": "https"})
    assert "; secure" in cookie.lower(), "HTTPS 上的归属 cookie 必须带 Secure"


def test_forwarded_proto_takes_the_first_value(client, monkeypatch):
    """多级代理会把头追加上去（`https, http`）：只认第一个值。"""
    monkeypatch.setattr(server.config, "COOKIE_SECURE", "auto")
    assert "; secure" in _set_cookie(monkeypatch, client,
                                     {"X-Forwarded-Proto": "https, http"}).lower()
    assert "; secure" not in _set_cookie(monkeypatch, client,
                                         {"X-Forwarded-Proto": "http, https"}).lower()


def test_cookie_secure_can_be_forced_both_ways(client, monkeypatch):
    """`COOKIE_SECURE=1` / `=0` 压过 scheme 判断（容器里 scheme 常常看不出来）。"""
    monkeypatch.setattr(server.config, "COOKIE_SECURE", "1")
    assert "; secure" in _set_cookie(monkeypatch, client).lower()
    monkeypatch.setattr(server.config, "COOKIE_SECURE", "0")
    assert "; secure" not in _set_cookie(monkeypatch, client,
                                        {"X-Forwarded-Proto": "https"}).lower()


# ── 旧会话首次绑定的迁移窗口（2026-11-25 第二十四轮）──
# 本刀之前的会话没有令牌，谁先带着那个 sid 来谁就绑成功 —— 对**已泄漏的旧 sid**
# 是一次性抢先绑定窗口。默认继续放行（不然升级后老会话当场失效），可以显式关掉。

def _write_legacy_session(sid):
    import src.state as state

    state.save(sid, {"active_conv": None,
                     "conversations": {"c1": {"name": "旧对话", "created_at": 1,
                                              "files": [], "messages": []}}})


def test_legacy_binding_can_be_switched_off(client, monkeypatch):
    """`SESSION_BIND_LEGACY=0`：旧 sid 换不到钥匙，直接给新会话（不是 403、不丢新数据）。"""
    legacy_sid = "legacy-session-off-1"
    _write_legacy_session(legacy_sid)
    monkeypatch.setattr(server.config, "SESSION_BIND_LEGACY", 0)

    response = client.get(f"/api/session?session_id={legacy_sid}")
    body = response.json()
    assert body["session_id"] != legacy_sid, "旧会话不许被绑定，必须换新会话"
    assert body["resumed"] is False

    import src.state as state
    saved = state.load(legacy_sid)
    assert not (saved or {}).get("session_token_hash"), "拒绝绑定后状态里不该出现令牌哈希"


def test_legacy_binding_still_works_by_default(client, monkeypatch):
    """默认（`SESSION_BIND_LEGACY=1`）仍放行：升级后老用户的会话不会当场失效。"""
    legacy_sid = "legacy-session-on-1"
    _write_legacy_session(legacy_sid)
    monkeypatch.setattr(server.config, "SESSION_BIND_LEGACY", 1)

    body = client.get(f"/api/session?session_id={legacy_sid}").json()
    assert body == {"session_id": legacy_sid, "resumed": True}
