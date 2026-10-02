# -*- coding: utf-8 -*-
"""SSE 聊天流程回归测试（把 LLM 和检索都替换掉，不产生真实 API 调用）。

覆盖两点：
- 流式回答结束后必须落盘（改造前会写进已被换掉的对象里，静默丢失）
- 当前提问不能在传给 LLM 的 history 里出现两次
"""
import io
import json

import pytest
from fastapi.testclient import TestClient

import src.server as server
import src.state as state


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", tmp_path / "kb_state.json")
    state._cache.clear()
    state._locks.clear()
    server.sessions.clear()
    monkeypatch.setattr(server, "init_vector_store", lambda name=None: (None, False))
    monkeypatch.setattr(server, "retrieve_and_build_context",
                        lambda *a, **k: ("背景：ROE 是净利润/股东权益", ["背景"]))
    yield TestClient(server.app)
    server.sessions.clear()
    state._cache.clear()
    state._locks.clear()


@pytest.fixture
def asked():
    """记录 stream_answer 实际收到的 question / history。"""
    seen = []

    def fake_stream(question, context, history):
        seen.append({"question": question, "history": list(history), "context": context})
        yield "净资产"
        yield "收益率"

    return seen, fake_stream


def _prepare(client, sid):
    conv_id = client.post("/api/conversations", json={"session_id": sid, "name": "T"}).json()["id"]
    conv = server.sessions[sid]["conversations"][conv_id]
    conv["files"] = [{"file_name": "a.txt", "file_text": "x"}]   # 跳过上传，避免碰 Chroma
    conv["full_text"] = "x"
    return conv_id


def test_answer_is_persisted_and_question_not_duplicated(client, asked, monkeypatch):
    seen, fake_stream = asked
    monkeypatch.setattr(server, "stream_answer", fake_stream)

    sid = client.get("/api/session").json()["session_id"]
    conv_id = _prepare(client, sid)

    r = client.post("/api/chat", json={"question": "ROE 是什么", "session_id": sid, "conv_id": conv_id})
    assert r.status_code == 200

    # SSE 事件要拼回完整回答（json.dumps 默认把中文转义成 \uXXXX，不能直接查字面量）
    events = [json.loads(line[6:]) for line in r.text.split("\n") if line.startswith("data: ")]
    assert events[-1]["type"] == "done"
    assert "".join(e["content"] for e in events if e["type"] == "token") == "净资产收益率"

    # 落盘：user + assistant 都在
    saved = json.loads(state.file_for(sid).read_text(encoding="utf-8"))
    msgs = saved["conversations"][conv_id]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "净资产收益率"

    # 第一轮：history 里不该有当前提问
    assert [m["content"] for m in seen[0]["history"] if m["role"] == "user"] == []

    # 第二轮：history 里第一轮的问题只出现一次（修复前会因为列表被原地 append 而重复）
    r2 = client.post("/api/chat", json={"question": "那 PE 呢", "session_id": sid, "conv_id": conv_id})
    assert r2.status_code == 200
    assert [m["content"] for m in seen[1]["history"] if m["role"] == "user"] == ["ROE 是什么"]
    assert seen[1]["question"] == "那 PE 呢"

    saved = json.loads(state.file_for(sid).read_text(encoding="utf-8"))
    assert len(saved["conversations"][conv_id]["messages"]) == 4


def test_chat_rejects_missing_file(client, asked, monkeypatch):
    seen, fake_stream = asked
    monkeypatch.setattr(server, "stream_answer", fake_stream)

    sid = client.get("/api/session").json()["session_id"]
    conv_id = client.post("/api/conversations", json={"session_id": sid, "name": "空对话"}).json()["id"]

    r = client.post("/api/chat", json={"question": "随便问问", "session_id": sid, "conv_id": conv_id})
    assert r.status_code == 400
    assert not seen, "没上传文件就不该调用 LLM"


def test_partial_answer_persisted_but_error_text_is_not(client, monkeypatch):
    """流式中途失败：落已生成的部分回答，错误文案只推给前端、不进历史。

    改造前错误文案是当作正常 token yield 出去的，会被写进 messages，
    下一轮又作为上下文喂回模型。
    """
    def flaky_stream(question, context, history):
        yield "半截回答"
        raise server.LLMStreamError("调用大模型失败：connection reset")

    monkeypatch.setattr(server, "stream_answer", flaky_stream)

    sid = client.get("/api/session").json()["session_id"]
    conv_id = _prepare(client, sid)

    r = client.post("/api/chat", json={"question": "问题", "session_id": sid, "conv_id": conv_id})
    assert r.status_code == 200
    assert "connection reset" in r.text, "错误文案仍然要推给前端"

    msgs = server.sessions[sid]["conversations"][conv_id]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "半截回答", "已生成的部分应当落盘"
    assert "失败" not in msgs[1]["content"], "错误文案绝不能进历史"


def test_empty_answer_is_not_persisted(client, monkeypatch):
    monkeypatch.setattr(server, "stream_answer", lambda q, c, h: iter([]))

    sid = client.get("/api/session").json()["session_id"]
    conv_id = _prepare(client, sid)

    r = client.post("/api/chat", json={"question": "问题", "session_id": sid, "conv_id": conv_id})
    assert r.status_code == 200

    msgs = server.sessions[sid]["conversations"][conv_id]["messages"]
    assert [m["role"] for m in msgs] == ["user"], "空回答不该落一条空的 assistant 消息"


# ── P1-16 上传大小上限 ──

def test_upload_size_prefers_reported_size():
    class Reported:
        size = 123
        file = None

    assert server._upload_size(Reported()) == 123


def test_upload_size_falls_back_to_seek_when_unreported():
    class Stream:
        def __init__(self, data):
            self.file = io.BytesIO(data)

    assert server._upload_size(Stream(b"x" * 7)) == 7


def test_upload_over_size_limit_is_rejected_before_parsing(client, monkeypatch):
    """超限文件必须在上传入口就被挡掉：解析大 PDF 是最贵的一步。"""
    parsed = []
    monkeypatch.setattr(server, "load_file", lambda f: parsed.append(f) or "内容")
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 10)
    monkeypatch.setattr(server, "MAX_UPLOAD_MB", 1)

    sid = client.get("/api/session").json()["session_id"]
    r = client.post("/api/upload",
                    files={"file": ("big.txt", b"x" * 64, "text/plain")},
                    data={"session_id": sid})

    assert r.status_code == 413
    assert "过大" in r.json()["error"], "要给中文提示"
    assert not parsed, "超限文件不该再走解析"


# ── P1-16 缺 API_KEY 时的降级 ──

def test_chat_without_api_key_gives_chinese_hint_and_persists_nothing(client, monkeypatch):
    """缺 API_KEY 时用户要看到中文提示，且提示绝不能落进 messages。"""
    import src.llm as llm
    monkeypatch.setattr(llm, "client", None)

    sid = client.get("/api/session").json()["session_id"]
    conv_id = _prepare(client, sid)

    r = client.post("/api/chat", json={"question": "问题", "session_id": sid, "conv_id": conv_id})
    assert r.status_code == 200
    assert "API_KEY" in r.text, "缺 API_KEY 的中文提示要推给前端"

    msgs = server.sessions[sid]["conversations"][conv_id]["messages"]
    assert [m["role"] for m in msgs] == ["user"], "错误提示不该作为回答落盘"
