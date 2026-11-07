# -*- coding: utf-8 -*-
"""拒答事件的回归：服务端说「这是拒答」，前端才单独渲染 —— 不靠事后猜文案。

背景（2026-11-06）：`ANSWERABILITY_CHECK` 默认关之后，不能答的问题由**生成那次调用自己拒答**。
如果只靠「回答里有没有『没有提到』」这种启发式去判断，正常回答里引用到那几个字就会被误标成拒答。
所以改成**结构化前缀**：`prompts.REFUSAL_MARKER`（`【无答案】`）+ 服务端推 `refused` 事件。

这里钉住三件事：
1. 判定要在**流式开头**做：标记可能被拆在两个 token 里（`【无答` + `案】`），不许漏判；
2. 判成拒答后正文要**去掉标记**再给用户，并且落盘带 `refused: true`（刷新后还要能渲染）；
3. 正常回答一个字节都不许改，也不许冒 `refused` 事件。
"""
import json

import pytest
from fastapi.testclient import TestClient

import src.llm as llm
import src.server as server
import src.state as state
from src.prompts import REFUSAL_MARKER


# ── 开头判定（纯函数）──

@pytest.mark.parametrize("pending, expected", [
    (REFUSAL_MARKER, True),
    (REFUSAL_MARKER + "参考背景只写了保留策略", True),
    ("【无答", None),                       # 标记还没收全
    ("【", None),
    ("", None),
    ("  ", None),
    ("根据《a.md》，", False),
    ("根", False),
])
def test_refusal_state_three_way(pending, expected):
    assert llm.refusal_state(pending) is expected


def test_refusal_state_tolerates_markdown_emphasis():
    """模型爱把标记写成加粗的 `**【无答案】**`，判定要认，还得能把标记剥掉。"""
    assert llm.refusal_state("**【无答案】**") is True
    assert llm.refusal_state("## 【无答案】") is True

    assert llm.strip_refusal_marker("**【无答案】**材料里没有") == "材料里没有"
    assert llm.strip_refusal_marker("【无答案】  材料里没有") == "材料里没有"


def test_strip_refusal_marker_leaves_normal_answers_alone():
    normal = "根据《01-财务指标.md》，市盈率等于股价除以每股收益。"

    assert llm.strip_refusal_marker(normal) == normal


def test_qa_prompt_asks_for_the_marker():
    assert REFUSAL_MARKER in llm.QA_SYSTEM_PROMPT_TEMPLATE
    assert "{context}" in llm.QA_SYSTEM_PROMPT_TEMPLATE, "占位符不能被标记替换掉"


# ── 端到端（SSE 事件 + 落盘）──

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", tmp_path / "kb_state.json")
    state._cache.clear()
    state._locks.clear()
    server.sessions.clear()
    monkeypatch.setattr(server, "init_vector_store", lambda name=None: (None, False))
    monkeypatch.setattr(server, "retrieve_and_build_context",
                        lambda *a, **k: ("背景：ROE 是净利润/股东权益",
                                         [{"text": "背景", "file_name": "a.txt"}]))
    yield TestClient(server.app)
    server.sessions.clear()
    state._cache.clear()
    state._locks.clear()


def _prepare(client):
    sid = client.get("/api/session").json()["session_id"]
    conv_id = client.post("/api/conversations",
                          json={"session_id": sid, "name": "T"}).json()["id"]
    conv = server.sessions[sid]["conversations"][conv_id]
    conv["files"] = [{"file_name": "a.txt", "file_text": "x"}]
    conv["full_text"] = "x"
    return sid, conv_id


def _tokens(tokens):
    def fake(question, context, history):
        for token in tokens:
            yield token
    return fake


def _events(resp):
    return [json.loads(line[len("data: "):]) for line in resp.text.splitlines()
            if line.startswith("data: ")]


def test_refusal_pushes_event_and_strips_the_marker(client, monkeypatch):
    monkeypatch.setattr(server, "stream_answer",
                        _tokens([REFUSAL_MARKER, "材料里只有保留策略，", "没有异地机房步骤。"]))
    sid, conv_id = _prepare(client)

    resp = client.post("/api/chat", json={"session_id": sid, "conv_id": conv_id,
                                         "question": "异地机房怎么恢复"})

    events = _events(resp)
    kinds = [e["type"] for e in events]
    assert "refused" in kinds, "拒答必须推 refused 事件，前端才知道要特殊渲染"
    assert kinds.index("refused") < kinds.index("token"), "事件要在正文之前到"
    answer = "".join(e.get("content", "") for e in events if e["type"] == "token")
    assert answer == "材料里只有保留策略，没有异地机房步骤。", "标记不能留在正文里"
    saved = server.sessions[sid]["conversations"][conv_id]["messages"][-1]
    assert saved["refused"] is True, "刷新后还要能渲染成拒答"
    assert REFUSAL_MARKER not in saved["content"]


def test_refusal_marker_split_across_tokens_is_still_detected(client, monkeypatch):
    """标记被拆在两个 token 里（`【无答` + `案】`）也要判出来 —— 一拿到首 token 就下结论会漏。"""
    monkeypatch.setattr(server, "stream_answer",
                        _tokens(["【无答", "案】", "材料里没有相关内容。"]))
    sid, conv_id = _prepare(client)

    resp = client.post("/api/chat", json={"session_id": sid, "conv_id": conv_id,
                                         "question": "随便问问"})

    events = _events(resp)
    assert any(e["type"] == "refused" for e in events)
    answer = "".join(e.get("content", "") for e in events if e["type"] == "token")
    assert answer == "材料里没有相关内容。"


def test_normal_answer_is_untouched_and_not_flagged(client, monkeypatch):
    monkeypatch.setattr(server, "stream_answer",
                        _tokens(["根据《01-财务指标.md》，", "市盈率等于股价除以每股收益。"]))
    sid, conv_id = _prepare(client)

    resp = client.post("/api/chat", json={"session_id": sid, "conv_id": conv_id,
                                         "question": "市盈率怎么算"})

    events = _events(resp)
    assert not any(e["type"] == "refused" for e in events)
    answer = "".join(e.get("content", "") for e in events if e["type"] == "token")
    assert answer == "根据《01-财务指标.md》，市盈率等于股价除以每股收益。"
    saved = server.sessions[sid]["conversations"][conv_id]["messages"][-1]
    assert "refused" not in saved


def test_marker_in_the_middle_is_not_a_refusal(client, monkeypatch):
    """只有**开头**的标记算拒答：正文中间提到这几个字属于正常回答。"""
    monkeypatch.setattr(server, "stream_answer",
                        _tokens(["根据《a.md》，", f"原文里写着{REFUSAL_MARKER}这种标记的用法。"]))
    sid, conv_id = _prepare(client)

    resp = client.post("/api/chat", json={"session_id": sid, "conv_id": conv_id,
                                         "question": "标记怎么用"})

    events = _events(resp)
    assert not any(e["type"] == "refused" for e in events)


def test_bare_marker_only_answer_still_marks_refusal(client, monkeypatch):
    """模型只回了标记本身（没有解释）也要算拒答；正文为空就不落一条空消息。"""
    monkeypatch.setattr(server, "stream_answer", _tokens([REFUSAL_MARKER]))
    sid, conv_id = _prepare(client)

    resp = client.post("/api/chat", json={"session_id": sid, "conv_id": conv_id,
                                         "question": "无关问题"})

    assert any(e["type"] == "refused" for e in _events(resp))
    messages = server.sessions[sid]["conversations"][conv_id]["messages"]
    assert [m for m in messages if m["role"] == "assistant"] == [], "空回答不落一条空消息"
