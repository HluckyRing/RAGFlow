# -*- coding: utf-8 -*-
"""可答性判断的回归：只做一件事 —— 判断【材料】里到底有没有能回答问题的信息。

为什么需要它（2026-11-05 实测，见 docs/retrieval-baseline-2026-10-06.md）：
- 域内「看着像但没答案」的提问 **8/8 全被作答**；
- 阈值扫描（0.3/0.5/0.8）与「兜底命中强度」闸门都分不开 —— 重排分数里域内负样本
  0.0004–0.9529 与正样本 0.0284–0.9993 完全重叠，命中词数 ≥5 只能把域内降到 1/8 而误杀 16/36 正样本。

这里的用例把三条纪律钉住：
1. 只认「能」/「不能」，解析要稳（模型爱写「不能，因为…」）；
2. 这次调用必须关掉思考（deepseek-flash 会把小 max_tokens 花在 reasoning 上，content 返回空串）；
3. **任何异常/空内容都不许拒答**，只能记 WARNING 退回「按老行为作答」——一次判定故障
   不能把整个问答堵死。
"""
from types import SimpleNamespace

import pytest

import src.answerability as answerability
import src.server as server
import src.state as state


def _fake_client(monkeypatch, content="能", finish_reason="stop", boom=False):
    completions = SimpleNamespace(calls=[])

    def create(**kwargs):
        completions.calls.append(kwargs)
        if boom:
            raise RuntimeError("网关 500")
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(content=content))])

    completions.create = create
    monkeypatch.setattr(answerability, "client", SimpleNamespace(
        chat=SimpleNamespace(completions=completions)))
    return completions


# ── 判定解析 ──

@pytest.mark.parametrize("raw, expected", [
    ("能", True),
    ("能。", True),
    ("  能  ", True),
    ("可以", True),
    ("能，材料里有 RPO 与 RTO 的定义", True),
    ("不能", False),
    ("不能。", False),
    ("不能，材料只写了响应时效，没有违约条款", False),
    ("否", False),
])
def test_parse_verdict_recognises_both_answers(raw, expected):
    assert answerability.parse_verdict(raw) is expected


@pytest.mark.parametrize("raw", ["", "   ", None, "嗯…我想想", "maybe"])
def test_parse_verdict_is_unknown_for_garbage(raw):
    """看不懂就返回 None（= 不知道），绝不猜成「不能」把好问题拒掉。"""
    assert answerability.parse_verdict(raw) is None


# ── 调用约定 ──

def test_judge_disables_thinking_and_keeps_the_budget_tiny(monkeypatch):
    """这次调用必须关思考：deepseek-flash 是推理模型，200 的 cap 曾被 reasoning 吃光。"""
    completions = _fake_client(monkeypatch, content="不能")

    verdict = answerability.judge_answerable("年会奖品预算谁批", "【来源：a.md】\n报销流程…")

    assert verdict is False
    sent = completions.calls[0]
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}
    assert sent["temperature"] == 0, "判定要尽量确定，不要随机性"
    assert sent["max_tokens"] <= 32, "只要两个字的答案，别给它编小作文的空间"


def test_judge_returns_true_when_the_material_answers(monkeypatch):
    _fake_client(monkeypatch, content="能")

    assert answerability.judge_answerable("RPO 是多少", "【来源：a.md】\nRPO 为 15 分钟") is True


def test_judge_skips_the_call_for_empty_context(monkeypatch):
    """没有材料就没什么可判的（检索层已经决定返回空），不该白花一次调用。"""
    completions = _fake_client(monkeypatch)

    assert answerability.judge_answerable("问题", "   ") is True
    assert completions.calls == []


# ── 降级必须出声 ──

def test_judge_returns_none_with_warning_when_api_key_is_missing(monkeypatch, caplog):
    monkeypatch.setattr(answerability, "client", None)

    with caplog.at_level("WARNING", logger="ai_rag"):
        verdict = answerability.judge_answerable("问题", "材料")

    assert verdict is None, "没配 API_KEY 时不能拒答，交给上层按老行为处理"
    assert any("API_KEY" in r.getMessage() for r in caplog.records)


def test_judge_returns_none_with_warning_when_the_call_fails(monkeypatch, caplog):
    _fake_client(monkeypatch, boom=True)

    with caplog.at_level("WARNING", logger="ai_rag"):
        verdict = answerability.judge_answerable("问题", "材料")

    assert verdict is None
    assert any("可答性判断" in r.getMessage() for r in caplog.records)


def test_judge_returns_none_with_warning_when_content_is_empty(monkeypatch, caplog):
    """推理模型吃光预算时 content 是空串 —— 必须出声，不能静默当成「能」。"""
    _fake_client(monkeypatch, content="", finish_reason="length")

    with caplog.at_level("WARNING", logger="ai_rag"):
        verdict = answerability.judge_answerable("问题", "材料")

    assert verdict is None
    assert any("空内容" in r.getMessage() for r in caplog.records)


# ── 服务端接入（拒答走已有的「未找到相关内容」路径）──

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

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


def _prepare(client, sid):
    conv_id = client.post("/api/conversations", json={"session_id": sid, "name": "T"}).json()["id"]
    conv = server.sessions[sid]["conversations"][conv_id]
    conv["files"] = [{"file_name": "a.txt", "file_text": "x"}]
    conv["full_text"] = "x"
    return conv_id


def test_chat_refuses_when_the_material_cannot_answer(client, monkeypatch):
    """判「不能」→ 走已有的「未找到相关内容」，而且提问**不入库**（与检索为空时一致）。"""
    monkeypatch.setattr(answerability, "ANSWERABILITY_ENABLED", True)
    monkeypatch.setattr(answerability, "judge_answerable", lambda q, c: False)
    called = []
    monkeypatch.setattr(server, "stream_answer", lambda *a: called.append(a) or iter(["x"]))

    sid = client.get("/api/session").json()["session_id"]
    conv_id = _prepare(client, sid)

    resp = client.post("/api/chat", json={"session_id": sid, "conv_id": conv_id,
                                         "question": "年会奖品预算谁批"})

    assert resp.status_code == 400
    assert "未找到相关内容" in resp.text
    assert called == [], "判「不能」就不该再去生成回答"
    history = server.sessions[sid]["conversations"][conv_id]["messages"]
    assert [m for m in history if m["role"] == "user"] == [], "拒答的提问不入库"


def _sse_events(resp):
    """把 SSE 响应体还原成事件列表（中文在 JSON 里是 \\uXXXX 转义，不能直接搜字符串）。"""
    import json as _json

    events = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            events.append(_json.loads(line[len("data: "):]))
    return events


def test_chat_still_answers_when_the_judge_is_unknown(client, monkeypatch):
    """判定不可用（None）必须 fail-open：宁可像老行为一样作答，也不让判定故障堵死问答。"""
    monkeypatch.setattr(answerability, "ANSWERABILITY_ENABLED", True)
    monkeypatch.setattr(answerability, "judge_answerable", lambda q, c: None)
    monkeypatch.setattr(server, "stream_answer", lambda *a: iter(["答案"]))

    sid = client.get("/api/session").json()["session_id"]
    conv_id = _prepare(client, sid)

    resp = client.post("/api/chat", json={"session_id": sid, "conv_id": conv_id,
                                         "question": "ROE 是什么"})

    assert resp.status_code == 200
    tokens = "".join(e.get("content", "") for e in _sse_events(resp) if e.get("type") == "token")
    assert tokens == "答案"


def test_chat_skips_the_judge_when_the_switch_is_off(client, monkeypatch):
    monkeypatch.setattr(answerability, "ANSWERABILITY_ENABLED", False)
    touched = []
    monkeypatch.setattr(answerability, "judge_answerable", lambda q, c: touched.append(q))
    monkeypatch.setattr(server, "stream_answer", lambda *a: iter(["答案"]))

    sid = client.get("/api/session").json()["session_id"]
    conv_id = _prepare(client, sid)

    resp = client.post("/api/chat", json={"session_id": sid, "conv_id": conv_id,
                                         "question": "ROE 是什么"})

    assert resp.status_code == 200
    assert touched == [], "开关关掉时一次都不许调用"
