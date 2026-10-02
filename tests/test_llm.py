# -*- coding: utf-8 -*-
"""指代消解回归：既要取对轮次，也不能被「这/那/其他」这类词误触发。"""
import pytest

import src.llm as llm


def _hist(*questions):
    msgs = []
    for q in questions:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": "(略)"})
    return msgs


def test_uses_immediately_previous_question():
    """history 是不含当前提问的快照，所以上一轮就是最后一条用户消息。"""
    history = _hist("什么是净资产收益率", "什么是市盈率")
    out = llm.resolve_query("它的计算公式是什么", history)
    assert "市盈率" in out
    assert "净资产收益率" not in out


def test_entity_extraction_strips_question_words():
    assert llm.extract_entities("什么是市盈率") == ["市盈率"]
    assert llm.extract_entities("请问净资产收益率是什么意思") == ["净资产收益率"]


def test_plain_deictic_does_not_trigger_coref():
    history = _hist("什么是市盈率")
    assert llm.resolve_query("这是什么", history) == "这是什么"


def test_qita_does_not_trigger_coref():
    """「其他」里含「他」，但不是指代。"""
    history = _hist("什么是市盈率")
    assert llm.resolve_query("其他指标有哪些", history) == "其他指标有哪些"


def test_long_question_is_left_alone():
    history = _hist("什么是市盈率")
    q = "请详细说明一下这些财务指标在实际投资分析当中的具体用法"
    assert llm.resolve_query(q, history) == q


def test_no_history_is_safe():
    assert llm.resolve_query("它的计算公式是什么", []) == "它的计算公式是什么"


def test_pronoun_late_in_question_still_triggers():
    """指代出现在句中偏后也要能识别，不能被位置规则漏掉。"""
    history = _hist("什么是市盈率")
    out = llm.resolve_query("请帮我详细解释一下它", history)
    assert "市盈率" in out


# ── 流式健壮性 ──

class _Delta:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, choices):
        self.choices = choices


def test_usage_only_chunk_is_skipped(monkeypatch):
    """choices 为空的 usage-only chunk 不该让流式回答崩掉（原来是 IndexError）。"""
    stream = [
        _Chunk([]),                       # usage-only
        _Chunk([_Choice("净资")]),
        _Chunk([_Choice(None)]),          # 空 delta
        _Chunk([_Choice("产")]),
    ]
    monkeypatch.setattr(llm.client.chat.completions, "create", lambda **kwargs: iter(stream))
    assert list(llm.stream_answer("q", "ctx", [])) == ["净资", "产"]


def test_stream_error_raises_instead_of_yielding_error_text(monkeypatch):
    """失败要抛 LLMStreamError，而不是把错误文案当成回答 yield 出去。"""
    def boom(**kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(llm.client.chat.completions, "create", boom)

    with pytest.raises(llm.LLMStreamError) as ei:
        list(llm.stream_answer("q", "ctx", []))
    assert "connection reset" in str(ei.value)


def test_error_mid_stream_still_returns_earlier_tokens(monkeypatch):
    def gen(**kwargs):
        yield _Chunk([_Choice("半截")])
        raise RuntimeError("mid-stream boom")

    monkeypatch.setattr(llm.client.chat.completions, "create", gen)

    got = []
    with pytest.raises(llm.LLMStreamError):
        for token in llm.stream_answer("q", "ctx", []):
            got.append(token)
    assert got == ["半截"], "出错前已经产出的 token 应当照常返回"


def test_stream_answer_without_api_key_raises_chinese_hint(monkeypatch):
    """没配 API_KEY 时 client 是 None，要抛 LLMStreamError 给用户看中文提示，
    而不是把 AttributeError 当成「调用大模型失败」糊弄过去。"""
    monkeypatch.setattr(llm, "client", None)

    with pytest.raises(llm.LLMStreamError) as ei:
        list(llm.stream_answer("q", "ctx", []))

    assert "API_KEY" in str(ei.value)
