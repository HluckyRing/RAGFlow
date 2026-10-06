# -*- coding: utf-8 -*-
"""指代消解回归：既要取对轮次，也不能被「这/那/其他」这类词误触发。

后半部分守参考背景装配（P1）：按块边界、按原文序、不给半截块。
"""
import logging

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


# ── 参考背景装配（P1：块边界；P2：来源标签）──

def _block(text, label="未标注"):
    """context 里一块的样式：`【来源：…】` + 换行 + 正文。"""
    return f"【来源：{label}】\n{text}"


def _hits(chunks, meta=None):
    return [(chunk, meta) for chunk in chunks]


def test_context_is_assembled_at_chunk_boundaries(monkeypatch):
    """预算只够两块时，结果正好是整块的两块，且不超过预算。"""
    chunks = ["甲" * 40, "乙" * 40, "丙" * 40]
    monkeypatch.setattr(llm, "hyde_retrieve_hits", lambda *a, **k: _hits(chunks))
    monkeypatch.setattr(llm, "MAX_CONTEXT_LENGTH", 120)

    context, used = llm.retrieve_and_build_context("q", "".join(chunks), None, False)

    assert used == chunks[:2]
    assert context == _block(chunks[0]) + "\n\n" + _block(chunks[1])
    assert len(context) <= 120


def test_context_never_contains_half_a_chunk(monkeypatch):
    """装不下的块整块丢掉 —— 旧实现是从中间硬切，会留下半截话。"""
    first = "甲" * 90
    second = "乙" * 90
    monkeypatch.setattr(llm, "hyde_retrieve_hits", lambda *a, **k: _hits([first, second]))
    monkeypatch.setattr(llm, "MAX_CONTEXT_LENGTH", 150)

    context, used = llm.retrieve_and_build_context("q", first + "\n\n" + second, None, False)

    assert used == [first]
    assert context == _block(first), "宁可比预算短，也不能切出半截块"


def test_context_selects_by_relevance_but_orders_by_document_position(monkeypatch):
    """选中集合按相关度决定；摆进 prompt 时改成原文顺序，跨块的答案才连得上。"""
    first = "前面的内容" * 5
    second = "后面的内容" * 5
    full_text = first + "\n\n" + second
    # 相关度序是反的
    monkeypatch.setattr(llm, "hyde_retrieve_hits", lambda *a, **k: _hits([second, first]))

    context, used = llm.retrieve_and_build_context("q", full_text, None, False)

    assert used == [first, second], "返回的块要与 context 同序"
    assert context.index(first) < context.index(second)


def test_context_keeps_unknown_position_chunks_at_the_end(monkeypatch):
    """索引与当前全文不一致（切片改过但没重建）时，找不到位置的块不许丢。"""
    known = "在原文里的块"
    stale = "索引里遗留的块，当前全文里没有"
    monkeypatch.setattr(llm, "hyde_retrieve_hits", lambda *a, **k: _hits([stale, known]))

    context, used = llm.retrieve_and_build_context("q", known, None, False)

    assert used == [known, stale]
    assert stale in context and known in context


def test_context_truncates_single_chunk_when_budget_is_contradictory(monkeypatch, caplog):
    """MAX_CONTEXT_LENGTH 比单块还小时：只能截断这一块，但必须告警且不能返回空。"""
    chunk = "甲" * 200
    monkeypatch.setattr(llm, "hyde_retrieve_hits", lambda *a, **k: _hits([chunk]))
    monkeypatch.setattr(llm, "MAX_CONTEXT_LENGTH", 50)

    with caplog.at_level(logging.WARNING, logger="ai_rag"):
        context, used = llm.retrieve_and_build_context("q", chunk, None, False)

    assert context.startswith("【来源：未标注】")
    assert context.endswith("...(截断)")
    assert used == [chunk]
    assert any("MAX_CONTEXT_LENGTH" in r.getMessage() for r in caplog.records)


def test_context_is_empty_when_nothing_retrieved(monkeypatch):
    monkeypatch.setattr(llm, "hyde_retrieve_hits", lambda *a, **k: [])

    assert llm.retrieve_and_build_context("q", "全文", None, False) == ("", [])


# ── 来源标签（P2）──

def test_context_labels_each_block_with_its_source(monkeypatch):
    """每块前面要有【来源：…】，引用才可核验（文件名 + 页码/工作表名）。"""
    meta = {"file_name": "01-财务指标.md", "section": "第 2 页"}
    text = "净资产收益率是净利润与平均股东权益的百分比"
    monkeypatch.setattr(llm, "hyde_retrieve_hits", lambda *a, **k: [(text, meta)])

    context, used = llm.retrieve_and_build_context(
        "q", text, None, False, files=[("01-财务指标.md", text)])

    assert context == "【来源：01-财务指标.md · 第 2 页】\n" + text
    assert used == [text]


def test_context_omits_section_when_there_is_none(monkeypatch):
    """没有页/工作表信息的格式（md/csv/docx）只写文件名，不要硬凑一个「第 None 页」。"""
    meta = {"file_name": "05-费用报销流程.md", "section": None}
    monkeypatch.setattr(llm, "hyde_retrieve_hits", lambda *a, **k: [("正文", meta)])

    context, _ = llm.retrieve_and_build_context("q", "正文", None, False)

    assert context.startswith("【来源：05-费用报销流程.md】")
    assert "None" not in context


def test_context_marks_unknown_source_without_inventing_one(monkeypatch):
    """老索引没有元数据时标「未标注」，绝不编造来源 —— 上次引用功能就栽在不准确上。"""
    monkeypatch.setattr(llm, "hyde_retrieve_hits", lambda *a, **k: [("某段正文", None)])

    context, _ = llm.retrieve_and_build_context("q", "某段正文", None, False)

    assert context.startswith("【来源：未标注】")
    assert "某段正文" in context


def test_context_passes_files_through_to_retrieval(monkeypatch):
    """server 传来的 files 必须透传给检索层，否则来源永远是「未标注」。"""
    seen = {}

    def fake_hits(question, collection, full_text, use_vector, top_k=None, files=None):
        seen["files"] = files
        return [("正文", {"file_name": "a.md", "section": None})]

    monkeypatch.setattr(llm, "hyde_retrieve_hits", fake_hits)
    files = [("a.md", "正文")]

    llm.retrieve_and_build_context("q", "正文", None, False, files=files)

    assert seen["files"] == files
