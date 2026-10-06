# -*- coding: utf-8 -*-
"""关键词降级检索回归：必须按相关性排序，而不是永远返回文档前两块。"""
import logging
from types import SimpleNamespace

import src.retrieval as retrieval
from src.text_utils import extract_terms, strip_question_words

PE_CHUNK = "市盈率（PE）等于股价除以每股收益，是衡量股票估值高低的重要指标。"
ROE_CHUNK = "净资产收益率（ROE）是净利润与平均股东权益的百分比，反映股东权益的收益水平。"
CHUNKS = [PE_CHUNK, ROE_CHUNK]


# ── 抽词 ──

def test_strip_question_words():
    assert strip_question_words("什么是市盈率").strip() == "市盈率"
    assert strip_question_words("ROE 是什么").strip() == "ROE"


def test_extract_terms_uses_ngrams():
    terms = extract_terms("什么是净资产收益率")
    assert {"净资", "资产", "收益"} <= terms
    # 疑问词已被剥掉，不会留下「什么」「么是」这类垃圾词
    assert "什么" not in terms
    assert "么是" not in terms


# ── 打分排序 ──

def test_ranks_relevant_chunk_first():
    # 旧实现在这里两块都是 0 分，于是回退成「返回前 2 块」，PE 块反而排在前面
    top = retrieval.retrieve_keyword("什么是净资产收益率", CHUNKS)
    assert top, "应该至少命中一块"
    assert top[0] == ROE_CHUNK


def test_returns_empty_when_nothing_matches():
    assert retrieval.retrieve_keyword("量子纠缠的退相干时间", ["完全无关的一段文字内容"]) == []


def test_respects_top_n():
    assert len(retrieval.retrieve_keyword("净资产收益率", [ROE_CHUNK] * 5, top_n=2)) == 2


# ── HyDE 只在真正查向量时才调用 ──

class _FakeCollection:
    def __init__(self):
        self.queries = []

    def count(self):
        return 1

    def query(self, query_texts, n_results):
        self.queries.append(query_texts[0])
        return {"documents": [[ROE_CHUNK]]}


def test_vector_mode_uses_hyde(monkeypatch):
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设性答案文本")
    coll = _FakeCollection()
    out = retrieval.hyde_retrieve("什么是净资产收益率", coll, ROE_CHUNK, use_vector=True)
    assert coll.queries == ["假设性答案文本"]
    assert out == [ROE_CHUNK]


def test_keyword_mode_skips_hyde_and_uses_original_question(monkeypatch):
    called = []
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: called.append(q) or "假设性答案文本")

    out = retrieval.hyde_retrieve("什么是净资产收益率", None, ROE_CHUNK, use_vector=False)

    assert called == [], "关键词模式下不该白花一次 API 调用"
    assert out and out[0] == ROE_CHUNK


# ── 相关性阈值（l2 距离）──

class _DistanceCollection:
    """按给定距离返回候选块，用来确定性地测阈值过滤。"""

    def __init__(self, docs, distances):
        self._docs = docs
        self._distances = distances

    def count(self):
        return len(self._docs)

    def query(self, query_texts, n_results):
        return {"documents": [self._docs], "distances": [self._distances]}


def test_filter_by_distance_keeps_close_drops_far():
    kept = retrieval._filter_by_distance(["近的块", "远的块"], [0.40, 0.90], max_distance=0.55)
    assert kept == ["近的块"]


def test_filter_by_distance_keeps_everything_when_no_distances():
    docs = ["a", "b"]
    assert retrieval._filter_by_distance(docs, [], max_distance=0.55) == docs


def test_vector_results_over_threshold_are_dropped(monkeypatch):
    """阈值内的块保留，超阈值的块不进 context。"""
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _DistanceCollection(["相关的块", "无关的块"], [0.40, 0.62])

    out = retrieval.hyde_retrieve("问题", coll, "全文", use_vector=True)

    assert out == ["相关的块"]


def test_all_above_threshold_falls_back_to_keyword(monkeypatch):
    """向量全部超阈值 → 走关键词兜底；关键词也不命中才最终返回空。"""
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _DistanceCollection(["块A", "块B"], [0.80, 0.90])

    out = retrieval.hyde_retrieve("量子纠缠退相干时间", coll, "完全无关的一段文字", use_vector=True)

    assert out == []


# ── 检索路径标签（评测靠它区分「向量召回」与「关键词兜底」）──

def test_hits_with_path_reports_the_vector_path(monkeypatch):
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _DistanceCollection([ROE_CHUNK], [0.40])

    path, hits = retrieval.hyde_retrieve_hits_with_path(
        "净资产收益率是什么", coll, ROE_CHUNK, use_vector=True)

    assert path == retrieval.PATH_VECTOR
    assert [text for text, _ in hits] == [ROE_CHUNK]


def test_hits_with_path_reports_the_keyword_fallback(monkeypatch):
    """向量全超阈值、靠兜底捞回来：路径必须标成兜底，不能记成向量。

    两条路的分数含义不同（l2 距离 vs 2-gram 打分），混在一起会误判阈值的好坏。
    """
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _DistanceCollection([ROE_CHUNK], [0.90])

    path, hits = retrieval.hyde_retrieve_hits_with_path(
        "净资产收益率是什么", coll, ROE_CHUNK, use_vector=True)

    assert path == retrieval.PATH_KEYWORD
    assert [text for text, _ in hits] == [ROE_CHUNK], "兜底按词面命中，这条应该被捞回来"


def test_hits_with_path_stays_keyword_when_the_fallback_finds_nothing(monkeypatch):
    """兜底也没命中（负样本的样子）：路径仍然是兜底 —— 那条路确实跑了。"""
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _DistanceCollection([ROE_CHUNK], [0.90])

    path, hits = retrieval.hyde_retrieve_hits_with_path(
        "量子纠缠退相干时间", coll, ROE_CHUNK, use_vector=True)

    assert path == retrieval.PATH_KEYWORD
    assert hits == []


def test_hits_with_path_in_pure_keyword_mode():
    path, hits = retrieval.hyde_retrieve_hits_with_path(
        "净资产收益率是什么", None, ROE_CHUNK, use_vector=False)

    assert path == retrieval.PATH_KEYWORD
    assert [text for text, _ in hits] == [ROE_CHUNK]


def test_hits_without_path_keeps_the_old_signature(monkeypatch):
    """`hyde_retrieve_hits` / `hyde_retrieve` 的老签名不能变（调用方与老测试都在用）。"""
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _DistanceCollection([ROE_CHUNK], [0.40])

    assert retrieval.hyde_retrieve_hits(
        "净资产收益率是什么", coll, ROE_CHUNK, use_vector=True) == [(ROE_CHUNK, None)]
    assert retrieval.hyde_retrieve(
        "净资产收益率是什么", coll, ROE_CHUNK, use_vector=True) == [ROE_CHUNK]


# ── 缺 API_KEY 时的降级（P1-16）──

def test_hyde_skips_with_clear_hint_when_client_is_none(monkeypatch, caplog):
    """没配 API_KEY 时 client 是 None：HyDE 直接用原问题，并给出明确中文提示，
    而不是把它混在「HyDE 生成失败」的普通告警里。"""
    monkeypatch.setattr(retrieval, "client", None)

    with caplog.at_level(logging.WARNING, logger="ai_rag"):
        out = retrieval._generate_hyde("什么是净资产收益率")

    assert out == "什么是净资产收益率"
    assert any("API_KEY" in r.getMessage() for r in caplog.records)


# ── HyDE 必须真的生成内容（2026-10-06：推理模型吃光预算导致静默退化）──

def _use_fake_client(monkeypatch, content=None, finish_reason="stop", boom=False):
    """装上假 client，返回记录调用参数的 completions 替身。"""
    completions = SimpleNamespace(calls=[], )

    def create(**kwargs):
        completions.calls.append(kwargs)
        if boom:
            raise RuntimeError("网关 500")
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(content=content))])

    completions.create = create
    monkeypatch.setattr(retrieval, "client", SimpleNamespace(
        chat=SimpleNamespace(completions=completions)))
    return completions


def test_hyde_disables_thinking_so_reasoning_does_not_eat_the_budget(monkeypatch):
    """deepseek-flash 是推理模型：不关思考的话，200 的 cap 会被 reasoning 吃光、
    content 返回空串，HyDE 就变成「原问题直查」而且一声不响（2026-10-06 实测：
    finish_reason=length、reasoning_tokens=200、content 长度 0）。"""
    completions = _use_fake_client(monkeypatch, content="假设性文档片段")

    out = retrieval._generate_hyde("漏损率怎么算")

    assert out == "假设性文档片段"
    sent = completions.calls[0]
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}
    assert sent["max_tokens"] == 200, "cap 留着当防跑飞闸，靠关思考而不是放宽它"


def test_hyde_warns_and_falls_back_when_content_is_empty(monkeypatch, caplog):
    """空内容必须出声：静默退回原问题会让所有「HyDE 口径」的结论变成假的。"""
    _use_fake_client(monkeypatch, content="", finish_reason="length")

    with caplog.at_level(logging.WARNING, logger="ai_rag"):
        out = retrieval._generate_hyde("漏损率怎么算")

    assert out == "漏损率怎么算"
    assert any("空内容" in r.getMessage() for r in caplog.records)


def test_hyde_falls_back_on_api_error(monkeypatch, caplog):
    _use_fake_client(monkeypatch, boom=True)

    with caplog.at_level(logging.WARNING, logger="ai_rag"):
        out = retrieval._generate_hyde("漏损率怎么算")

    assert out == "漏损率怎么算"
    assert any("HyDE 生成失败" in r.getMessage() for r in caplog.records)


# ── 来源元数据（P2）──

class _MetaCollection:
    """返回带 metadatas 的候选块，用来验证来源能一路带到上层。"""

    def __init__(self, docs, distances, metas):
        self._docs, self._distances, self._metas = docs, distances, metas

    def count(self):
        return len(self._docs)

    def query(self, query_texts, n_results):
        return {"documents": [self._docs], "distances": [self._distances],
                "metadatas": [self._metas]}


def test_hits_carry_source_metadata(monkeypatch):
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    meta = {"file_name": "a.md", "chunk_index": 0, "section": "第 1 页"}
    coll = _MetaCollection(["相关的块"], [0.30], [meta])

    hits = retrieval.hyde_retrieve_hits("问题", coll, "全文", use_vector=True)

    assert hits == [("相关的块", meta)]


def test_hits_without_metadata_give_none_not_a_fake_source(monkeypatch):
    """老索引没有 metadatas：来源必须是 None，由上层标「未标注」，不能编。"""
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _MetaCollection(["相关的块"], [0.30], [None])

    hits = retrieval.hyde_retrieve_hits("问题", coll, "全文", use_vector=True)

    assert hits == [("相关的块", None)]


def test_chroma_meta_drops_none_fields():
    """Chroma 的 metadata 不收 None，所以建索引前要把值为 None 的字段删掉。"""
    assert retrieval.chroma_meta({"file_name": "a.md", "section": None}) == {"file_name": "a.md"}
    assert retrieval.chroma_meta(None) == {"file_name": "未知文件"}


def test_hyde_retrieve_still_returns_plain_texts(monkeypatch):
    """旧签名保持兼容：只给文本，老调用方不用改。"""
    monkeypatch.setattr(retrieval, "_generate_hyde", lambda q: "假设答案")
    coll = _MetaCollection(["块A"], [0.30], [{"file_name": "a.md"}])

    assert retrieval.hyde_retrieve("问题", coll, "全文", use_vector=True) == ["块A"], "应提示缺少 API_KEY"
