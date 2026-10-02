# -*- coding: utf-8 -*-
"""关键词降级检索回归：必须按相关性排序，而不是永远返回文档前两块。"""
import logging

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


# ── 缺 API_KEY 时的降级（P1-16）──

def test_hyde_skips_with_clear_hint_when_client_is_none(monkeypatch, caplog):
    """没配 API_KEY 时 client 是 None：HyDE 直接用原问题，并给出明确中文提示，
    而不是把它混在「HyDE 生成失败」的普通告警里。"""
    monkeypatch.setattr(retrieval, "client", None)

    with caplog.at_level(logging.WARNING, logger="ai_rag"):
        out = retrieval._generate_hyde("什么是净资产收益率")

    assert out == "什么是净资产收益率"
    assert any("API_KEY" in r.getMessage() for r in caplog.records), "应提示缺少 API_KEY"
