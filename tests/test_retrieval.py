# -*- coding: utf-8 -*-
"""关键词降级检索回归：必须按相关性排序，而不是永远返回文档前两块。"""
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
