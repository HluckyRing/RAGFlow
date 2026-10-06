# -*- coding: utf-8 -*-
"""跨编码器精排的回归（不加载模型 —— 仓库约定测试套件不碰 torch）。

重点守三条性质：
- **排序与阈值**：按分数降序、只留达标的前 top_n 条；
- **两种「没有结果」语义不同**：模型不可用返回 None（调用方退回距离阈值），用了但一条都不
  达标返回空列表（交给关键词兜底）—— 混起来就会要么把远块塞给模型、要么整批丢掉；
- **不可用要出声**：加载失败/打分失败都记 WARNING，并记住失败不再反复重试。
"""
import logging

import pytest

from src import rerank


# ── 排序与阈值 ──

def test_rerank_orders_by_score_and_keeps_top_n(monkeypatch):
    monkeypatch.setattr(rerank, "score_pairs", lambda q, texts: [0.1, 0.9, 0.5])
    hits = [("块A", None), ("块B", {"f": 1}), ("块C", None)]

    got = rerank.rerank("问题", hits, top_n=2, min_score=0.0)

    assert [text for text, _, _ in got] == ["块B", "块C"]
    assert got[0][1] == {"f": 1}, "meta 要跟着块一起走（引用还要靠它）"
    assert got[0][2] == pytest.approx(0.9)


def test_rerank_drops_candidates_below_min_score(monkeypatch):
    monkeypatch.setattr(rerank, "score_pairs", lambda q, texts: [0.9, 0.2])

    got = rerank.rerank("问题", [("A", None), ("B", None)], top_n=5, min_score=0.5)

    assert [text for text, _, _ in got] == ["A"]


def test_rerank_empty_candidates_is_empty_not_none():
    """没有候选 ≠ 模型不可用：前者是「这批都没东西」，后者要退回距离阈值。"""
    assert rerank.rerank("问题", [], top_n=5, min_score=0.5) == []


# ── 「模型不可用」与「都不达标」必须分开 ──

def test_rerank_returns_none_when_scores_unavailable(monkeypatch):
    monkeypatch.setattr(rerank, "score_pairs", lambda q, texts: None)

    assert rerank.rerank("问题", [("A", None)], top_n=5, min_score=0.5) is None


def test_rerank_returns_empty_when_nothing_passes_the_threshold(monkeypatch):
    monkeypatch.setattr(rerank, "score_pairs", lambda q, texts: [0.1, 0.2])

    assert rerank.rerank("问题", [("A", None), ("B", None)], top_n=5, min_score=0.5) == []


# ── 打分本身 ──

def test_score_pairs_returns_none_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(rerank, "_get_scorer", lambda *a, **k: None)

    assert rerank.score_pairs("问题", ["A"]) is None


def test_score_pairs_returns_none_and_warns_when_predict_fails(monkeypatch, caplog):
    class Boom:
        def predict(self, pairs):
            raise RuntimeError("内存不足")

    monkeypatch.setattr(rerank, "_get_scorer", lambda *a, **k: Boom())

    with caplog.at_level(logging.WARNING, logger="ai_rag"):
        got = rerank.score_pairs("问题", ["A"])

    assert got is None
    assert any("reranker 打分失败" in r.getMessage() for r in caplog.records)


def test_score_pairs_passes_query_text_pairs(monkeypatch):
    seen = {}

    class Recorder:
        def predict(self, pairs):
            seen["pairs"] = pairs
            return [0.7, 0.3]

    monkeypatch.setattr(rerank, "_get_scorer", lambda *a, **k: Recorder())

    assert rerank.score_pairs("原问题", ["块A", "块B"]) == [pytest.approx(0.7),
                                                            pytest.approx(0.3)]
    assert seen["pairs"] == [("原问题", "块A"), ("原问题", "块B")], "跨编码器吃的是 query×块 对"


# ── 加载失败：出声、且别反复重试 ──

def test_get_scorer_warns_and_remembers_failure(monkeypatch, caplog):
    monkeypatch.setattr(rerank, "_scorer", None)
    monkeypatch.setattr(rerank, "_scorer_failed", False)
    calls = []

    def boom(model_name):
        calls.append(model_name)
        raise RuntimeError("模型没下载")

    monkeypatch.setattr(rerank, "_make_cross_encoder", boom)

    with caplog.at_level(logging.WARNING, logger="ai_rag"):
        first = rerank._get_scorer("某个模型")
        second = rerank._get_scorer("某个模型")

    assert first is None and second is None
    assert calls == ["某个模型"], "失败过就别每次问答都重试一遍加载"
    assert any("reranker 加载失败" in r.getMessage() for r in caplog.records)
