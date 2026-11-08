# -*- coding: utf-8 -*-
"""线上「每问分路检索」的接线：单问行为必须一个字都不变。

三条边界（都在 `llm._retrieve_hits` 里）：
- 拆问关着 / 只拆出一条 → 走**今天那条路**（HyDE + 双路融合 + intersect 闸门），
  不多花一次 embedding，也不掉精度；
- 拆成 ≥2 问 → 每问**单探针**检索（不调 HyDE，否则延迟 n 倍），再按名额轮流合并；
- 分路一条都没捞到 → 退回整句检索，绝不让一次拆问把问答搞空。
"""
import pytest

import src.llm as llm
from src.hits_merge import interleave_hits


def _hit(file_name, chunk_index, text="正文"):
    return (text, {"file_name": file_name, "chunk_index": chunk_index})


@pytest.fixture
def fake_retrieval(monkeypatch):
    """把 `llm` 里用到的检索函数换成替身，记下每次调用的参数。"""
    calls = []

    def fake_hyde_retrieve_hits(question, collection, full_text, use_vector,
                                top_k=None, files=None, use_hyde=True):
        calls.append({"question": question, "use_hyde": use_hyde, "top_k": top_k})
        if use_hyde:
            return [_hit("整体", 0)]
        return [_hit(f"分路{question}", 0)]

    monkeypatch.setattr(llm, "hyde_retrieve_hits", fake_hyde_retrieve_hits)
    return calls


def test_no_decomposer_keeps_todays_path_exactly(fake_retrieval):
    hits = llm._retrieve_hits("年假有几天？迟到怎么算？", None, "", False, 5, None, None)

    assert [call["question"] for call in fake_retrieval] == ["年假有几天？迟到怎么算？"]
    assert fake_retrieval[0]["use_hyde"] is True, "没开拆问时仍然走 HyDE + 融合"
    assert hits == [_hit("整体", 0)]


def test_single_question_result_also_keeps_todays_path(fake_retrieval):
    """拆问说「只有一问」时不许改道 —— 否则单问提问会白掉 HyDE 与 intersect 闸门。"""
    hits = llm._retrieve_hits("年假有几天", None, "", False, 5, None,
                              lambda question: ["年假有几天"])

    assert len(fake_retrieval) == 1
    assert fake_retrieval[0]["use_hyde"] is True
    assert hits == [_hit("整体", 0)]


def test_compound_question_retrieves_each_part_without_hyde(fake_retrieval):
    hits = llm._retrieve_hits("年假有几天？迟到怎么算？", None, "", False, 5, None,
                              lambda question: ["年假有几天", "迟到怎么算"])

    assert [call["question"] for call in fake_retrieval] == ["年假有几天", "迟到怎么算"]
    assert all(call["use_hyde"] is False for call in fake_retrieval), \
        "每问都调 HyDE 会让延迟变成 n 倍"
    assert [hit[1]["file_name"] for hit in hits] == ["分路年假有几天", "分路迟到怎么算"]


def test_compound_merge_respects_the_context_budget(monkeypatch):
    def fake(question, collection, full_text, use_vector, top_k=None, files=None, use_hyde=True):
        return [_hit(f"{question}-{index}", index) for index in range(5)]

    monkeypatch.setattr(llm, "hyde_retrieve_hits", fake)
    hits = llm._retrieve_hits("甲？乙？丙？", None, "", False, 5, None,
                              lambda question: ["甲", "乙", "丙"])

    assert len(hits) == 5, "三问各 5 块，也只能进 5 块 —— 和现状同一个上下文预算"
    assert len(interleave_hits([hits], 5)) == 5


def test_empty_parts_fall_back_to_the_whole_question(monkeypatch, caplog):
    """分路全空时退回整句：宁可像今天一样答，也不许因为拆问把问答搞空。"""
    calls = []

    def fake(question, collection, full_text, use_vector, top_k=None, files=None, use_hyde=True):
        calls.append(use_hyde)
        return [_hit("整体", 0)] if use_hyde else []      # 分路一条都没有

    monkeypatch.setattr(llm, "hyde_retrieve_hits", fake)
    hits = llm._retrieve_hits("年假有几天？迟到怎么算？", None, "", False, 5, None,
                              lambda question: ["甲", "乙"])

    assert hits == [_hit("整体", 0)]
    assert calls == [False, False, True], "先试分路（各单探针），全空后回到整句"
    assert "退回整句检索" in caplog.text


def test_decomposer_exception_falls_back_to_the_whole_question(fake_retrieval, caplog):
    def boom(question):
        raise RuntimeError("拆问炸了")

    hits = llm._retrieve_hits("年假有几天？迟到怎么算？", None, "", False, 5, None, boom)

    assert hits == [_hit("整体", 0)]
    assert "拆问抛错" in caplog.text


def test_decompose_flag_off_returns_no_decomposer(monkeypatch):
    import src.server as server_mod
    import src.config as config

    monkeypatch.setattr(config, "DECOMPOSE_ENABLED", False)
    assert server_mod._decomposer() is None

    monkeypatch.setattr(config, "DECOMPOSE_ENABLED", True)
    assert callable(server_mod._decomposer())
