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


# ── 多轮 + 复合：拆出来的每条子问也要过一遍指代消解 ──

HISTORY = [{"role": "user", "content": "年假有几天？"},
           {"role": "assistant", "content": "司龄 1 至 3 年每年 5 天。"}]


def test_history_reaches_the_decomposer(monkeypatch):
    seen = {}

    def fake(question, collection, full_text, use_vector, top_k=None, files=None, use_hyde=True):
        return [_hit("x", 0)]

    def fake_decompose(question, history=None):
        seen["history"] = history
        return ["甲子问", "乙子问"]

    monkeypatch.setattr(llm, "hyde_retrieve_hits", fake)
    llm._retrieve_hits("那这个规定还有效吗？乙问？", None, "", False, 5, None,
                       fake_decompose, history=HISTORY)

    assert seen["history"] == HISTORY, "拆问要拿到历史，否则省略式追问补不全"


def test_each_decomposed_question_gets_coreference_resolution(monkeypatch):
    """省略式追问被拆成单条后，主语还在上一轮里 —— 每条子问都要过 resolve_query。"""
    asked = []

    def fake(question, collection, full_text, use_vector, top_k=None, files=None, use_hyde=True):
        asked.append(question)
        return [_hit("x", 0)]

    monkeypatch.setattr(llm, "hyde_retrieve_hits", fake)
    llm._retrieve_hits("那这个规定还有效吗？乙问？", None, "", False, 5, None,
                       lambda question, history=None: ["那这个规定还有效吗", "乙问"],
                       history=HISTORY)

    assert asked == ["年假有几天 那这个规定还有效吗", "乙问"], \
        "第一条子问被 resolve_query 补上了上一轮的实体；第二条不需要消解就别动它"


def test_decomposer_that_only_takes_one_argument_still_works(monkeypatch):
    """历史版本 / 评测替身可能只接受一个参数，不能因此把问答搞崩。"""
    monkeypatch.setattr(llm, "hyde_retrieve_hits",
                        lambda *args, **kwargs: [_hit("x", 0)])

    hits = llm._retrieve_hits("甲？乙？", None, "", False, 5, None,
                              lambda question: ["甲", "乙"], history=HISTORY)

    assert hits
