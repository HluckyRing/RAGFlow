# -*- coding: utf-8 -*-
"""跨文件取样（`src/coverage.py` + `llm.ensure_file_coverage`）的回归。

这一刀治的是**真实日志里发现的静默失败**：13 份文件的语料问「该知识库的主要内容是什么」，
装配后的 context 只覆盖 12 份，而且没有任何提示。判据不是「答案变好」而是「覆盖不漏」，
所以这里钉住的是几个口径：

- 闸门 `looks_enumerative` 只决定「要不要多花一次补位查询」，**不参与判对错**；
- 按文件轮流取样：先每份 1 块，再轮到第 2 块（不是按相似度重新排一遍）；
- 开关关着 / 单文件 / 非枚举提问 → **原样返回**，一次查询都不多花；
- 块预算不变：补位块排最前，总数仍截到 `top_k`；
- 补位只针对缺的那几份，且受 `COVERAGE_BACKFILL_MAX` 限制；不许重复块。
"""
import pytest

import src.llm as llm
from src.coverage import (
    ENUMERATION_MARKERS,
    coverage_report,
    file_names_of,
    looks_enumerative,
    missing_files,
    sample_per_file,
)


def _hit(name, index=0, text=None):
    return (text or f"{name}#{index}", {"file_name": name, "chunk_index": index})


# ── 闸门 ──

@pytest.mark.parametrize("question", [
    "列出所有的实验",
    "分别有哪些实验",
    "一共有几个实验",
    "用三点总结主要内容",
    "帮我整理第一章的内容",
    "该知识库的主要内容是什么",
])
def test_enumerative_questions_are_recognised(question):
    assert looks_enumerative(question) is True


@pytest.mark.parametrize("question", [
    "实验一干啥的",
    "数据库评分标准是什么",
    "生产者和消费者有什么区别，谁先谁后",
    "",
])
def test_non_enumerative_questions_are_left_alone(question):
    assert looks_enumerative(question) is False


def test_marker_table_is_the_only_knob():
    assert looks_enumerative("随便一句话", markers=("随便",)) is True
    assert "列出" in ENUMERATION_MARKERS


# ── 按文件取样 ──

def test_sample_per_file_gives_every_file_one_block_before_any_gets_two():
    hits = [_hit("甲", 0), _hit("甲", 1), _hit("甲", 2), _hit("乙", 0), _hit("丙", 0)]
    picked = sample_per_file(hits, 3)
    assert [meta["file_name"] for _text, meta in picked] == ["甲", "乙", "丙"]

    picked = sample_per_file(hits, 4)
    assert [meta["file_name"] for _text, meta in picked] == ["甲", "乙", "丙", "甲"]


def test_sample_per_file_respects_budget_and_empty_input():
    hits = [_hit("甲", 0), _hit("乙", 0)]
    assert sample_per_file(hits, 1) == [_hit("甲", 0)]
    assert sample_per_file([], 5) == []
    assert sample_per_file(hits, 0) == []


def test_sample_per_file_keeps_sourceless_chunks():
    """老索引没有 metadatas 时 `meta` 是 None —— 这些块不许整批消失。"""
    hits = [("孤儿块", None), _hit("甲", 0)]
    picked = sample_per_file(hits, 2)
    assert len(picked) == 2
    assert picked[0][0] == "孤儿块"


# ── 覆盖盘点 ──

def test_file_names_of_dedupes_and_skips_missing_meta():
    hits = [_hit("甲", 0), _hit("甲", 1), ("孤儿", None), _hit("乙", 0)]
    assert file_names_of(hits) == ["甲", "乙"]


def test_missing_files_and_coverage_report():
    hits = [_hit("甲", 0), _hit("乙", 0)]
    assert missing_files(hits, ["甲", "乙", "丙"]) == ["丙"]
    assert coverage_report(hits, ["甲", "乙", "丙"]) == {
        "covered": 2, "total": 3, "missing": ["丙"], "blocks": 2}
    assert coverage_report([], []) == {"covered": 0, "total": 0, "missing": [], "blocks": 0}


# ── 接线：ensure_file_coverage ──

FILES = [("甲.md", "甲正文"), ("乙.md", "乙正文"), ("丙.md", "丙正文")]


@pytest.fixture
def switch_on(monkeypatch):
    monkeypatch.setattr(llm, "COVERAGE_SAMPLING", True)
    monkeypatch.setattr(llm, "COVERAGE_BACKFILL_MAX", 8)
    return True


def test_switch_off_returns_hits_untouched(monkeypatch):
    monkeypatch.setattr(llm, "COVERAGE_SAMPLING", False)
    calls = []
    hits = [_hit("甲", 0)]
    out = llm.ensure_file_coverage("列出所有的实验", None, FILES, hits, 5,
                                   backfill=lambda *a, **k: calls.append(a) or [])
    assert out is hits
    assert calls == [], "开关关着时连补位函数都不许调用"


def test_non_enumerative_question_is_left_untouched(switch_on):
    calls = []
    hits = [_hit("甲", 0)]
    out = llm.ensure_file_coverage("实验一干啥的", None, FILES, hits, 5,
                                   backfill=lambda *a, **k: calls.append(a) or [])
    assert out is hits
    assert calls == []


def test_single_file_corpus_is_left_untouched(switch_on):
    calls = []
    hits = [_hit("甲", 0)]
    out = llm.ensure_file_coverage("列出所有的实验", None, [("甲.md", "甲正文")], hits, 5,
                                   backfill=lambda *a, **k: calls.append(a) or [])
    assert out is hits
    assert calls == []


def test_backfill_only_the_missing_files_and_put_them_first(switch_on):
    calls = []

    def backfill(_collection, _question, name, top_k=1):
        calls.append(name)
        return [(f"{name}补位", {"file_name": name, "chunk_index": 99})]

    hits = [_hit("甲.md", 0), _hit("甲.md", 1), _hit("乙.md", 0)]
    out = llm.ensure_file_coverage("列出所有的实验", None, FILES, hits, 4, backfill=backfill)

    assert calls == ["丙.md"], "只补缺的那一份"
    assert out[0][1]["file_name"] == "丙.md", "补位块必须排最前，否则装配时先被字符上限截掉"
    assert len(out) == 4, "块预算不变"


def test_backfill_is_capped(switch_on, monkeypatch):
    monkeypatch.setattr(llm, "COVERAGE_BACKFILL_MAX", 1)
    calls = []
    files = [(f"文件{i}.md", "正文") for i in range(5)]

    def backfill(_collection, _question, name, top_k=1):
        calls.append(name)
        return [(f"{name}补位", {"file_name": name, "chunk_index": 0})]

    out = llm.ensure_file_coverage("列出所有的实验", None, files, [_hit("文件0.md", 0)], 3,
                                   backfill=backfill)
    assert len(calls) == 1, "一次提问不能变成 n 次检索"
    assert len(out) <= 3


def test_backfill_that_returns_an_existing_chunk_does_not_duplicate(switch_on):
    existing = _hit("甲", 0)

    def backfill(_collection, _question, name, top_k=1):
        return [existing]

    out = llm.ensure_file_coverage("列出所有的实验", None, FILES, [existing], 5,
                                   backfill=backfill)
    keys = [(text, meta["file_name"], meta["chunk_index"]) for text, meta in out]
    assert len(keys) == len(set(keys)), "同一个块不许出现两次（白占预算）"


def test_decompose_branch_does_not_also_sample(monkeypatch):
    """拆问已经按子问分路分预算，两套取样叠在一起会把预算切碎两次。"""
    monkeypatch.setattr(llm, "COVERAGE_SAMPLING", True)
    calls = []
    monkeypatch.setattr(llm, "ensure_file_coverage",
                        lambda *a, **k: calls.append("sampled") or a[3])

    hits = llm._retrieve_hits("列出所有的实验，还有主要内容", None, "", False, 5,
                              [("甲.md", "甲正文提到实验与主要内容")], 
                              lambda question, _history=None: ["实验有哪些", "主要内容是什么"],
                              history=None)
    assert isinstance(hits, list)
    assert calls == [], "拆问这条路不该再套跨文件取样"
