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
    blocks_that_fit,
    chapter_numbers,
    pick_target_file,
    sample_evenly,
    section_order,
    coverage_report,
    file_names_of,
    is_navigational,
    looks_enumerative,
    missing_files,
    prefer_content,
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

# ── 导航页 / 行政页（2026-11-12 第十四轮的真病根）──
#
# 实测：13 文件语料问「该知识库的主要内容是什么」，头部 14 块里 10 块是「第N章知识导图」，
# 另外还有「内容导航」目录页与「第一次作业 1 2 3 …」编号页。模型拿到这些只能复述章节名，
# 于是「覆盖变全」并不等于「答得更全」。这里钉住三件事：
# 1. 三类结构页都要能认出来，而正文页不许误伤；
# 2. 只**降权不删除** —— 小语料里那页可能就是唯一的答案（「有哪些题型」的清单就长在提纲页上）；
# 3. 稳定分区：内容页之间的相对顺序不许变。

@pytest.mark.parametrize("text", [
    "第1章知识导图 第1章 | 操作系统引论 第2章 | 进程的描述与控制 第3章 | 处理机调度 "
    "第4章 | 进程同步 第5章 | 存储器管理 第6章 | 虚拟存储器",
    "内容导航： 1.1 操作系统的目标和作用 1.2 操作系统的发展过程",
    "第一次作业 1 简答题 2 3 4 5 标黄色为本次作业",
    "学而时习之（第6章总结）",
    "课程思政小讲堂 积极推进操作系统产品化",
])
def test_structural_pages_are_recognised(text):
    assert is_navigational(text) is True


@pytest.mark.parametrize("text", [
    "操作系统的目标 方便性 通过OS命令操纵计算机，方便用户 有效性 提高系统资源的利用率",
    "第2章 进程的描述与控制 2.1 前趋图和程序执行 进程是资源分配的基本单位",
    "最佳置换算法OPT 被置换的页将是之后最长时间不被使用的页",
    "",
])
def test_content_pages_are_not_flagged(text):
    assert is_navigational(text) is False


def test_prefer_content_puts_content_first_but_keeps_relative_order():
    nav1 = ("第1章知识导图 第1章 | …", {"file_name": "甲", "chunk_index": 1})
    body1 = ("正文一", {"file_name": "甲", "chunk_index": 2})
    nav2 = ("第一次作业 1 2 3 标黄色为本次作业", {"file_name": "乙", "chunk_index": 1})
    body2 = ("正文二", {"file_name": "乙", "chunk_index": 2})
    picked = prefer_content([nav1, body1, nav2, body2])
    assert [meta["chunk_index"] for _t, meta in picked] == [2, 2, 1, 1]
    assert [text for text, _m in picked] == ["正文一", "正文二", nav1[0], nav2[0]]


def test_prefer_content_never_drops_navigation_only_hits():
    """小语料里那页可能就是唯一答案 —— 只降权，不许删。"""
    nav = [("第1章知识导图 第1章 | …", {"file_name": "甲", "chunk_index": 1})]
    assert prefer_content(nav) == nav
    assert prefer_content([]) == []


def test_sampling_picks_a_content_block_over_a_navigation_page(switch_on):
    """每份文件的代表块不许是它的知识导图页（第十四轮实测的失败模式）。"""
    nav = ("第2章知识导图 第1章 | … 第2章 | …", {"file_name": "甲.md", "chunk_index": 1})
    body = ("进程是资源分配的基本单位，前趋图描述执行先后。",
            {"file_name": "甲.md", "chunk_index": 7})
    other = ("索引结点把文件名与描述信息分开。", {"file_name": "乙.md", "chunk_index": 2})
    out = llm.ensure_file_coverage(
        "该知识库的主要内容是什么", None,
        [("甲.md", "正文"), ("乙.md", "正文")],
        [nav, body, other], 5, backfill=lambda *a, **k: [])
    assert out[0][1]["chunk_index"] == 7, "内容页要排到导图页前面"


def test_backfill_picks_a_content_candidate_over_a_navigation_one(switch_on):
    """补位要取几个候选再挑正文页：一份文件里最像「内容是什么」的往往正是它的目录页。"""
    nav = ("内容导航： 1.1 … 1.2 …", {"file_name": "乙.md", "chunk_index": 1})
    body = ("索引结点把文件名与描述信息分开。", {"file_name": "乙.md", "chunk_index": 9})
    asked = {}

    def backfill(_collection, _question, name, top_k=1):
        asked["top_k"] = top_k
        return [nav, body]

    out = llm.ensure_file_coverage("列出所有的实验", None, FILES, [_hit("甲.md", 0)], 5,
                                   backfill=backfill)
    assert asked["top_k"] >= 3, "只取 1 个候选就没有挑正文页的余地"
    assert out[0][1]["chunk_index"] == 9

# ── 同一份文件内部跨节取样（2026-11-12 第十六轮）──
#
# 真实需求：「帮我整理第一章的内容」——要的是把那一章（64 页）铺开讲，不是跨文件代表性。
# 这里钉住四件事：章号解析、目标文件定位（歧义就要放弃）、按页均匀铺开、以及
# 「只在问句点明章号且是枚举/整理类时才走这条路」（否则会毁掉精确事实问答）。

@pytest.mark.parametrize("question,expected", [
    ("帮我整理第一章的内容", {1}),
    ("整理第2章", {2}),
    ("第十二章讲了什么", {12}),
    ("第 3 章 重点", {3}),
    ("总结这份材料", set()),
    ("", set()),
])
def test_chapter_numbers(question, expected):
    assert chapter_numbers(question) == expected


def test_pick_target_file_uses_the_chapter_number():
    names = ["01-第1章_操作系统引论.md", "02-第2章_进程的描述与控制.md"]
    assert pick_target_file("帮我整理第一章的内容", names) == "01-第1章_操作系统引论.md"
    assert pick_target_file("整理第2章", names) == "02-第2章_进程的描述与控制.md"


def test_pick_target_file_gives_up_on_no_match_or_ambiguity():
    """定位错了会答成另一章，比不取样更糟 —— 找不到或有歧义都必须放弃。"""
    names = ["第1章_甲.md", "第1章_乙.md"]
    assert pick_target_file("整理第1章", names) is None, "两个文件都自称第1章，不许瞎猜"
    assert pick_target_file("整理第9章", names) is None
    assert pick_target_file("总结一下", ["第1章_甲.md"]) is None, "没提章号就不定位"


def test_section_order_parses_page_numbers_and_falls_back_to_chunk_index():
    assert section_order({"section": "第 12 页", "chunk_index": 11}) == (0, 12, 11)
    assert section_order({"section": None, "chunk_index": 7}) == (1, 7, 0)


def test_sample_evenly_spreads_across_the_whole_file():
    """12 段取 4 块要铺满首尾，不是抓前 4 段。"""
    hits = [(f"正文{i}", {"section": f"第 {i} 页", "chunk_index": i}) for i in range(1, 13)]
    picked = sample_evenly(hits, 4)
    assert [meta["chunk_index"] for _t, meta in picked] == [1, 4, 7, 10]
    assert sample_evenly(hits, 20) == hits, "块数够就原样按节序返回"
    assert sample_evenly([], 5) == []


def test_intra_file_sampling_only_returns_blocks_of_the_target_file(switch_on):
    """点名第一章时，不许再把别的文件的块塞进来（那正是之前拒答的原因）。"""
    same_file = [("第1章正文 " + str(i), {"file_name": "01-第1章_甲.md", "section": f"第 {i} 页",
                                        "chunk_index": i}) for i in range(1, 21)]
    other = [("别的内容", {"file_name": "02-第2章_乙.md", "section": "第 1 页", "chunk_index": 0})]
    out = llm.ensure_file_coverage(
        "帮我整理第一章的内容", None,
        [("01-第1章_甲.md", "正文"), ("02-第2章_乙.md", "正文")],
        other + same_file, 39, backfill=lambda *a, **k: [])
    assert out, "必须挑出块来"
    assert {meta["file_name"] for _t, meta in out} == {"01-第1章_甲.md"}
    assert len(out) <= llm.COVERAGE_INTRA_FILE_BLOCKS, "块数不超过上限（上限内再按字符预算反推）"


def test_intra_file_sampling_does_not_trigger_without_a_chapter_mention(switch_on):
    """没点名章号就走原来的跨文件路 —— 精确事实问答不许被铺开取样毁掉。"""
    same_file = [("第1章正文 " + str(i), {"file_name": "01-第1章_甲.md", "section": f"第 {i} 页",
                                        "chunk_index": i}) for i in range(1, 6)]
    files = [("01-第1章_甲.md", "正文"), ("02-第2章_乙.md", "正文")]
    out = llm.ensure_file_coverage("列出所有的实验", None, files, same_file, 5,
                                   backfill=lambda *a, **k: [])
    assert len(out) == len(same_file)

def test_blocks_that_fit_adapts_to_chunk_length():
    """块数按字符预算反推：短页多取、长块少取，写死块数两头都会错。"""
    short = [("x" * 140, {}) for _ in range(60)]
    long_chunks = [("x" * 480, {}) for _ in range(60)]
    assert blocks_that_fit(short, 30, 8000) == 30, "PPT 短页应当取满上限"
    assert blocks_that_fit(long_chunks, 30, 8000) == 15, "480 字的长块只装得下 ~15 块"
    assert blocks_that_fit(short, 30, 1000) >= 1
    assert blocks_that_fit([], 30, 8000) == 0
