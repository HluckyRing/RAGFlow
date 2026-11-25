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
import src.retrieval as retrieval
from scripts.eval_retrieval import normalize
from src.config import coverage_switches

from src.coverage import (
    ENUMERATION_MARKERS,
    TOPIC_LIMIT,
    blocks_that_fit,
    chapter_numbers,
    intra_file_notice,
    intra_file_report,
    page_title,
    pick_target_file,
    sample_evenly,
    section_order,
    structural_texts,
    topic_gaps,
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
    monkeypatch.setattr(llm, "COVERAGE_CROSS_FILE", True)
    monkeypatch.setattr(llm, "COVERAGE_INTRA_FILE", True)
    monkeypatch.setattr(llm, "COVERAGE_BACKFILL_MAX", 8)
    return True


def test_switch_off_returns_hits_untouched(monkeypatch):
    monkeypatch.setattr(llm, "COVERAGE_CROSS_FILE", False)
    monkeypatch.setattr(llm, "COVERAGE_INTRA_FILE", False)
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
    monkeypatch.setattr(llm, "COVERAGE_CROSS_FILE", True)
    monkeypatch.setattr(llm, "COVERAGE_INTRA_FILE", True)
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

# ── 语言无关的结构页检测（2026-11-12 第十七轮）──
#
# 措辞表换语言就失效，但结构页有一条**语言无关**的特征：同一张目录/知识导图会在每份文件里
# 几乎逐字重复。这里钉住：跨 ≥3 份文件的近重复才算结构页；同文件内重复、只跨 2 份文件、
# 以及唯一的正文块都不许误伤 —— 而且要能在**完全没有中文措辞**的语料上生效。

EN_TOC = ("Table of contents Chapter 1 Introduction Chapter 2 Processes "
          "Chapter 3 Scheduling Chapter 4 Synchronization Chapter 5 Memory")
EN_BODY = "The scheduler picks the next process from the ready queue using its priority."


def test_structural_texts_flags_near_duplicates_across_three_files():
    hits = [(EN_TOC + " v1", {"file_name": "a.md"}),
            (EN_TOC + " v2", {"file_name": "b.md"}),
            (EN_TOC + " v3", {"file_name": "c.md"}),
            (EN_BODY, {"file_name": "d.md"})]
    flagged = structural_texts(hits)
    assert len(flagged) == 3, "三份文件里的目录页都要被抓出来"
    assert normalize(EN_BODY) not in flagged


def test_structural_texts_needs_at_least_three_files():
    """只跨两份文件的重复可能是正常引用（同一句话被两处引用），不能当结构页。"""
    hits = [(EN_TOC, {"file_name": "a.md"}), (EN_TOC, {"file_name": "b.md"}),
            (EN_BODY, {"file_name": "c.md"})]
    assert structural_texts(hits) == set()


def test_structural_texts_ignores_repeats_within_one_file():
    """同一份文件里的页眉页脚会重复，那不是结构页。"""
    hits = [(EN_TOC, {"file_name": "a.md"}), (EN_TOC, {"file_name": "a.md"}),
            (EN_TOC, {"file_name": "b.md"}), (EN_BODY, {"file_name": "c.md"})]
    assert structural_texts(hits) == set()


def test_is_navigational_prefers_the_language_independent_signal():
    """英文目录页：中文措辞表一个字都命中不了，靠跨文件近重复仍要判成结构页。"""
    hits = [(EN_TOC + " v1", {"file_name": "a.md"}),
            (EN_TOC + " v2", {"file_name": "b.md"}),
            (EN_TOC + " v3", {"file_name": "c.md"})]
    flagged = structural_texts(hits)
    assert is_navigational(EN_TOC + " v1", flagged) is True
    assert is_navigational(EN_TOC + " v1") is False, "没有结构信号时措辞表认不出来（这正是要被替代的）"


def test_prefer_content_puts_english_structural_pages_last():
    hits = [(EN_TOC + " v1", {"file_name": "a.md"}),
            (EN_TOC + " v2", {"file_name": "b.md"}),
            (EN_TOC + " v3", {"file_name": "c.md"}),
            (EN_BODY, {"file_name": "d.md"})]
    ordered = prefer_content(hits)
    assert ordered[0][0] == EN_BODY, "正文必须排到结构页前面"
    assert [meta["file_name"] for _t, meta in ordered[-3:]] == ["a.md", "b.md", "c.md"]


# ── A+B：整章枚举 + 块数按字符预算反推（2026-11-15）──
#
# 机器学习三份 PDF（96–110 页/章）上量到的病：取样池是「最相似的 40 个候选」，30 块只铺开
# 约 27% 的页，章尾整块落空（第1章 sklearn 94–96 页、第2章 决定系数 88–101 页）。
# 这里钉住两件事：① 文件内取样走**整章枚举**（collection.get，不按相似度挑）；
# ② 块数由**字符预算**反推，硬上限不再是 30。

class _StubCollection:
    """只实现 `get(where=...)` 的假 collection：真实现走 file_blocks，不许偷偷 query()。"""

    def __init__(self, blocks):
        self._blocks = blocks
        self.get_calls = []

    def get(self, where=None, limit=None):
        name = (where or {}).get("file_name")
        picked = [(t, m) for t, m in self._blocks if (m or {}).get("file_name") == name]
        self.get_calls.append((name, limit))
        docs = [t for t, _m in picked]
        metas = [m for _t, m in picked]
        if limit:
            docs, metas = docs[:limit], metas[:limit]
        return {"documents": docs, "metadatas": metas}


def _chapter_pages(name, pages, filler=60):
    """造一份「一页一块」的课件：每块 ~124 字（120 字正文 + 页码短前缀），页码连续。"""
    return [(f"第{i}页 " + "正文" * filler, {"file_name": name, "section": f"第 {i} 页",
                                            "chunk_index": i - 1}) for i in range(1, pages + 1)]


def test_blocks_that_fit_scales_with_the_char_budget_not_a_fixed_cap():
    """块数按字符预算反推：一页 124 字 + 8000 字预算 ≈ 48 块，而不是写死的 30。"""
    pages = _chapter_pages("02-第2章_甲.md", 200)
    fitted = blocks_that_fit(pages, 200, 8000)
    assert fitted > 30, "100 页以上的章，30 块铺不开（这正是 A 之前的问题）"
    assert fitted <= 50, f"不许超预算（每块 124 字正文 + 来源标签 40）: {fitted}"


def test_file_blocks_enumerates_whole_file_in_page_order():
    """整章枚举：一次 get 拿全，按页序返回，cap 生效。"""
    collection = _StubCollection(_chapter_pages("02-第2章_甲.md", 60)
                                 + _chapter_pages("03-第3章_乙.md", 10))
    got = retrieval.file_blocks(collection, "02-第2章_甲.md", cap=400)
    assert len(got) == 60, "要拿到整份文件的块，不是最相似的几个"
    assert [m["section"] for _t, m in got][:2] == ["第 1 页", "第 2 页"], "按页序排好"
    assert len(retrieval.file_blocks(collection, "02-第2章_甲.md", cap=5)) == 5
    assert retrieval.file_blocks(None, "02-第2章_甲.md") == []


def test_intra_file_sampling_enumerates_the_whole_chapter(monkeypatch):
    """点名章号时：整章枚举 + 按预算取块 = 首尾都要覆盖到（章尾不许整段落空）。"""
    monkeypatch.setattr(llm, "COVERAGE_CROSS_FILE", False)
    monkeypatch.setattr(llm, "COVERAGE_INTRA_FILE", True)
    pages = _chapter_pages("02-第2章_甲.md", 100)
    collection = _StubCollection(pages)
    out = llm.ensure_file_coverage("帮我整理第2章的内容", collection,
                                   [("02-第2章_甲.md", "正文")], [], 10)
    page_nums = [int(m["section"].replace("第", "").replace("页", "")) for _t, m in out]
    assert collection.get_calls, "必须走整章枚举（collection.get），不是相似度查询"
    assert len(out) > 30, "预算允许就要多取（硬上限 200）"
    assert min(page_nums) <= 5 and max(page_nums) >= 90, "铺满整章：首尾都要有"


# ── 覆盖说明：只摘了 N/M 段要说出来（2026-11-15）──
#
# 8000 字上下文装不下 100 页课件（整章 10 541–15 616 字符），取样必然只覆盖一部分。
# 不说清楚，模型会假装整理完了、用户会以为拿到了整章。所以账要两头走：
# ① 插一句【覆盖说明】进 context（模型据此如实交代）；② 通过 SSE 的 coverage 事件告诉前端。

def test_intra_file_report_counts_segments():
    assert intra_file_report([("a", {}), ("b", {})], 10, "第2章.pdf") == {
        "kind": "intra_file", "file_name": "第2章.pdf", "covered": 2, "total": 10}
    # 取到的块数不许比「整份文件一共几块」还多（口径自洽，避免前端显示 12/10）
    assert intra_file_report([("a", {})] * 3, 1, "x.pdf")["total"] == 3
    assert intra_file_report(None, 0, "") == {"kind": "intra_file", "file_name": "",
                                              "covered": 0, "total": 0}


def test_intra_file_notice_only_when_partial():
    full = {"kind": "intra_file", "file_name": "第2章.pdf", "covered": 10, "total": 10}
    assert intra_file_notice(full) == "", "全覆盖时不许吓唬模型"
    assert intra_file_notice(None) == ""
    partial = {"kind": "intra_file", "file_name": "第2章.pdf", "covered": 43, "total": 110}
    notice = intra_file_notice(partial)
    assert "43/110" in notice and "第2章.pdf" in notice
    assert "不要假装看过全部内容" in notice


def test_intra_file_sampling_fills_the_coverage_report(monkeypatch):
    """取样要顺手把账填进 `coverage_out`（server 靠它推 SSE、llm 靠它插说明）。"""
    monkeypatch.setattr(llm, "COVERAGE_CROSS_FILE", False)
    monkeypatch.setattr(llm, "COVERAGE_INTRA_FILE", True)
    collection = _StubCollection(_chapter_pages("02-第2章_甲.md", 100))
    report = {}
    picked = llm.ensure_file_coverage("帮我整理第2章的内容", collection,
                                      [("02-第2章_甲.md", "正文")], [], 10,
                                      coverage_out=report)
    assert report["kind"] == "intra_file"
    assert report["file_name"] == "02-第2章_甲.md"
    assert report["covered"] == len(picked) and report["total"] == 100
    assert report["covered"] < report["total"], "100 页装不进 8000 字，本该是部分覆盖"


def test_context_gets_the_coverage_notice(monkeypatch):
    """【覆盖说明】必须真的进 context —— 这是模型「如实交代只覆盖了一部分」的唯一依据。"""
    monkeypatch.setattr(llm, "_retrieve_hits",
                        lambda *a, **k: [("正文内容", {"file_name": "第2章.pdf",
                                                       "section": "第 1 页"})])
    report = {"kind": "intra_file", "file_name": "第2章.pdf", "covered": 43, "total": 110}
    context, used = llm.retrieve_and_build_context(
        "帮我整理第2章的内容", "正文内容", None, False, 5,
        [("第2章.pdf", "正文内容")], None, None, report)
    assert context.startswith("【覆盖说明】"), "覆盖说明要排在最前面，别被块内容淹掉"
    assert "43/110" in context
    assert used and used[0]["file_name"] == "第2章.pdf", "来源里不许混进这句说明"
    # 全覆盖时不许插
    full = {"kind": "intra_file", "file_name": "第2章.pdf", "covered": 110, "total": 110}
    context2, _ = llm.retrieve_and_build_context(
        "帮我整理第2章的内容", "正文内容", None, False, 5,
        [("第2章.pdf", "正文内容")], None, None, full)
    assert not context2.startswith("【覆盖说明】")



# ── 两个开关分开（2026-11-14 第十八轮）──
#
# 之前一个 `COVERAGE_SAMPLING` 管着两套行为：跨文件取样（线上预算下实测 no-op）与同文件跨节
# 取样（唯一有正面结果的一条）。共用一个开关就没法分别评估，也没法只开有效的那一条。
# 这里钉住：两个开关各自独立生效、互不误触发、老开关按「两个都开」兼容并标记出来（好打 WARNING）。

def test_coverage_switch_parsing_is_independent():
    """默认值：跨文件 0（线上预算下 no-op）、文件内 1（2026-11-15 过质量门）。"""
    assert coverage_switches({}) == (False, True, False)
    assert coverage_switches({"COVERAGE_CROSS_FILE": "1"}) == (True, True, False)
    assert coverage_switches({"COVERAGE_INTRA_FILE": "0"}) == (False, False, False)
    assert coverage_switches({"COVERAGE_CROSS_FILE": "1", "COVERAGE_INTRA_FILE": "0"}) == \
        (True, False, False)
    assert coverage_switches({"COVERAGE_CROSS_FILE": "false"}) == (False, True, False)


def test_legacy_coverage_sampling_overrides_both_defaults():
    """老开关显式写了就以它为准（= 两个都按它的值）：第十三~十六轮的评测脚本都按这个名字跑。"""
    assert coverage_switches({"COVERAGE_SAMPLING": "1"}) == (True, True, True)
    assert coverage_switches({"COVERAGE_SAMPLING": "0"}) == (False, False, True), \
        "COVERAGE_SAMPLING=0 仍然是「两个都关」，不许被新默认值偷偷打开"
    assert coverage_switches({"COVERAGE_SAMPLING": "0", "COVERAGE_CROSS_FILE": "1"}) == \
        (False, False, True)


def test_cross_file_switch_alone_falls_back_to_the_cross_file_path(monkeypatch):
    """关了同文件跨节 → 点名章号的提问退回第十三~十五轮那条路（每份文件补位 3 候选）。"""
    monkeypatch.setattr(llm, "COVERAGE_CROSS_FILE", True)
    monkeypatch.setattr(llm, "COVERAGE_INTRA_FILE", False)
    asked = []

    def backfill(_collection, _question, name, top_k=1):
        asked.append((name, top_k))
        return [(f"{name}补位", {"file_name": name, "chunk_index": 99})]

    files = [("01-第1章_甲.md", "正文"), ("02-第2章_乙.md", "正文")]
    hits = [("第1章正文", {"file_name": "01-第1章_甲.md", "section": "第 1 页", "chunk_index": 1})]
    out = llm.ensure_file_coverage("帮我整理第一章的内容", None, files, hits, 10, backfill=backfill)

    assert asked == [("02-第2章_乙.md", llm.COVERAGE_PER_FILE_CANDIDATES)], \
        "走的是跨文件补位（每份 3 个候选），不是文件内跨节（40 个候选）"
    assert {meta["file_name"] for _t, meta in out} == {"01-第1章_甲.md", "02-第2章_乙.md"}


def test_intra_file_switch_alone_leaves_cross_file_questions_untouched(monkeypatch):
    monkeypatch.setattr(llm, "COVERAGE_CROSS_FILE", False)
    monkeypatch.setattr(llm, "COVERAGE_INTRA_FILE", True)
    calls = []
    hits = [_hit("甲.md", 0), _hit("乙.md", 0)]
    out = llm.ensure_file_coverage("列出所有的实验", None, FILES, hits, 5,
                                   backfill=lambda *a, **k: calls.append(a) or [])
    assert out is hits, "没点名章号 + 跨文件开关关着 → 原样返回（不许顺手做另一套取样）"
    assert calls == []


def test_intra_file_sampling_backfills_when_the_target_file_has_no_hits(monkeypatch):
    """整句检索走关键词兜底时，命中块**没有 metadatas** —— 目标文件一块都认不出来也必须补位。

    2026-11-14 扩 gold 时踩到的真 bug：`blocks_that_fit([])` 是 0，`len(inside) < blocks`
    （0 < 0）不成立，于是直接返回空列表 → 模型拿到空 context → 空回答（q-ch2 就是这样）。
    """
    monkeypatch.setattr(llm, "COVERAGE_CROSS_FILE", False)
    monkeypatch.setattr(llm, "COVERAGE_INTRA_FILE", True)
    keyword_hits = [("关键词兜底块一", None), ("关键词兜底块二", None)]
    body = [(f"第2章正文{i}", {"file_name": "02-第2章_乙.md", "section": f"第 {i} 页",
                              "chunk_index": i}) for i in range(1, 21)]
    asked = {}

    def backfill(_collection, _question, name, top_k=1):
        asked["name"], asked["top_k"] = name, top_k
        return body

    out = llm.ensure_file_coverage("帮我整理第2章的内容", None,
                                   [("02-第2章_乙.md", "正文")], keyword_hits, 39,
                                   backfill=backfill)
    assert asked["name"] == "02-第2章_乙.md"
    assert asked["top_k"] == llm.COVERAGE_INTRA_FILE_CANDIDATES
    assert out, "目标文件没有命中时要靠文件内补位，不能返回空 context"
    assert {meta["file_name"] for _t, meta in out} == {"02-第2章_乙.md"}


def test_intra_file_sampling_works_on_a_single_file_corpus(monkeypatch):
    """只有一份文件时也能「整理第一章」：跨文件那套才需要 ≥2 份文件。"""
    monkeypatch.setattr(llm, "COVERAGE_CROSS_FILE", False)
    monkeypatch.setattr(llm, "COVERAGE_INTRA_FILE", True)
    hits = [(f"第1章正文{i}", {"file_name": "01-第1章_甲.md", "section": f"第 {i} 页",
                              "chunk_index": i}) for i in range(1, 21)]
    out = llm.ensure_file_coverage("帮我整理第一章的内容", None,
                                   [("01-第1章_甲.md", "正文")], hits, 39,
                                   backfill=lambda *a, **k: [])
    assert out, "单文件语料也要能铺开整章"
    assert {meta["file_name"] for _t, meta in out} == {"01-第1章_甲.md"}
    assert len(out) >= 10, "铺开整章，不是只抓头几块"

# ── 缺口清单：缺的是哪几段、那几段讲什么（2026-11-20 第二十轮）──
#
# 待办 ①「缺的是哪一节」：光有总账（N/M 段）用户不知道少了什么。负结果先记着——按「小节编号」
# 归属这条路原型实测是坏的（大纲页一次性列全章小节、正文页首行与小节标题不互为子串、ML 的 PDF
# 没有编号大纲还把正文算式当标题），所以这里只报**能验证的东西**：没进 context 的块，
# 它们的页标题是什么。判据是「这个标题在 picked 块里一次都没出现过」，不需要知道小节边界。

def _page(name, page, title, body="正文"):
    return (f"{title}\n{body}", {"file_name": name, "section": f"第 {page} 页",
                                 "chunk_index": page})


def test_page_title_strips_page_prefix_and_noise():
    assert page_title("67 / 岭回归\n正文") == "岭回归"
    assert page_title("⚫ 采用数值近似\n正文") == "采用数值近似"
    assert page_title("Date\n1.1 甲") == "1.1 甲", "噪音行要跳过，取下一个像标题的行"
    # 结构页（目录 / 知识导图）后面的行是全章清单，当标题会张冠李戴 —— 就地判死
    assert page_title("Date\n内容导航：\n1.1 甲") == ""
    assert page_title("第5章知识导图\n第1章 | 甲") == ""
    assert page_title("主讲：王红玲    主审：汤小丹\n正文") == "正文", "版权行跳过"
    # 正文行（> 24 字）绝不许截断后冒充标题 —— 这是「不编名字」的底线
    assert page_title("连续分配方式：为一个用户程序分配一个连续的内存空间。\n正文") == ""
    # 只扫前 3 行：第 4 行才出现的「标题」不算
    assert page_title("正文\n正文\n正文\n看起来像标题") == "正文"
    assert page_title("") == "" and page_title(None) == ""


def test_topic_gaps_names_what_was_never_read():
    pool = [_page("甲.pdf", 1, "概述"), _page("甲.pdf", 2, "线性回归模型"),
            _page("甲.pdf", 3, "岭回归"), _page("甲.pdf", 4, "Lasso 回归")]
    picked = [pool[0], pool[2]]
    gaps = topic_gaps(pool, picked)
    assert gaps["topics_missing"] == ["线性回归模型", "Lasso 回归"]
    assert gaps["missing_count"] == 2
    assert gaps["topics_total"] == 4


def test_topic_gaps_counts_a_title_once_even_if_it_spans_pages():
    """同一标题连排好几页时，读了其中 1 页就算这个标题摸到了 —— 报成「缺」会夸大缺口。"""
    pool = [_page("甲.pdf", 1, "岭回归"), _page("甲.pdf", 2, "岭回归"),
            _page("甲.pdf", 3, "Lasso 回归")]
    gaps = topic_gaps(pool, [pool[0]])
    assert gaps["topics_missing"] == ["Lasso 回归"], "岭回归读到了 1 页，不许再说缺"
    assert gaps["topics_total"] == 2


def test_topic_gaps_precision_every_named_title_is_really_absent():
    """判据 1（精确率 100%）：点名的每个标题，在 picked 块里一次都不许出现过。"""
    # 池子里标题有重复（同一标题连排好几页）：这时「读过」的标题绝不许再被点名
    pool = [_page("甲.pdf", i, f"标题{i % 5}") for i in range(1, 21)]
    picked = pool[:3]
    gaps = topic_gaps(pool, picked)
    read_titles = {page_title(text) for text, _meta in picked}
    assert gaps["topics_missing"], "构造里本来就该有缺口"
    assert not (set(gaps["topics_missing"]) & read_titles), "读过的标题不许再点名"


def test_topic_gaps_caps_the_list_but_reports_the_count():
    """上限不再是拍出来的「4 条」：`TOPIC_LIMIT=12` 是可读性上限，不是 4% 那种拍脑袋的数。"""
    pool = [_page("甲.pdf", i, f"标题{i}") for i in range(1, 41)]
    gaps = topic_gaps(pool, [])
    assert len(gaps["topics_missing"]) == TOPIC_LIMIT
    assert gaps["missing_count"] == 40


def test_topic_gaps_reports_more_than_the_old_four():
    """旧口径只列 4 条（真实语料里那是缺口的 4%），现在列满上限。"""
    pool = [_page("甲.pdf", i, f"小节{i}") for i in range(1, 31)]
    gaps = topic_gaps(pool, [])
    assert len(gaps["topics_missing"]) > 4


def test_topic_gaps_cost_has_a_provable_upper_bound():
    """清单字数上界 = TOPIC_LIMIT × (单个标题上限 + 1)：这是「上限」这个数唯一可证的依据。

    曾在 `TOPIC_LIMIT` 之外再加一层「字符预算 300 字」，但 12 × (18+1) = 228 < 300 ——
    那一层**永远不生效**（变异检查 E 实测：把预算逻辑关掉，测试全绿）。所以删掉它，
    改成把上界本身钉在这里：谁将来把 `_TITLE_MAX` 或 `TOPIC_LIMIT` 调大，这条会红。
    """
    from src.coverage import _TITLE_MAX

    pool = [_page("甲.pdf", i, f"{i:02d}超长页标题测试字数上界填充填充") for i in range(1, 41)]
    gaps = topic_gaps(pool, [])
    assert len(gaps["topics_missing"]) == TOPIC_LIMIT
    cost = sum(len(title) + 1 for title in gaps["topics_missing"])
    assert cost <= TOPIC_LIMIT * (_TITLE_MAX + 1), "清单字数不许越过可证明的上界"
    assert cost < 8000 * 0.03, "清单最多占 context 预算（8000 字）的 3%"


def test_topic_gaps_spreads_across_the_file_instead_of_only_the_head():
    """旧口径按文件顺序取前 N 条，122 块的文件永远只报开头那几页 —— 章末整段从不出现。"""
    pool = [_page("甲.pdf", i, f"小节{i}") for i in range(1, 101)]
    gaps = topic_gaps(pool, [])
    missing = gaps["topics_missing"]
    assert missing[0] == "小节1", "首条必含"
    assert missing[-1] == "小节100", "末条必含：尾部缺口也必须被点名"


def test_topic_gaps_skips_admin_pages_that_differ_only_in_spacing():
    """真实语料里行政页是「学 习 进 步 ！」（逐字加空格），子串噪音表一个都没命中。"""
    pool = [_page("甲.pdf", 1, "概述"),
            _page("甲.pdf", 2, "学 习 进 步 ！"),
            _page("甲.pdf", 3, "课程思政小讲堂"),
            _page("甲.pdf", 4, "第二次作业"),
            _page("甲.pdf", 5, "作业调度算法")]
    gaps = topic_gaps(pool, [pool[0]])
    assert "作业调度算法" in gaps["topics_missing"], "真内容不许被行政词误杀"
    for admin in ("学 习 进 步 ！", "课程思政小讲堂", "第二次作业"):
        assert admin not in gaps["topics_missing"]
    assert gaps["missing_count"] == 1


def test_topic_gaps_skips_structural_pages_and_does_not_name_them_by_page():
    """知识导图 / 目录页不是内容：既不点名，也不该被「第 N 页」兜底捡回来。"""
    pool = [_page("甲.pdf", 1, "概述"), _page("甲.pdf", 2, "第5章知识导图"),
            _page("甲.pdf", 3, "内容导航：")]
    gaps = topic_gaps(pool, [pool[0]])
    assert gaps["topics_missing"] == []
    assert gaps["missing_count"] == 0


def test_topic_gaps_names_untitled_pages_by_page_number():
    """页首就是正文长句的页（真实语料占 34.6%）不许再静默消失：用页码诚实命名。"""
    body = "连续分配方式：为一个用户程序分配一个连续的内存空间。\n正文"
    pool = [_page("甲.pdf", 1, "概述"), (body, {"file_name": "甲.pdf",
                                               "section": "第 7 页", "chunk_index": 7})]
    gaps = topic_gaps(pool, [pool[0]])
    assert gaps["topics_missing"] == ["第 7 页"]
    assert gaps["missing_count"] == 1


def test_topic_gaps_lists_titles_before_page_numbers():
    """页码条目排内容标题之后（稳定分区）：否则 PPT 的封面/教材页会把清单头部占满。"""
    body = "连续分配方式：为一个用户程序分配一个连续的内存空间。\n正文"
    pool = [(body, {"file_name": "甲.pdf", "section": "第 2 页", "chunk_index": 2}),
            _page("甲.pdf", 3, "岭回归"),
            _page("甲.pdf", 4, "Lasso 回归")]
    gaps = topic_gaps(pool, [])
    assert gaps["topics_missing"][:2] == ["岭回归", "Lasso 回归"]
    assert gaps["topics_missing"][-1] == "第 2 页"


def test_topic_gaps_returns_nothing_when_titles_cannot_be_parsed():
    """判据 2（不编名字）：一条标题都解析不出来、也没有页码时返回空 dict，调用方据此退回老话术。"""
    pool = [("", {"file_name": "甲.pdf", "chunk_index": 1}),
            ("主讲：王红玲", {"file_name": "甲.pdf", "chunk_index": 2})]
    assert topic_gaps(pool, []) == {}


def test_intra_file_notice_falls_back_to_the_old_wording_without_titles():
    """判据 2：没有缺口清单时，文本必须与 2026-11-15 那版**逐字一致**。"""
    report = intra_file_report([("a", {}), ("b", {})], 10, "第2章.pdf")
    assert "topics_missing" not in report
    assert intra_file_notice(report) == (
        "【覆盖说明】本次只从《第2章.pdf》里摘录了 2/10 段（按小节均匀取样），"
        "未被摘录的小节可能正是你要的细节。回答时请如实说明「本次只覆盖了这份文件的部分小节」，"
        "不要假装看过全部内容。")


def test_intra_file_notice_names_the_missing_titles():
    pool = [_page("甲.pdf", 1, "概述"), _page("甲.pdf", 2, "岭回归")]
    report = intra_file_report([pool[0]], 2, "甲.pdf", pool=pool)
    notice = intra_file_notice(report)
    assert "岭回归" in notice and "概述" not in notice
    assert "1/2" in notice
    assert "不要假装看过全部内容" in notice


def test_intra_file_report_keeps_the_old_contract_without_a_pool():
    assert intra_file_report([("a", {})], 10, "x.pdf") == {
        "kind": "intra_file", "file_name": "x.pdf", "covered": 1, "total": 10}


def test_intra_file_sampling_fills_the_gap_list(monkeypatch):
    """接线：`ensure_file_coverage` 要把缺口清单填进 `coverage_out`（SSE 与提示都靠它）。"""
    monkeypatch.setattr(llm, "COVERAGE_CROSS_FILE", False)
    monkeypatch.setattr(llm, "COVERAGE_INTRA_FILE", True)
    pages = [(f"小节{i}标题\n正文{i} " + "正文" * 60,
              {"file_name": "02-第2章_甲.md", "section": f"第 {i} 页", "chunk_index": i})
             for i in range(1, 101)]
    collection = _StubCollection(pages)
    report = {}
    picked = llm.ensure_file_coverage("帮我整理第2章的内容", collection,
                                      [("02-第2章_甲.md", "正文")], [], 10,
                                      coverage_out=report)
    read_titles = {page_title(text) for text, _meta in picked}
    assert report["covered"] < report["total"]
    assert report["missing_count"] > 0
    assert report["topics_missing"], "缺口清单要真的点名，不能只有一个数字"
    assert not (set(report["topics_missing"]) & read_titles), "点名的标题必须真的没读到"
