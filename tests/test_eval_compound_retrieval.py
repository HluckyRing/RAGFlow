# -*- coding: utf-8 -*-
"""复合问题检索覆盖实验的纯函数回归。

这个实验的 gold 是**现场派生**的（锚点 + 子问题 + 复合问句），指标全靠几个纯函数算出来；
它们一旦算错，脚本会一本正经地吐出一张错表。所以这里用假数据把每一环钉住：

- 锚点必须真的长在那片正文里、且不跨文件重复（否则「命中」两个字没有意义）；
- 复合样本的每个子问题必须来自**不同文件**（否则测到的还是单文件装不装得下）；
- 覆盖判定的边界（同一片 / 同文件同节 / 同文件不同节）—— 口径是我定的，必须有测试钉住；
- 单问基线没命中的子问题**不进**合并覆盖率的分母；
- 汇总在 2 问与 4 问、空输入下都不崩；
- 出错路径（状态目录不存在 / 没有带文件的对话）要返回非 0 且是中文提示。

全部用假数据：**不加载 embedding 模型、不碰 Chroma、不读真实 state**（那条路只能真跑脚本验证）。
"""
import json
import random

import pytest

from scripts.eval_compound_retrieval import (
    ROUTE_DECOMPOSE,
    ROUTE_INTERLEAVE,
    ROUTE_MERGED,
    ROUTE_RRF,
    anchor_candidates,
    anchor_fits,
    anchor_visible,
    baseline_hit,
    baseline_kind,
    build_compound_question,
    build_probes,
    build_samples,
    build_sub_question,
    covered,
    find_leaks,
    hit_match,
    hit_key,
    interleave_hits,
    make_sample,
    measure_baseline,
    pick_anchor,
    rrf_merge_hits,
    route_retrieve,
    run_sample,
    run_sample_arm,
    sample_files,
    summarize,
    summarize_routes,
    validate_sample,
)
from scripts.eval_retrieval import normalize


# ── 假语料（合成字符串，绝不是真实语料）──

FILE_A = ("岗前安全培训要求所有人员在进入作业区之前完成确认。"
          "带班负责人必须逐项核对防护用品的完好情况。"
          "出现异常时立即停止作业并上报值班室。")
FILE_B = ("月度成本分析需要把折旧与人工分开列示。"
          "折旧按年限平均法在预计使用年限内分摊。"
          "人工成本包含基本工资与绩效奖励两部分。")
REPEATED = "数据结构的基本概念" * 30


def _probe(file_name, chunk_index, anchor, section=None):
    """造一个探针：目标块的身份就是 (file_name, chunk_index, section)。"""
    return {
        "file": file_name,
        "anchor": anchor,
        "sub_question": build_sub_question(anchor),
        "target": {"file_name": file_name, "chunk_index": chunk_index, "section": section},
        "chunk": "正文不参与判定",
    }


def _part(part_id, covered_top, covered_extra, eligible=True,
          strict_top=None, strict_extra=None, baseline_top1=None, baseline_rank=None,
          raw_rank=None):
    return {
        "id": part_id,
        "eligible": eligible,
        "baseline_top1": eligible if baseline_top1 is None else baseline_top1,
        "baseline_rank": baseline_rank,
        "raw_rank": raw_rank,
        "file_ix": 0,
        "chunk_index": 3,
        "covered_top": covered_top,
        "covered_extra": covered_extra,
        "strict_top": covered_top if strict_top is None else strict_top,
        "strict_extra": covered_extra if strict_extra is None else strict_extra,
    }


def _record(path, parts, record_id="s"):
    return {"id": record_id, "n_parts": len(parts), "path": path, "path_extra": path,
            "hits_top": 5, "hits_extra": 10,
            "best_distance": None, "parts": parts}


# ── 锚点：必须长在那片正文里，且不跨文件重复 ──

def test_pick_anchor_lives_in_the_chunk_and_is_unique_in_its_file():
    anchor = pick_anchor(FILE_A, FILE_A, [FILE_A], random.Random(0))

    assert anchor, "一段正常长度的正文应该挑得出锚点"
    assert anchor in normalize(FILE_A), "锚点必须是这片正文里的连续片段，否则子问题定位不到它"
    assert normalize(FILE_A).count(anchor) == 1, "本文件里必须唯一"


def test_pick_anchor_gives_up_instead_of_returning_a_useless_phrase():
    """整篇都在重复同一句时，任何 8–14 字窗口都出现几十次 —— 宁可返回空，不许凑数。"""
    assert pick_anchor(REPEATED, REPEATED, [REPEATED], random.Random(0)) == ""


def test_pick_anchor_is_reproducible_for_the_same_seed():
    first = pick_anchor(FILE_A, FILE_A, [FILE_A, FILE_B], random.Random(11))
    second = pick_anchor(FILE_A, FILE_A, [FILE_A, FILE_B], random.Random(11))

    assert first == second, "同一 seed 必须挑出同一个锚点，否则两轮数字不可比"


def test_anchor_fits_rejects_a_phrase_that_repeats_in_the_same_file():
    assert anchor_fits("数据结构的基本概念", REPEATED, [REPEATED]) is False


def test_anchor_fits_rejects_a_phrase_that_is_too_common_across_files():
    """锚点在别的文件里也出现时，检索命中别处是合理的，那不算「这条子问题没被装下」。"""
    phrase = "带班负责人必须逐项核对"
    a = "前言说明。" + phrase + "。后记。"
    b = phrase + "。" + phrase + "。"

    assert anchor_fits(phrase, a, [a, b]) is False
    assert anchor_fits(phrase, a, [a]) is True
    assert anchor_fits(phrase, a, [a, b], max_corpus=3) is True, "上限可调，口径才可重标"


def test_anchor_candidates_never_cut_an_ascii_word_in_half():
    """锚点两端必须是汉字：把英文单词/编号切成半截的窗口包进「」里既不像话，也不好打分。"""
    text = "abc123 " + "缓存层负责热点数据" + " xyz789"

    candidates = anchor_candidates(text, random.Random(0))

    assert candidates, "正文里有正常中文片段，应该能给出候选"
    for window in candidates:
        assert window[0].isascii() is False and window[-1].isascii() is False
        assert window in normalize(text)


def test_anchor_candidates_are_shuffled_so_the_seed_decides():
    first = anchor_candidates(FILE_A, random.Random(3))
    second = anchor_candidates(FILE_A, random.Random(3))

    assert first == second
    assert set(first) == set(anchor_candidates(FILE_A, random.Random(4))), "候选集合与种子无关"


# ── 子问题 / 复合问句 ──

def test_build_sub_question_quotes_the_anchor_verbatim():
    question = build_sub_question("带班负责人必须逐项核对")

    assert "带班负责人必须逐项核对" in question, "子问题必须逐字含锚点，覆盖判定才可核验"
    assert question.startswith("材料里")


def test_build_compound_question_joins_all_parts_with_question_marks():
    compound = build_compound_question(["甲问句？", "乙问句", "丙问句"])

    assert compound.count("？") == 3
    assert compound.index("甲问句") < compound.index("乙问句") < compound.index("丙问句")
    assert build_compound_question([]) == ""


# ── 复合样本：每个子问题必须来自不同文件 ──

def test_build_samples_only_takes_one_probe_per_file():
    probes = [_probe("A", 0, "甲甲甲甲甲甲甲甲"), _probe("A", 1, "乙乙乙乙乙乙乙乙"),
              _probe("B", 0, "丙丙丙丙丙丙丙丙"), _probe("C", 0, "丁丁丁丁丁丁丁丁")]

    samples, skipped = build_samples(probes, size=2)

    assert skipped == 0
    assert len(samples) == 2
    for sample in samples:
        names = sample_files(sample)
        assert len(set(names)) == len(names), "一条样本里的子问题必须分散在不同文件"
        assert sample["n_parts"] == 2


def test_build_samples_stops_when_not_enough_files_for_four_parts():
    probes = [_probe("A", 0, "甲甲甲甲甲甲甲甲"), _probe("B", 0, "乙乙乙乙乙乙乙乙")]

    samples, _skipped = build_samples(probes, size=4)

    assert samples == [], "只有两个文件，造不出 4 问的分散样本"
    assert [s["n_parts"] for s in build_samples(probes, size=2)[0]] == [2]


def test_validate_sample_rejects_two_parts_from_the_same_file():
    sample = make_sample([_probe("A", 0, "甲甲甲甲甲甲甲甲"), _probe("A", 1, "乙乙乙乙乙乙乙乙")])

    ok, reason = validate_sample(sample)

    assert ok is False
    assert "同一文件" in reason


def test_build_samples_counts_samples_whose_targets_collide():
    """构造标签与目标块不一致的探针：必须是**跳过并计数**，绝不静默进分母。"""
    bad = [_probe("A", 0, "甲甲甲甲甲甲甲甲"), _probe("B", 0, "乙乙乙乙乙乙乙乙")]
    bad[1]["target"] = {"file_name": "A", "chunk_index": 9, "section": None}

    samples, skipped = build_samples(bad, size=2)

    assert samples == []
    assert skipped == 1, "「子问题挤在同一文件」的样本要单独计数报出来"


def test_build_probes_counts_chunks_without_anchor():
    files = [("A", FILE_A), ("B", FILE_B)]

    probes, missing = build_probes(files, random.Random(0), chunks_per_file=1,
                                  picker=lambda *a, **k: "")

    assert probes == []
    assert missing == 2, "挑不出锚点的片要如实计数，不能静默当成功"


# ── 覆盖判定的边界（口径见脚本 docstring）──

def test_hit_match_counts_the_same_chunk():
    target = {"file_name": "A", "chunk_index": 2, "section": None}

    assert hit_match({"file_name": "A", "chunk_index": 2}, target) == "same_chunk"


def test_hit_match_counts_a_sibling_chunk_on_the_same_section():
    """同文件同节的兄弟块算命中：线上引用粒度就是页，读者翻到那页能核验。"""
    target = {"file_name": "A", "chunk_index": 2, "section": "第 3 页"}

    assert hit_match({"file_name": "A", "chunk_index": 7, "section": "第 3 页"},
                     target) == "same_section"


def test_hit_match_rejects_same_file_different_section():
    """同文件但不同页不算：那只是「这份文件里检索到了别处」，模型手里没有那条事实。"""
    target = {"file_name": "A", "chunk_index": 2, "section": "第 3 页"}

    assert hit_match({"file_name": "A", "chunk_index": 7, "section": "第 4 页"}, target) is None


def test_hit_match_rejects_same_file_without_sections():
    """两侧都没有节（docx/md 这类没有页结构的格式）时只能认「同一片」。"""
    target = {"file_name": "A", "chunk_index": 2, "section": None}

    assert hit_match({"file_name": "A", "chunk_index": 3, "section": None}, target) is None


def test_hit_match_rejects_another_file_and_missing_meta():
    target = {"file_name": "A", "chunk_index": 2, "section": None}

    assert hit_match({"file_name": "B", "chunk_index": 2}, target) is None
    assert hit_match(None, target) is None
    assert hit_match({"file_name": "A", "chunk_index": 2}, None) is None


def test_covered_scans_every_hit_not_just_the_first():
    target = {"file_name": "A", "chunk_index": 2, "section": None}
    hits = [("无关正文", {"file_name": "B", "chunk_index": 0}),
            ("目标正文", {"file_name": "A", "chunk_index": 2})]

    assert covered(hits, target) is True
    assert covered(hits[:1], target) is False
    assert covered([], target) is False


def test_anchor_visible_ignores_whitespace_and_fullwidth_forms():
    anchor = "带班负责人必须逐项核对"
    hits = [("带班负责人必须 逐项核对", {"file_name": "A", "chunk_index": 0})]

    assert anchor_visible(hits, anchor) is True
    assert anchor_visible([("完全无关的一句话", {})], anchor) is False
    assert anchor_visible(hits, "") is False


def test_baseline_hit_needs_the_fact_not_just_the_file():
    """同页兄弟块里没有那条事实 —— 把它算成基线命中，会让「本来就检不到」溜进分母。"""
    target = {"file_name": "A", "chunk_index": 2, "section": "第 3 页"}
    sibling = [("同一页的另一段正文", {"file_name": "A", "chunk_index": 7, "section": "第 3 页"})]
    exact = [("带班负责人必须逐项核对防护用品", {"file_name": "A", "chunk_index": 2,
                                              "section": "第 3 页"})]

    assert baseline_hit(sibling, target, "带班负责人必须逐项核对") is False
    assert baseline_hit(exact, target, "带班负责人必须逐项核对") is True
    assert baseline_hit([], target, "带班负责人必须逐项核对") is False


def test_measure_baseline_marks_each_probe_and_survives_a_failure():
    good = _probe("A", 0, "甲甲甲甲甲甲甲甲")
    bad = _probe("B", 0, "乙乙乙乙乙乙乙乙")

    def retrieve(question, k):
        assert k == 1, "单问基线必须用 top-1 判"
        if question == bad["sub_question"]:
            raise RuntimeError("向量库挂了")
        return "vector", [("甲甲甲甲甲甲甲甲 讲的是什么", {"file_name": "A", "chunk_index": 0})]

    measure_baseline([good, bad], retrieve)

    assert good["baseline_top1"] is True
    assert bad["baseline_top1"] is False, "单条失败按未命中处理（保守排除出分母），不带走整轮"


# ── 单条样本的判定与落盘卫生 ──

def test_run_sample_judges_every_part_against_the_injected_retrieval():
    sample = make_sample([_probe("A", 0, "甲甲甲甲甲甲甲甲"),
                          _probe("B", 0, "乙乙乙乙乙乙乙乙")])
    for probe in sample["parts"]:
        probe["baseline_top1"] = True

    def retrieve(question, k):
        if k == 5:
            return "vector", [("甲甲甲甲甲甲甲甲在这里", {"file_name": "A", "chunk_index": 0})]
        return "vector", [("甲甲甲甲甲甲甲甲在这里", {"file_name": "A", "chunk_index": 0}),
                          ("乙乙乙乙乙乙乙乙在这里", {"file_name": "B", "chunk_index": 0})]

    record = run_sample(sample, retrieve, top_k=5, extra_k=10)

    assert record["path"] == "vector" and record["path_extra"] == "vector"
    assert [part["covered_top"] for part in record["parts"]] == [True, False]
    assert [part["covered_extra"] for part in record["parts"]] == [True, True]
    assert [part["strict_top"] for part in record["parts"]] == [True, False], \
        "严格口径要求锚点真的出现在正文里"
    assert record["hits_top"] == 1 and record["hits_extra"] == 2, \
        "命中块数要记下来 —— 闸门先砍到比 k 少时，调大 k 本来就不会多出块"


def test_summarize_reports_where_the_excluded_parts_were_ranked():
    """被排除的子问里有多少只是「排序没排上来」，而不是「根本没召回」。"""
    parts = [_part("p0", True, True),
             _part("p1", False, False, eligible=False, baseline_rank=3),
             _part("p2", False, False, eligible=False, baseline_rank=None)]
    stats = summarize([_record("vector", parts)], top_k=5, extra_k=10)["overall"]

    assert stats["excluded_ranked"] == 1
    assert stats["excluded_rank_median"] == 3
    assert stats["mean_hits_top"] == 5 and stats["mean_hits_extra"] == 10


def test_run_sample_detail_never_carries_corpus_text_or_file_names():
    sample = make_sample([_probe("文件名不许落盘.md", 3, "甲甲甲甲甲甲甲甲")])
    sample["parts"][0]["baseline_top1"] = True

    record = run_sample(sample, lambda q, k: ("vector", [("甲甲甲甲甲甲甲甲 正文", 
                                                          {"file_name": "文件名不许落盘.md",
                                                           "chunk_index": 3})]))

    detail = json.dumps(record, ensure_ascii=False)
    assert "甲甲甲甲甲甲甲甲" not in detail, "明细里只许有 id/布尔/距离/路径"
    assert "文件名不许落盘" not in detail
    assert find_leaks(detail, ["甲甲甲甲甲甲甲甲", "文件名不许落盘.md"]) == 0


def test_find_leaks_counts_without_echoing_the_content():
    assert find_leaks('{"a": 1}', ["甲甲甲甲甲甲甲甲"]) == 0
    assert find_leaks('{"a": "甲甲甲甲甲甲甲甲"}', ["甲甲甲甲甲甲甲甲", "x.md"]) == 1


def test_summarize_keeps_ineligible_parts_out_of_the_denominator():
    parts = [_part("p0", True, True), _part("p1", False, True, eligible=False)]
    stats = summarize([_record("vector", parts)], top_k=5, extra_k=10)["overall"]

    assert stats["parts"] == 2
    assert stats["parts_eligible"] == 1
    assert stats["parts_excluded"] == 1
    assert stats[f"coverage_at_5"] == 1.0, "分母只有基线命中的那条"
    assert stats[f"covered_at_5"] == 1


def test_summarize_computes_two_and_four_part_groups():
    two = [_record("vector", [_part("a0", True, True), _part("a1", False, True)], "s2"),
           _record("keyword", [_part("b0", True, True), _part("b1", False, True)], "s3")]
    four = [_record("vector", [_part(f"c{i}", i == 2, True) for i in range(4)], "s4")]

    stats = summarize(two + four, top_k=5, extra_k=10)
    overall = stats["overall"]
    by_size = {(group["n_parts"], group["path"]): group for group in stats["groups"]}

    assert overall["samples"] == 3
    assert overall["parts_eligible"] == 8
    assert overall[f"covered_at_5"] == 3, "2 问两条样本各 1 命中 + 4 问 1 命中"
    assert overall[f"covered_at_10"] == 8
    assert overall["gain"] == pytest.approx(5 / 8), "k 变大后每条样本都装下了剩下的子问题"

    assert by_size[(2, "all")]["samples"] == 2
    assert by_size[(2, "vector")]["samples"] == 1
    assert by_size[(2, "keyword")][f"covered_at_5"] == 1
    assert by_size[(4, "vector")]["samples"] == 1
    assert by_size[(4, "vector")][f"samples_all_covered_at_5"] == 0
    assert by_size[(4, "vector")][f"samples_all_covered_at_10"] == 1


def test_summarize_separates_rescuable_misses_from_out_of_reach_ones():
    """「调 k 能救」与「调 k 救不回」必须分开数，否则「边际收益 0」会被读成「k 够了」。"""
    parts = [_part("p0", False, False, raw_rank=7),
             _part("p1", False, False, raw_rank=23),
             _part("p2", False, False, raw_rank=None)]
    stats = summarize([_record("vector", parts)], top_k=5, extra_k=10)["overall"]

    assert stats["uncovered_rescuable"] == 1, "第 7 名：把 TOP_K 调到 10 本来能捞回来"
    assert stats["uncovered_out_of_reach"] == 2


def test_summarize_reports_sample_level_full_coverage():
    parts = [_part("a0", True, True), _part("a1", True, True)]
    stats = summarize([_record("vector", parts)], top_k=5, extra_k=10)["overall"]

    assert stats["samples_all_covered_at_5"] == 1
    assert stats["samples_all_covered_rate_at_5"] == 1.0


def test_summarize_of_nothing_is_an_empty_structure_not_a_crash():
    stats = summarize([], top_k=5, extra_k=10)

    assert stats["groups"] == []
    assert stats["overall"]["samples"] == 0
    assert stats["overall"]["parts_eligible"] == 0
    assert stats["overall"]["coverage_at_5"] is None
    assert stats["overall"]["gain"] is None, "没有分母时不许给一个会骗人的比例"


# ── 每问分路检索的对照实验（A/B）──
#
# 这三臂必须**只在「怎么分名额」上不同**：上下文预算一样、样本一样、分母一样。
# 下面几条就是这条纪律本身 —— 口径一旦松掉，分路臂就会靠「多塞几块」赢，
# 而那种赢法在上线时是要付 token 钱的。

def _hit(file_name, chunk_index, text="正文不参与判定"):
    """一个检索块：身份是 (文件名, 片号)，正文内容与判定无关。"""
    return (text, {"file_name": file_name, "chunk_index": chunk_index})


def _ids(hits):
    return [(meta.get("file_name"), meta.get("chunk_index")) for _text, meta in hits]


def test_hit_key_uses_chunk_identity_and_falls_back_to_text():
    assert hit_key(_hit("A", 3)) == ("A", 3)
    assert hit_key(("一段没有来源的正文", None)) == "一段没有来源的正文"


def test_interleave_gives_every_part_its_first_hit_before_any_second_hit():
    by_part = [[_hit("A", 0), _hit("A", 1), _hit("A", 2)],
               [_hit("B", 0), _hit("B", 1), _hit("B", 2)]]

    merged = interleave_hits(by_part, 5)

    assert _ids(merged) == [("A", 0), ("B", 0), ("A", 1), ("B", 1), ("A", 2)], \
        "名额必须轮着给：某一问的第 2 名不许把另一问的第 1 名挤掉"


def test_interleave_never_exceeds_the_budget_and_never_repeats_a_chunk():
    by_part = [[_hit("A", 0), _hit("A", 1)], [_hit("A", 1), _hit("B", 0)]]

    merged = interleave_hits(by_part, 5)

    assert _ids(merged) == [("A", 0), ("A", 1), ("B", 0)], "同一片被两问都召回时只占一个名额"
    assert len(interleave_hits(by_part, 2)) == 2


def test_interleave_keeps_going_when_one_part_returns_fewer_hits():
    by_part = [[_hit("A", 0)], [_hit("B", 0), _hit("B", 1), _hit("B", 2)]]

    assert _ids(interleave_hits(by_part, 2)) == [("A", 0), ("B", 0)]
    assert _ids(interleave_hits(by_part, 4)) == [("A", 0), ("B", 0), ("B", 1), ("B", 2)], \
        "一问没料时名额要让给还能出货的那些问，不许空转"


def test_interleave_of_nothing_is_empty_not_a_crash():
    assert interleave_hits([], 5) == []
    assert interleave_hits([[], []], 5) == []


def test_rrf_prefers_the_chunk_that_both_parts_found_near_the_top():
    by_part = [[_hit("A", 0), _hit("X", 9)], [_hit("B", 0), _hit("X", 9)]]

    merged = rrf_merge_hits(by_part, 1)

    assert _ids(merged) == [("X", 9)], \
        "两问都排第 2 的块，融合分高于只被一问排第 1 的块"


def test_rrf_agrees_with_the_online_fusion_rule():
    """脚本里的 RRF 必须和线上 `retrieval._rrf_fuse` 同一套算法，否则比的是两个东西。"""
    from src.retrieval import _rrf_fuse

    lists = [["甲甲", "乙"], ["乙", "丙丙"]]
    online = [text for text, _score in _rrf_fuse(lists)]
    local = _ids(rrf_merge_hits([[("甲甲", {"file_name": "甲甲", "chunk_index": 0}),
                                  ("乙", {"file_name": "乙", "chunk_index": 0})],
                                 [("乙", {"file_name": "乙", "chunk_index": 0}),
                                  ("丙丙", {"file_name": "丙丙", "chunk_index": 0})]], 5))

    assert [name for name, _index in local] == online


def test_route_retrieve_asks_each_part_with_its_own_question():
    asked = []

    def retrieve(question, k):
        asked.append(question)
        return "vector", []

    parts = [{"sub_question": "甲子问"}, {"sub_question": "乙子问"}]
    route_retrieve(parts, "甲子问？乙子问？", retrieve, 5, ROUTE_INTERLEAVE)

    assert asked == ["甲子问", "乙子问"], "分路检索必须用各问自己的问句去查"


def test_merged_arm_asks_the_whole_question_exactly_once():
    asked = []

    def retrieve(question, k):
        asked.append((question, k))
        return "vector", []

    path, hits, diag = route_retrieve([{"sub_question": "甲"}, {"sub_question": "乙"}],
                                      "甲？乙？", retrieve, 7, ROUTE_MERGED)

    assert asked == [("甲？乙？", 7)], "现状臂就是整句一次检索，k 就是预算"
    assert path == "vector" and hits == [] and diag["retrievals"] == 1


def test_route_retrieve_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        route_retrieve([{"sub_question": "甲"}], "甲？", lambda q, k: ("vector", []), 5, "瞎编")


def test_routed_arm_never_puts_more_than_the_budget_in_the_context():
    def retrieve(question, k):
        return "vector", [_hit(f"F{index}", index) for index in range(k)]

    parts = [{"sub_question": f"问{index}"} for index in range(4)]
    _path, hits, _diag = route_retrieve(parts, "问0？问1？问2？问3？", retrieve, 5, ROUTE_INTERLEAVE)

    assert len(hits) == 5, "4 问各召回 5 块，也不许把 20 块全塞进 context —— 预算和现状一样"


def test_path_label_says_mixed_when_parts_took_different_paths():
    answers = {"甲": ("vector", [_hit("A", 0)]), "乙": ("keyword", [_hit("B", 0)])}

    path, _hits, diag = route_retrieve([{"sub_question": "甲"}, {"sub_question": "乙"}],
                                       "甲？乙？", lambda q, k: answers[q], 5, ROUTE_INTERLEAVE)

    assert path == "mixed", "分路后每问可能走不同路径，汇总时必须如实标出来"
    assert diag["per_part"] == ["vector", "keyword"]


def test_run_sample_arm_merged_reproduces_the_existing_measurement():
    sample = make_sample([_probe("A", 0, "甲甲甲甲甲甲甲甲"),
                          _probe("B", 0, "乙乙乙乙乙乙乙乙")])
    for part in sample["parts"]:
        part["baseline_top1"] = True

    def retrieve(question, k):
        return "vector", [("甲甲甲甲甲甲甲甲在这里", {"file_name": "A", "chunk_index": 0})]

    old = run_sample(sample, retrieve, top_k=5, extra_k=10)
    arm = run_sample_arm(sample, retrieve, budget=5, mode=ROUTE_MERGED)

    assert [part["covered_top"] for part in old["parts"]] == \
        [part["covered"] for part in arm["parts"]], \
        "merged 臂必须逐位复现老口径，否则新旧数字没法比"
    assert arm["n_parts"] == 2 and arm["budget"] == 5 and arm["retrievals"] == 1


def test_routed_record_keeps_the_same_denominator_flags_as_the_merged_arm():
    sample = make_sample([_probe("A", 0, "甲甲甲甲甲甲甲甲"),
                          _probe("B", 0, "乙乙乙乙乙乙乙乙")])
    sample["parts"][0]["baseline_top1"] = True
    sample["parts"][1]["baseline_top1"] = False        # 单问基线就没命中 → 不进分母

    def retrieve(question, k):
        return "vector", []

    merged = run_sample_arm(sample, retrieve, budget=5, mode=ROUTE_MERGED)
    routed = run_sample_arm(sample, retrieve, budget=5, mode=ROUTE_INTERLEAVE)

    assert [part["eligible"] for part in routed["parts"]] == \
        [part["eligible"] for part in merged["parts"]] == [True, False]
    stats = summarize_routes([merged, routed], budget=5)
    assert all(group["parts_eligible"] == 1 for group in stats), \
        "分路臂不许偷偷换分母：两臂的分母都是「单问基线命中的子问题」"


def test_routed_record_never_carries_corpus_text_or_file_names():
    sample = make_sample([_probe("文件名不许落盘.md", 3, "甲甲甲甲甲甲甲甲")])
    sample["parts"][0]["baseline_top1"] = True

    record = run_sample_arm(
        sample,
        lambda q, k: ("vector", [("甲甲甲甲甲甲甲甲 正文",
                                  {"file_name": "文件名不许落盘.md", "chunk_index": 3})]),
        budget=5, mode=ROUTE_INTERLEAVE)

    detail = json.dumps(record, ensure_ascii=False)
    assert "甲甲甲甲甲甲甲甲" not in detail
    assert "文件名不许落盘" not in detail


def test_summarize_routes_reports_every_arm_with_sample_level_full_coverage():
    def _arm_record(arm, covered_flags):
        return {"id": "s", "arm": arm, "n_parts": len(covered_flags), "budget": 5,
                "path": "vector", "hits": 3, "retrievals": 1 if arm == ROUTE_MERGED else 2,
                "parts": [{"id": f"p{i}", "eligible": True, "baseline_top1": True,
                           "covered": flag, "strict": flag}
                          for i, flag in enumerate(covered_flags)]}

    stats = summarize_routes([_arm_record(ROUTE_MERGED, [True, False]),
                              _arm_record(ROUTE_INTERLEAVE, [True, True])], budget=5)
    by_arm = {group["arm"]: group for group in stats if group["n_parts"] is None}

    assert by_arm[ROUTE_MERGED]["covered"] == 1
    assert by_arm[ROUTE_MERGED]["coverage"] == 0.5
    assert by_arm[ROUTE_MERGED]["samples_all_covered"] == 0
    assert by_arm[ROUTE_INTERLEAVE]["coverage"] == 1.0
    assert by_arm[ROUTE_INTERLEAVE]["samples_all_covered"] == 1
    assert by_arm[ROUTE_INTERLEAVE]["mean_retrievals"] == 2, "成本要一起报：分路 = 每问查一次"


def test_summarize_routes_of_nothing_is_empty_not_a_crash():
    assert summarize_routes([], budget=5) == []


def test_arm_marks_a_hit_whose_anchor_lives_in_a_sibling_chunk_of_the_same_file():
    """「目标片没进 context」有两种：事实以**邻居片**的形式进来了（切片 100 字重叠），
    还是压根没进来。两者的身份口径都判 False，但对模型的意义完全不同 —— 必须分开记。"""
    sample = make_sample([_probe("A", 0, "甲甲甲甲甲甲甲甲"),
                          _probe("B", 0, "乙乙乙乙乙乙乙乙")])
    for part in sample["parts"]:
        part["baseline_top1"] = True

    def retrieve(question, k):
        return "vector", [("甲甲甲甲甲甲甲甲 出现在邻居片里", {"file_name": "A", "chunk_index": 1})]

    record = run_sample_arm(sample, retrieve, budget=5, mode=ROUTE_MERGED)

    first, second = record["parts"]
    assert first["covered"] is False, "身份口径：目标片 (A, 0) 没进 context"
    assert first["strict"] is True, "锚点可见：事实确实喂到了模型嘴边"
    assert first["same_file_anchor"] is True, "而且就在同一个文件的邻居片里"
    assert second["same_file_anchor"] is False, "乙的知识点没有被召回"


def test_arm_records_where_the_target_chunk_ranked_in_its_own_part():
    """分路臂漏一格有两种原因：那一问自己就没召回到目标片，还是召回到了但名额被截掉。"""
    sample = make_sample([_probe("A", 0, "甲甲甲甲甲甲甲甲"),
                          _probe("B", 0, "乙乙乙乙乙乙乙乙")])
    for part in sample["parts"]:
        part["baseline_top1"] = True

    def retrieve(question, k):
        if "甲甲甲甲甲甲甲甲" in question:
            return "vector", [_hit("A", 7), _hit("A", 0)]      # 目标片排第 2
        return "vector", [_hit("B", 0)]

    record = run_sample_arm(sample, retrieve, budget=5, mode=ROUTE_INTERLEAVE)

    assert [part["part_rank"] for part in record["parts"]] == [2, 1]
    assert [part["covered"] for part in record["parts"]] == [True, True]
    assert record["parts"][0]["own_hits"] == 2


def test_summarize_routes_splits_misses_into_sibling_chunk_and_elsewhere():
    def _arm_record(covered, same_file_anchor, strict=False):
        return {"id": "s", "arm": ROUTE_INTERLEAVE, "n_parts": 2, "budget": 5,
                "path": "vector", "hits": 5, "retrievals": 2, "per_part_paths": ["vector"] * 2,
                "parts": [{"id": "p0", "eligible": True, "baseline_top1": True,
                           "covered": covered, "strict": strict,
                           "same_file_anchor": same_file_anchor, "part_rank": None,
                           "own_hits": 1}]}

    stats = summarize_routes([_arm_record(False, True),
                              _arm_record(False, False),
                              _arm_record(False, False, strict=True)], budget=5)
    overall = next(group for group in stats if group["n_parts"] is None)

    assert overall["covered"] == 0
    assert overall["miss_same_file_anchor"] == 1, "一处只是「邻居片」口径盲区"
    assert overall["miss_elsewhere"] == 2, "另外两处是真的没进 context"
    assert overall["strict_cross_file"] == 1, \
        "「锚点可见」里来自别的文件的那一处要单独数，它会把这个口径吹到 100%"


def test_baseline_kind_separates_a_real_chunk_hit_from_an_anchor_only_hit():
    """分母本身也是同一把检索器定义的：锚点可见但来自**别的文件**的命中必须标出来。"""
    target = {"file_name": "A", "chunk_index": 0}
    anchor = "甲甲甲甲甲甲甲甲"

    assert baseline_kind([_hit("A", 0)], target, anchor) == "same_chunk"
    assert baseline_kind([("甲甲甲甲甲甲甲甲 正文", {"file_name": "B", "chunk_index": 1})],
                         target, anchor) == "anchor_only"
    assert baseline_kind([("完全不相干", {"file_name": "A", "chunk_index": 9})],
                         target, anchor) is None
    assert baseline_kind([], target, anchor) is None


def test_summarize_routes_also_reports_a_strict_denominator_of_real_chunk_baselines():
    record = {"id": "s", "arm": ROUTE_MERGED, "n_parts": 2, "budget": 5,
              "path": "vector", "hits": 5, "retrievals": 1, "per_part_paths": ["vector"],
              "parts": [
                  {"id": "p0", "eligible": True, "baseline_top1": True,
                   "baseline_kind": "same_chunk", "covered": True, "strict": True,
                   "same_file_anchor": True, "part_rank": None, "own_hits": None},
                  {"id": "p1", "eligible": True, "baseline_top1": True,
                   "baseline_kind": "anchor_only", "covered": True, "strict": True,
                   "same_file_anchor": False, "part_rank": None, "own_hits": None}]}

    overall = next(group for group in summarize_routes([record], budget=5)
                   if group["n_parts"] is None)

    assert overall["parts_eligible"] == 2
    assert overall["parts_strict_denominator"] == 1, "只有 top-1 真是目标片的那条进严格分母"
    assert overall["covered_strict_denominator"] == 1


def test_summarize_sample_level_coverage_skips_samples_without_any_eligible_part():
    """一个可用子问都没有的样本按定义算不出「全覆盖」，不能留在分母里当天然失败。"""
    good = _record("vector", [_part("p0", True, True), _part("p1", True, True)], "s1")
    nothing = _record("vector", [_part("p2", False, False, eligible=False),
                                 _part("p3", False, False, eligible=False)], "s2")

    stats = summarize([good, nothing], top_k=5, extra_k=10)["overall"]

    assert stats["samples"] == 2
    assert stats["samples_with_eligible"] == 1
    assert stats[f"samples_all_covered_rate_at_5"] == 1.0
    routes = summarize_routes(
        [{"id": "s1", "arm": ROUTE_MERGED, "n_parts": 2, "budget": 5, "path": "vector",
          "hits": 2, "retrievals": 1, "per_part_paths": ["vector"],
          "parts": [{"id": "a", "eligible": True, "baseline_top1": True,
                     "baseline_kind": "same_chunk", "covered": True, "strict": True,
                     "same_file_anchor": True},
                    {"id": "b", "eligible": True, "baseline_top1": True,
                     "baseline_kind": "same_chunk", "covered": True, "strict": True,
                     "same_file_anchor": True}]},
         {"id": "s2", "arm": ROUTE_MERGED, "n_parts": 2, "budget": 5, "path": "vector",
          "hits": 2, "retrievals": 1, "per_part_paths": ["vector"],
          "parts": [{"id": "c", "eligible": False, "baseline_top1": False,
                     "baseline_kind": None, "covered": False, "strict": False,
                     "same_file_anchor": False}]}], budget=5)
    overall = next(group for group in routes if group["n_parts"] is None)

    assert overall["samples_with_eligible"] == 1
    assert overall["samples_all_covered_rate"] == 1.0


# ── 拆问臂：子问题由模型现场拆（线上唯一可行的形态）──

def test_decompose_arm_retrieves_with_the_model_questions_not_the_gold_ones():
    sample = make_sample([_probe("A", 0, "甲甲甲甲甲甲甲甲"),
                          _probe("B", 0, "乙乙乙乙乙乙乙乙")])
    for part in sample["parts"]:
        part["baseline_top1"] = True
    asked = []

    def retrieve(question, k):
        asked.append(question)
        return "vector", [_hit("A", 0)]

    route_retrieve(sample["parts"], sample["question"], retrieve, 5, ROUTE_DECOMPOSE,
                   decompose=lambda question: ["模型拆的问一", "模型拆的问二"])

    assert asked == ["模型拆的问一", "模型拆的问二"], \
        "拆问臂只许用模型拆出来的问句去检索，不许偷看 gold 子问题"


def test_decompose_arm_falls_back_to_the_whole_question_when_not_split():
    sample = make_sample([_probe("A", 0, "甲甲甲甲甲甲甲甲"),
                          _probe("B", 0, "乙乙乙乙乙乙乙乙")])
    asked = []

    def retrieve(question, k):
        asked.append(question)
        return "vector", []

    _path, _hits, diag = route_retrieve(sample["parts"], sample["question"], retrieve, 5,
                                        ROUTE_DECOMPOSE,
                                        decompose=lambda question: [question])

    assert asked == [sample["question"]], "拆不出来就退回整句，只查一次"
    assert diag["fallback"] is True and diag["retrievals"] == 1


def test_decompose_arm_reports_which_gold_part_no_split_question_could_reach():
    """拆漏：某条 gold 子问的目标片，任何一条拆出来的问句都没捞到 —— 这是拆问那一步丢的。"""
    sample = make_sample([_probe("A", 0, "甲甲甲甲甲甲甲甲"),
                          _probe("B", 0, "乙乙乙乙乙乙乙乙")])
    for part in sample["parts"]:
        part["baseline_top1"] = True

    def retrieve(question, k):
        if "甲" in question:
            return "vector", [_hit("A", 0)]
        return "vector", [_hit("C", 0)]              # 第二问拆歪了，捞到别的文件

    record = run_sample_arm(sample, retrieve, budget=5, mode=ROUTE_DECOMPOSE,
                            decompose=lambda question: ["甲甲甲甲甲甲甲甲的说说", "拆歪了的问句"])

    first, second = record["parts"]
    assert first["recalled_by_split"] is True
    assert first["covered"] is True
    assert second["recalled_by_split"] is False, "第二问的正文一条拆出来的问句都没捞到"
    assert record["generated"] == 2
    assert record["decompose_fallback"] is False


def test_decompose_arm_needs_a_decomposer():
    with pytest.raises(ValueError):
        route_retrieve([{"sub_question": "甲"}], "甲？", lambda q, k: ("vector", []), 5,
                       ROUTE_DECOMPOSE)


def test_over_splitting_past_the_budget_still_fits_the_context():
    """模型拆过头（问数 > 预算）时，context 不许超预算 —— 代价只能是「有的问一个名额都没有」。"""
    sample = make_sample([_probe("A", 0, "甲甲甲甲甲甲甲甲"),
                          _probe("B", 0, "乙乙乙乙乙乙乙乙")])
    for part in sample["parts"]:
        part["baseline_top1"] = True

    def retrieve(question, k):
        return "vector", [_hit("A", 0), _hit("B", 0)]

    record = run_sample_arm(sample, retrieve, budget=2, mode=ROUTE_DECOMPOSE,
                            decompose=lambda question: [f"第{i}个问句" for i in range(6)])

    assert record["generated"] == 6
    assert record["retrievals"] == 6, "拆出 6 问就要查 6 次（成本随问数线性涨）"
    assert record["hits"] == 2, "预算 2 但去重后只有 2 块，绝不超预算"


def test_summarize_routes_reports_split_recall_and_fallbacks():
    def _record(recalled, fallback, generated):
        return {"id": "s", "arm": ROUTE_DECOMPOSE, "n_parts": 2, "budget": 5,
                "path": "vector", "hits": 5, "retrievals": generated,
                "generated": generated, "decompose_fallback": fallback,
                "per_part_paths": ["vector"] * generated,
                "parts": [{"id": "p0", "eligible": True, "baseline_top1": True,
                           "baseline_kind": "same_chunk", "covered": True, "strict": True,
                           "same_file_anchor": True, "recalled_by_split": recalled},
                          {"id": "p1", "eligible": False, "baseline_top1": False,
                           "baseline_kind": None, "covered": False, "strict": False,
                           "same_file_anchor": False, "recalled_by_split": False}]}

    stats = summarize_routes([_record(True, False, 2), _record(False, True, 1)], budget=5)
    overall = next(group for group in stats if group["n_parts"] is None)

    assert overall["parts_split_denominator"] == 2, \
        "拆漏的分母也要是 eligible：本来就检不到的子问不算拆问的锅"
    assert overall["parts_split_recalled"] == 1
    assert overall["decompose_fallbacks"] == 1
    assert overall["mean_generated"] == 1.5


def test_compound_question_can_hide_its_boundaries():
    """qmark 是简单模式（分隔符就是答案）；comma / runon 才逼模型自己找边界。"""
    pieces = ["年假有几天", "迟到怎么算"]

    assert build_compound_question(pieces, style="qmark") == "年假有几天？迟到怎么算？"
    assert build_compound_question(pieces, style="comma") == "年假有几天，迟到怎么算？"
    assert build_compound_question(pieces, style="runon") == "年假有几天还有迟到怎么算？"


# ── 出错路径：返回非 0 且是中文提示 ──

def test_main_rejects_a_missing_state_dir(tmp_path, capsys):
    from scripts import eval_compound_retrieval as ecr

    code = ecr.main(["--state-dir", str(tmp_path / "nope")])
    out = capsys.readouterr().out

    assert code == 2
    assert "状态目录不存在" in out


def test_main_rejects_a_state_dir_without_any_file(tmp_path, capsys):
    from scripts import eval_compound_retrieval as ecr

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "sid.json").write_text(json.dumps({
        "version": 2, "session_id": "sid",
        "conversations": {"c1": {"files": [], "messages": []}},
    }, ensure_ascii=False), encoding="utf-8")

    code = ecr.main(["--state-dir", str(state_dir)])
    out = capsys.readouterr().out

    assert code == 2, "0 个可测对话不许静默跑出 0 个样本"
    assert "带文件的对话" in out
