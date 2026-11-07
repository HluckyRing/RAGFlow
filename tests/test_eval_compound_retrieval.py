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
    anchor_candidates,
    anchor_fits,
    anchor_visible,
    baseline_hit,
    build_compound_question,
    build_probes,
    build_samples,
    build_sub_question,
    covered,
    find_leaks,
    hit_match,
    make_sample,
    measure_baseline,
    pick_anchor,
    run_sample,
    sample_files,
    summarize,
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
