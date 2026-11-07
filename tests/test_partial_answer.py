# -*- coding: utf-8 -*-
"""「部分可答」分桶的回归。

一个问句里一半能从语料里答、一半答不了，是产品里最常见的形状。这里把它的四种结局钉住：
整段拒答（过度拒答）/ 答了可答部分并点明缺口（理想）/ 答了但只字不提缺口 / 可答部分也没答到。
"""
import pytest

from scripts import measure_inline_abstention as mia
from src.prompts import REFUSAL_MARKER

ANSWER_TERMS = ["保留 30 天"]
ABSENT_TERMS = ["异地机房"]


@pytest.mark.parametrize("answer, expected", [
    # 整条消息被标成拒答、而且可答部分也没答 —— 最糟
    (REFUSAL_MARKER + "参考背景没有提到异地机房怎么切换。", "refused_all"),
    # 标成拒答但正文其实答了可答部分 —— 用户仍然会看到「知识库里没有」，属于过度拒答
    (REFUSAL_MARKER + "参考背景没有异地机房。另外全量备份保留 30 天。", "refused_despite_answer"),
    # 理想：答了可答部分，也点明缺什么（不写标记）
    ("全量备份保留 30 天。参考背景中没有提到异地机房的切换步骤。", "answered_with_gap"),
    ("全量备份保留 30 天；异地机房方面材料未涉及。", "answered_with_gap"),
    # 答了可答部分但对缺的那半只字不提 —— 有含糊/编造风险
    ("全量备份保留 30 天，增量备份保留 7 天。", "answered_silently"),
    # 可答部分都没答到、也没提缺口
    ("备份是重要的运维工作，建议定期检查。", "neither"),
])
def test_classify_partial_buckets(answer, expected):
    assert mia.classify_partial(answer, ANSWER_TERMS, ABSENT_TERMS) == expected


def test_classify_partial_needs_both_the_topic_and_an_absence_word():
    """只提到「异地机房」但没说它缺，不算点明缺口（那可能是在编造）。"""
    answer = "全量备份保留 30 天。异地机房的切换窗口建议安排在周末。"

    assert mia.classify_partial(answer, ANSWER_TERMS, ABSENT_TERMS) == "answered_silently"


def test_classify_partial_treats_empty_answer_as_refusal():
    assert mia.classify_partial("", ANSWER_TERMS, ABSENT_TERMS) == "refused_all"


# ── 复合问题的三种形状（2026-11-08 第二批）：三问 / 缺口在中间 / 多轮补问 ──
#
# 五桶口径只回答「整条消息有没有点明缺口」，回答不了「第 3 问被吞了没有」。三个子问题按序列出
# （`parts`）之后才能逐个子问题判生死，于是多出「漏答可答项」与「该拒却没写标记」两种失败。

GAP_IN_MIDDLE_PARTS = [
    {"text": "全量备份保留多久", "answer_terms": ["保留 30 天"], "absent_terms": []},
    {"text": "异地机房怎么切换", "answer_terms": [], "absent_terms": ["异地机房"]},
    {"text": "恢复演练多久一次", "answer_terms": ["每半年至少组织一次"], "absent_terms": []},
]


def test_classify_partial_flags_a_dropped_answerable_part():
    """点明了缺口、却把缺口后面那一问吞了 —— 五桶口径会把它判成 `answered_with_gap`（达标）。"""
    answer = "全量备份保留 30 天。参考背景中没有提到异地机房的切换方式。"

    assert mia.classify_partial(
        answer, ["保留 30 天", "每半年至少组织一次"], ["异地机房"]) == "dropped_part_with_gap"


def test_classify_partial_flags_a_dropped_part_even_without_the_gap():
    answer = "全量备份保留 30 天，增量备份保留 7 天。"

    assert mia.classify_partial(
        answer, ["保留 30 天", "每半年至少组织一次"], ["异地机房"]) == "dropped_part_silent"


def test_classify_partial_rewards_all_parts_answered_with_the_gap():
    answer = ("全量备份保留 30 天。恢复演练每半年至少组织一次。"
              "参考背景中没有提到异地机房的切换方式。")

    assert mia.classify_partial(
        answer, ["保留 30 天", "每半年至少组织一次"], ["异地机房"]) == "answered_with_gap"


@pytest.mark.parametrize("answer, expected", [
    # 这条样本本来就答不了：说了「材料里没有」却没写标记 —— 行为对、契约没守（前端不显示拒答标签）
    ("参考背景中没有病假工资的规定。", "refused_without_marker"),
    ("材料里未涉及病假工资。", "refused_without_marker"),
    # 直接硬答 —— 该拒没拒
    ("病假工资通常按正常工资的 60% 发放。", "answered_without_refusal"),
    # 写了标记：这才是服务端能推 refused 事件、前端能渲染拒答标签的形态
    (REFUSAL_MARKER + "参考背景中没有病假工资的规定。", "refused_all"),
])
def test_classify_partial_judges_samples_that_should_be_refused(answer, expected):
    """`expect_refused: true` 的样本（可答项为空）只判「拒没拒、拒得合不合契约」。"""
    assert mia.classify_partial(answer, [], ["病假工资"]) == expected


def test_classify_partial_ignores_spacing_differences_in_the_terms():
    """模型写「每年5天」、标注写「每年 5 天」不该判成漏答 —— 假失败会逼人去修不存在的问题。"""
    answer = "根据《02-考勤与假期制度.md》，司龄 1 至 3 年每年5天。"

    assert mia.classify_partial(answer, ["每年 5 天"], []) == "answered_silently"


# ── 逐个子问题判生死（三问 / 缺口在中间的关键粒度）──

def test_part_coverage_marks_each_part_separately():
    answer = "全量备份保留 30 天。恢复演练每半年至少组织一次。参考背景没有提到异地机房。"

    rows = mia.part_coverage(answer, GAP_IN_MIDDLE_PARTS)

    assert [row["answerable"] for row in rows] == [True, False, True]
    assert [row["ok"] for row in rows] == [True, True, True]


def test_post_gap_coverage_sees_the_swallowed_third_question():
    """缺口后面那一问被静默丢掉 —— 整条消息的桶看着仍然「达标」，只有这个数字露馅。"""
    answer = "全量备份保留 30 天。参考背景中没有提到异地机房的切换方式。"

    assert mia.post_gap_coverage(answer, GAP_IN_MIDDLE_PARTS) == {
        "covered": 0, "total": 1, "parts": ["恢复演练多久一次"]}


def test_post_gap_coverage_counts_all_answerable_parts_after_the_gap():
    answer = "全量备份保留 30 天。恢复演练每半年至少组织一次。材料里没有提到异地机房。"

    assert mia.post_gap_coverage(answer, GAP_IN_MIDDLE_PARTS)["covered"] == 1


def test_post_gap_coverage_is_none_without_a_gap_before_an_answerable_part():
    parts = [{"text": "甲", "answer_terms": ["保留 30 天"], "absent_terms": []},
             {"text": "乙", "answer_terms": [], "absent_terms": ["异地机房"]}]

    assert mia.post_gap_coverage("全量备份保留 30 天。", parts) is None


# ── 按形状汇总：该答的与该拒的绝不混进同一个分母 ──

def _row(row_id, shape, bucket, answer_terms=1, covered=1, post_gap=None,
         expect_refused=False):
    return {"id": row_id, "shape": shape, "bucket": bucket,
            "expect_refused": expect_refused,
            "coverage": {"covered": covered, "total": answer_terms},
            "post_gap": post_gap}


def test_summarize_partial_splits_should_answer_and_should_refuse():
    rows = [
        _row("p09", "three_parts", "answered_with_gap"),
        _row("p10", "three_parts", "dropped_part_with_gap", answer_terms=2, covered=1),
        _row("f01", "followup", "refused_all", answer_terms=0, covered=0, expect_refused=True),
        _row("f02", "followup", "refused_without_marker", answer_terms=0, covered=0,
             expect_refused=True),
        _row("f03", "followup", "answered_without_refusal", answer_terms=0, covered=0,
             expect_refused=True),
    ]

    summary = mia.summarize_partial(rows)

    assert summary["three_parts"]["should_answer"] == 2
    assert summary["three_parts"]["should_refuse"] == 0
    assert summary["three_parts"]["dropped_part"] == ["p10"]
    assert summary["three_parts"]["full_coverage"] == ["p09"]
    assert summary["three_parts"]["answer_terms_covered"] == 2
    assert summary["three_parts"]["answer_terms_total"] == 3
    # 该拒的那三条里，只有 refused_all 是「拒了且契约对」；另外两条分别缺标记和硬答
    assert summary["followup"]["should_refuse"] == 3
    assert summary["followup"]["refused_ok"] == ["f01"]
    assert summary["followup"]["refused_without_marker"] == ["f02"]
    assert summary["followup"]["answered_without_refusal"] == ["f03"]
    assert summary["followup"]["over_refused"] == []


def test_summarize_partial_does_not_count_a_correct_refusal_as_over_refusal():
    """该拒的样本拒了，不能被算进「过度拒答」—— 那会把正确行为报成失败。"""
    summary = mia.summarize_partial(
        [_row("f01", "followup", "refused_all", answer_terms=0, covered=0, expect_refused=True)])

    assert summary["followup"]["over_refused"] == []
    assert summary["followup"]["refused_ok"] == ["f01"]


def test_summarize_partial_aggregates_post_gap_coverage():
    summary = mia.summarize_partial([
        _row("p11", "gap_in_middle", "answered_with_gap",
             post_gap={"covered": 0, "total": 1, "parts": ["恢复演练多久一次"]}),
        _row("p12", "gap_in_middle", "answered_with_gap",
             post_gap={"covered": 1, "total": 1, "parts": ["餐补标准"]}),
    ])

    assert summary["gap_in_middle"]["post_gap_covered"] == 1
    assert summary["gap_in_middle"]["post_gap_total"] == 2


def test_summarize_partial_is_empty_for_no_rows():
    assert mia.summarize_partial([]) == {}


def test_print_partial_summary_labels_shapes_and_post_gap():
    lines = []

    mia.print_partial_summary(
        [_row("p11", "gap_in_middle", "answered_with_gap",
              post_gap={"covered": 0, "total": 1, "parts": ["恢复演练多久一次"]})],
        log=lines.append)

    text = "\n".join(lines)
    assert "缺口在中间" in text
    assert "缺口后一问存活    = 0/1" in text


# ── 多轮补问：history 必须真的进 resolve_query，不能静默丢掉 ──

def test_plan_search_queries_passes_history_into_resolution():
    seen = []

    def resolve(question, history):
        seen.append((question, history))
        return "消解后：" + question

    queries = [
        {"question": "这个要多久", "history": [{"role": "user", "content": "年假有几天"}]},
        {"question": "市盈率怎么算"},
    ]

    got = mia.plan_search_queries(queries, resolve)

    assert got == ["消解后：这个要多久", "消解后：市盈率怎么算"]
    assert seen[0][1] == [{"role": "user", "content": "年假有几天"}]
    assert seen[1][1] == []


def test_matcher_ignores_markdown_emphasis_inside_a_term():
    """模型把锚点写在 **粗体** 里不该判成漏答（2026-11-08 实测踩到：这条假失败会骗人去改提示词）。"""
    answer = "根据《04-数据备份与恢复.md》，全量备份保留 **30 天**，增量备份保留 7 天。"

    assert mia.classify_partial(answer, ["保留 30 天"], []) == "answered_silently"


def test_matcher_ignores_chinese_quotes_inside_a_term():
    answer = "材料里写的是：窑温“通常分三段缓慢抬升”，降温过快会让器物内部产生应力。"

    assert mia.classify_partial(answer, ["分三段缓慢抬升", "产生应力"], []) == "answered_silently"


def test_matcher_normalises_fullwidth_and_halfwidth_forms():
    answer = "调休需要在加班之日起 6 个月内使用（逾期作废）。"

    assert mia.classify_partial(answer, ["6 个月内"], []) == "answered_silently"


def test_matcher_does_not_paper_over_a_real_paraphrase():
    """边界：抹装饰可以，**同义改写不行** —— 「可以折算」与「可折算」不是同一个字符串。

    放宽到这一步就是把失败藏起来了：模型少答了原文表述，应当由标注（最短事实核）去适配，
    而不是让判定器学会脑补。
    """
    answer = "工作日加班满 1 小时起算，加班时长可折算为调休。"

    assert mia.classify_partial(answer, ["可以折算为调休"], []) == "neither"


def test_gap_flagging_accepts_any_of_several_topic_words():
    """缺口主题词要允许同义说法：模型写「参加人员」而不是问句里的「请哪些人参加」。"""
    answer = "恢复演练每半年至少组织一次。关于参加人员：材料里没有涉及。"

    assert mia.classify_partial(
        answer, ["每半年至少组织一次"], ["请哪些人参加", "参加人员"]) == "answered_with_gap"


# ── 多缺口：扁平字段看不见「只点了第一处」 ──
#
# `classify_partial` 拿的是**平铺**的 absent_terms，判据是「任一个主题词出现 + 有个缺席词」。
# 所以「四问两缺口」的样本里模型只点明第一处，整条消息的桶仍然是 answered_with_gap。
# 多缺口的漏答率必须按子问题逐处算（`gap_coverage`）—— 这是 parts 存在的第二个理由。

MULTI_GAP_PARTS = [
    {"text": "年假有几天", "answer_terms": ["每年 5 天"], "absent_terms": []},
    {"text": "病假工资怎么算", "answer_terms": [], "absent_terms": ["病假工资", "病假期间的工资"]},
    {"text": "补卡一个月能补几次", "answer_terms": ["补卡不超过三次"], "absent_terms": []},
    {"text": "病假最多能请多久", "answer_terms": [], "absent_terms": ["病假天数", "病假期限"]},
]


def test_flat_bucket_cannot_see_the_dropped_second_gap():
    """扁平口径会把「只点明一处缺口」判成达标 —— 这条把它的盲区钉成事实，别当成 bug 去改。"""
    answer = ("年假每年 5 天；每月补卡不超过三次。"
              "病假工资方面材料里没有涉及。")   # 只点了第一处缺口

    assert mia.classify_partial(
        answer, ["每年 5 天", "补卡不超过三次"],
        ["病假工资", "病假期间的工资", "病假天数", "病假期限"]) == "answered_with_gap"
    assert mia.gap_coverage(answer, MULTI_GAP_PARTS) == {
        "flagged": 1, "total": 2, "unflagged": ["病假最多能请多久"]}


def test_gap_coverage_counts_every_gap():
    answer = ("年假每年 5 天；每月补卡不超过三次。病假工资材料里没有涉及；"
              "至于病假期限，材料里也没有规定。")

    assert mia.gap_coverage(answer, MULTI_GAP_PARTS)["flagged"] == 2


def test_gap_coverage_is_none_without_any_gap():
    parts = [{"text": "年假有几天", "answer_terms": ["每年 5 天"], "absent_terms": []}]

    assert mia.gap_coverage("每年 5 天。", parts) is None


def test_summarize_partial_aggregates_gap_coverage():
    rows = [
        {"id": "p20", "shape": "multi_gap", "bucket": "answered_with_gap",
         "expect_refused": False, "coverage": {"covered": 2, "total": 2},
         "gaps": {"flagged": 1, "total": 2, "unflagged": ["病假最多能请多久"]}},
        {"id": "p21", "shape": "multi_gap", "bucket": "answered_with_gap",
         "expect_refused": False, "coverage": {"covered": 2, "total": 2},
         "gaps": {"flagged": 2, "total": 2, "unflagged": []}},
    ]

    summary = mia.summarize_partial(rows)["multi_gap"]

    assert summary["gaps_flagged"] == 3
    assert summary["gaps_total"] == 4
    assert summary["all_gaps_flagged"] == ["p21"], "只点明一处的那条不能算「缺口全点明」"


def test_print_partial_summary_reports_gap_coverage():
    lines = []
    mia.print_partial_summary(
        [{"id": "p20", "shape": "multi_gap", "bucket": "answered_with_gap",
          "expect_refused": False, "coverage": {"covered": 2, "total": 2},
          "gaps": {"flagged": 1, "total": 2, "unflagged": ["病假最多能请多久"]}}],
        log=lines.append)

    assert any("缺口全点明        = 1/2" in line for line in lines)


def test_summarize_partial_excludes_should_refuse_rows_from_gap_coverage():
    """该拒的样本正确行为是**整条拒答**：模型往往只说「材料里没有涉及」而不复述主题词。

    拿它算「缺口没点明」会冤枉正确行为（p22 就因此被算成 0/1）。所以缺口全点明只统计该答的样本。
    """
    rows = [
        {"id": "p22", "shape": "followup_partial", "bucket": "refused_all",
         "expect_refused": True, "coverage": {"covered": 0, "total": 0},
         "gaps": {"flagged": 0, "total": 1, "unflagged": ["那病假期间的工资到底怎么算"]}},
        {"id": "p20", "shape": "multi_gap", "bucket": "answered_with_gap",
         "expect_refused": False, "coverage": {"covered": 2, "total": 2},
         "gaps": {"flagged": 2, "total": 2, "unflagged": []}},
    ]

    summary = mia.summarize_partial(rows)

    assert summary["followup_partial"]["gaps_total"] == 0, "该拒的样本不该进缺口分母"
    assert summary["multi_gap"]["gaps_flagged"] == 2
