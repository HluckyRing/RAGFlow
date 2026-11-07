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
