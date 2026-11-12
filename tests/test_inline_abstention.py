# -*- coding: utf-8 -*-
"""「让 QA 自己拒答」（inline）这条路的判定口径回归。

拒答判定已经从「文案启发式」换成**结构化标记**（`prompts.REFUSAL_MARKER`）：
- 提示词要求模型答不出时以标记开头，服务端据此推 `refused` 事件；
- 测量脚本 `scripts/measure_inline_abstention.py` 用同一套契约判「这条算不算拒答」。

所以这里钉住：认标记、认 markdown 加粗、**不**把正文中间出现的标记当拒答、空回答算拒答。
"""
import pytest

from scripts import measure_inline_abstention as mia
from src.prompts import REFUSAL_MARKER


@pytest.mark.parametrize("answer", [
    REFUSAL_MARKER,
    REFUSAL_MARKER + "参考背景只写了保留策略，没有说明异地机房步骤。",
    "**" + REFUSAL_MARKER + "**材料里没有相关内容。",
    "  " + REFUSAL_MARKER + "没有。",
    "",
    "   ",
])
def test_recognises_real_abstention(answer):
    assert mia.looks_like_abstention(answer) is True


@pytest.mark.parametrize("answer", [
    "根据《01-财务指标.md》，市盈率（PE）等于股价除以每股收益。",
    "净资产收益率是净利润与平均股东权益的百分比。该文件还提到杜邦分析把它拆成三部分。",
    # 正文中间提到标记的用法 —— 那是正常回答，不是拒答
    "根据《a.md》，" + REFUSAL_MARKER + "这种标记用于拒答。",
])
def test_does_not_flag_a_normal_answer(answer):
    assert mia.looks_like_abstention(answer) is False

# ── 拆问臂：两臂只差一个开关（2026-11-12 第十一轮）──
#
# 这条路的纪律是「不许在这层重新实现拆问」：`decomposed` 是线上 `llm._retrieve_hits`
# 的薄包装，「拆出 1 条退回整句」「分路全空退回整句」「每条子问过 resolve_query」都归
# 线上那一份实现。这里只钉住「开关有没有生效」和「关着时一次多余调用都没有」。

def test_pick_hits_uses_whole_retrieval_when_decompose_is_off():
    calls = []

    def whole(question):
        calls.append(("whole", question))
        return ["整句块"]

    def decomposed(question):
        calls.append(("decomposed", question))
        raise AssertionError("关着拆问时不该调用分路取块")

    assert mia.pick_hits("问一句", whole, decomposed=decomposed, decompose=None) == ["整句块"]
    assert calls == [("whole", "问一句")]


def test_pick_hits_uses_decomposed_retrieval_when_switched_on():
    calls = []

    def whole(question):
        calls.append(("whole", question))
        raise AssertionError("开着拆问时不该只走整句")

    def decomposed(question):
        calls.append(("decomposed", question))
        return ["子问块1", "子问块2"]

    hits = mia.pick_hits("复合问", whole, decomposed=decomposed, decompose=lambda q: ["a", "b"])
    assert hits == ["子问块1", "子问块2"]
    assert calls == [("decomposed", "复合问")]


def test_pick_hits_falls_back_to_whole_when_no_decomposed_backend():
    """防御性：给了 decompose 但没给分路取块函数时不许崩，退回整句。"""
    assert mia.pick_hits("问", lambda q: ["整句块"], decomposed=None,
                         decompose=lambda q: ["a"]) == ["整句块"]


def test_top_k_is_shared_by_both_arms():
    """两臂预算必须一样：拆问臂「每问各查 top_k 再分名额到 top_k」，整句臂一次 top_k。"""
    assert mia.TOP_K == 5, "改预算会同时改变两臂；改之前先想清楚这条实验还比什么"
