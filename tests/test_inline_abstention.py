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
