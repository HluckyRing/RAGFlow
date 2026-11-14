# -*- coding: utf-8 -*-
"""闸门措辞包（`src/gates/`）与「闸门失效必须看得见」的回归（2026-11-12 第十七轮）。

闸门（要不要多花一次调用 / 要不要改变排序）靠**措辞**判断意图。措辞表天生跟语言与语料
绑死：换英文语料一个字都命中不了。这类失效是 fail-safe 的（闸门不触发 → 退回改造前行为），
真正危险的是**静默** —— 功能悄悄不生效而没人发现（HyDE 当年就这样静默失效过很久）。

所以这里钉住：
- 包能选、缺字段不崩（按空表处理 = 闸门不触发，fail-safe）；
- 未知包名退回 `zh` 且**告警**；
- 闸门的命中 / 未命中都要留下可数的日志（`闸门 compound=hit|miss`）；
- 不在包里的两张表（`PRONOUNS` / `REFUSAL_MARKER`）不许被顺手挪进来 —— 它们不是
  「降级即可」的闸门：前者影响答案正确性、后者是前后端契约。
"""
import logging

import pytest

import src.config as config
import src.coverage as coverage
import src.decompose as decompose
from src.gates import PACKS, load_pack, pack_names, zh


def test_default_pack_is_zh_and_has_all_four_tables():
    pack = load_pack("zh")
    assert pack.name == "zh"
    for field in ("compound_markers", "question_words",
                  "enumeration_markers", "navigation_markers"):
        assert getattr(pack, field), f"{field} 不该是空的"


def test_pack_names_lists_zh():
    assert "zh" in pack_names()


def test_unknown_pack_falls_back_to_zh_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        pack = load_pack("en-not-yet", logger=config.logger)
    assert pack.name == "zh", "拼错包名不许把服务搞挂，但也绝不能安静地换口径"
    assert any("未知的 GATE_PACK" in record.message for record in caplog.records)


def test_pack_with_missing_field_degrades_to_empty_tables(caplog, monkeypatch):
    """缺字段按空表处理：闸门不触发（fail-safe），绝不 AttributeError。"""
    class _Partial:
        COMPOUND_MARKERS = ("还有",)

    monkeypatch.setitem(PACKS, "partial", _Partial)
    with caplog.at_level(logging.WARNING):
        pack = load_pack("partial", logger=config.logger)
    assert pack.compound_markers == ("还有",)
    assert pack.question_words == ()
    assert sum(1 for record in caplog.records if "缺少" in record.message) == 3


def test_modules_read_the_tables_from_the_pack():
    """decompose / coverage 用的是包里的表，不是各自抄一份。"""
    assert decompose.COMPOUND_MARKERS == zh.COMPOUND_MARKERS
    assert coverage.ENUMERATION_MARKERS == zh.ENUMERATION_MARKERS
    assert coverage.NAVIGATION_MARKERS == zh.NAVIGATION_MARKERS


def test_gates_still_work_through_the_pack():
    assert decompose.looks_compound("年假有几天还有迟到怎么算") is True
    assert decompose.looks_compound("这门课有几个学时") is False
    assert coverage.looks_enumerative("列出所有的实验") is True
    assert coverage.looks_enumerative("数据库评分标准是什么") is False


def test_compound_gate_logs_hit_and_miss(caplog, monkeypatch):
    """命中与未命中都要留下可数日志 —— 触发率掉到 0 必须能被发现。"""
    monkeypatch.setattr(decompose, "client", None)      # 别真的调模型

    with caplog.at_level(logging.INFO):
        decompose.decompose_question("两件事：甲是什么，乙呢？")   # 命中
        decompose.decompose_question("这门课有几个学时")           # 未命中

    messages = [record.message for record in caplog.records]
    assert any("闸门 compound=hit" in m for m in messages)
    assert any("闸门 compound=miss" in m for m in messages)


def test_pronouns_and_refusal_marker_stay_out_of_the_gate_pack():
    """这两张表不是闸门：一个影响答案正确性、一个是前后端契约，不许被打包成「可降级」。"""
    from src import llm, prompts

    assert llm.PRONOUNS, "指代词表仍在 llm 里（换语言要单独做策略，不能靠降级）"
    assert prompts.REFUSAL_MARKER == "【无答案】", "拒答标记是契约，不许跟着措辞包变"
    assert not hasattr(zh, "PRONOUNS")
    assert not hasattr(zh, "REFUSAL_MARKER")
