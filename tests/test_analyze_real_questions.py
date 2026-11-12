# -*- coding: utf-8 -*-
"""真实用户提问盘点（`scripts/analyze_real_questions.py`）的纯函数回归。

这个脚本量的是「真实流量到底怎么问」，结论会直接影响产品决策（拆问该不该默认开、
要不要做跨文件取样），所以几个计数口径必须钉死：

- 只数 `role == "user"` 的提问，空内容跳过；**不去重** —— 同一问题问两遍本身是信息；
- 闸门抛错按「不触发」记，一条坏数据不许把整张表带走；
- 文件覆盖数的是**装配后**（模型真的看到）的文件，不是检索返回的；
- 枚举/汇总措辞只用于分组，不参与判对错。
"""
import pytest

from scripts.analyze_real_questions import (
    assembly_report,
    gate_report,
    looks_like_enumeration,
    real_questions,
    summarize,
)


def _conv(messages, files=None):
    return {"messages": messages, "files": files or []}


CONVERSATIONS = {
    "c1": _conv([
        {"role": "user", "content": "有哪些题型"},
        {"role": "assistant", "content": "根据参考背景……"},
        {"role": "user", "content": "   "},                      # 空内容跳过
        {"role": "user", "content": "实验一是什么"},
        {"role": "user", "content": "实验一是什么"},              # 重复保留
    ]),
    "c2": _conv([
        {"role": "user", "content": "用三点总结主要内容"},
        {"role": "system", "content": "系统消息不算提问"},
    ]),
}


def test_real_questions_only_takes_user_messages_and_keeps_duplicates():
    rows = real_questions(CONVERSATIONS)
    assert [row["question"] for row in rows] == ["有哪些题型", "实验一是什么", "实验一是什么",
                                                 "用三点总结主要内容"]
    assert [row["conv_id"] for row in rows] == ["c1", "c1", "c1", "c2"]


def test_real_questions_handles_empty_state():
    assert real_questions({}) == []
    assert real_questions(None) == []


def test_gate_report_counts_hits_and_survives_a_throwing_gate():
    def gate(question):
        if "炸" in question:
            raise RuntimeError("闸门坏了")
        return question.startswith("实验")

    rows = [{"conv_id": "c1", "question": "实验一是什么"},
            {"conv_id": "c1", "question": "有哪些题型"},
            {"conv_id": "c2", "question": "炸一下"}]
    count, triggered, detail = gate_report(rows, gate)
    assert count == 1
    assert [row["question"] for row in triggered] == ["实验一是什么"]
    assert [row["gate"] for row in detail] == [True, False, False], "抛错要按不触发记，不能带崩整轮"


def test_gate_report_on_empty_input():
    assert gate_report([], lambda q: True) == (0, [], [])


@pytest.mark.parametrize("question,expected", [
    ("列出所有的实验", True),
    ("分别有哪些实验", True),
    ("一共有几个实验", True),
    ("用三点总结主要内容", True),
    ("帮我整理第一章的内容", True),
    ("实验一干啥的", False),
    ("数据库评分标准是什么", False),
    ("", False),
])
def test_enumeration_markers(question, expected):
    assert looks_like_enumeration(question) is expected


def test_assembly_report_counts_files_that_actually_reached_the_context():
    """检索 5 块但只装配了 3 块时，覆盖数必须按装配后的算 —— 装不下就等于没给模型。"""
    used = [{"text": "a", "file_name": "甲.md"},
            {"text": "b", "file_name": "乙.md"},
            {"text": "c", "file_name": "乙.md"}]
    seen, total, missing, blocks = assembly_report(used, ["甲.md", "乙.md", "丙.md"])
    assert (seen, total, blocks) == (2, 3, 3)
    assert missing == ["丙.md"]


def test_assembly_report_ignores_entries_without_source():
    """老索引没有 metadatas 时 file_name 为空，不许把它算成某个文件。"""
    used = [{"text": "a"}, {"text": "b", "file_name": ""}, {"text": "c", "file_name": "甲.md"}]
    seen, total, missing, blocks = assembly_report(used, ["甲.md"])
    assert (seen, total, blocks) == (1, 1, 3)
    assert missing == []


def test_assembly_report_on_empty_input():
    assert assembly_report([], []) == (0, 0, [], 0)


def test_summarize_aggregates_questions_conversations_gate_hits_and_enumeration():
    rows = real_questions(CONVERSATIONS)
    _count, _triggered, detail = gate_report(rows, lambda q: q.startswith("实验"))
    stats = summarize(rows, detail)
    assert stats == {"questions": 4, "conversations": 2, "gate_hits": 2, "enumeration": 2}
