# -*- coding: utf-8 -*-
"""离线复核脚本的回归：判定口径的改动必须能拿**旧回答**重算，不许重跑模型。"""
import json

import pytest

from scripts import rejudge_abstention as rj


def _queries():
    return [{
        "id": "p01", "kind": "partial", "question": "全量备份保留多久，异地机房怎么切换",
        "expect_files": ["04-数据备份与恢复.md"],
        "answer_terms": ["保留 30 天"], "absent_terms": ["异地机房"],
        "parts": [{"text": "全量备份保留多久", "answer_terms": ["保留 30 天"], "absent_terms": []},
                  {"text": "异地机房怎么切换", "answer_terms": [], "absent_terms": ["异地机房"]}],
        "shape": "two_parts", "expect_refused": False,
    }]


def test_rejudge_recomputes_the_bucket_from_the_stored_answer():
    rows = [{"id": "p01", "kind": "partial", "bucket": "neither",
             "answer": "全量备份保留 **30 天**。参考背景中没有提到异地机房的切换。"}]

    rows, changed = rj.rejudge(rows, _queries())

    assert rows[0]["bucket"] == "answered_with_gap"
    assert rows[0]["coverage"] == {"covered": 1, "total": 1}
    assert changed == ["p01"], "旧判与新判不同时必须报出来，否则看不出量尺修过"


def test_rejudge_keeps_rows_without_an_answer_untouched():
    """没生成出回答的行不参与重算 —— 否则「检索没召回」会被误报成「模型拒答」。"""
    rows = [{"id": "p01", "kind": "partial", "returned": 0, "answer": ""}]

    rows, changed = rj.rejudge(rows, _queries())

    assert rows[0] == {"id": "p01", "kind": "partial", "returned": 0, "answer": ""}
    assert changed == []


def test_rejudge_ignores_rows_with_unknown_ids():
    rows = [{"id": "q99", "kind": "positive", "answer": "随便"}]

    rows, changed = rj.rejudge(rows, _queries())

    assert rows[0]["answer"] == "随便"
    assert changed == []


def test_main_reports_a_missing_detail_file(tmp_path, capsys):
    assert rj.main([str(tmp_path / "nope.json")]) == 2
    assert "找不到明细文件" in capsys.readouterr().out


def test_main_prints_a_summary_and_can_write_output(tmp_path, capsys):
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps(_queries(), ensure_ascii=False), encoding="utf-8")
    detail = tmp_path / "detail.json"
    detail.write_text(json.dumps([
        {"id": "p01", "kind": "partial", "bucket": "neither",
         "answer": "全量备份保留 30 天。参考背景中没有提到异地机房的切换。"},
    ], ensure_ascii=False), encoding="utf-8")

    assert rj.main([str(detail), "--queries", str(queries), "--write"]) == 0

    out = capsys.readouterr().out
    assert "与原判不同的样本： ['p01']" in out
    assert "复合问题的形状" in out
    written = json.loads((tmp_path / "detail-rejudged.json").read_text(encoding="utf-8"))
    assert written[0]["bucket"] == "answered_with_gap"
