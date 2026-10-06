# -*- coding: utf-8 -*-
"""真实语料基线的纯函数回归（P0 收尾）。

真实语料测不了 recall/MRR（没有 gold 标注），能诚实测的只有两类数字：切窗诊断与距离带。
这里只测「读状态 → 语料/提问」「距离带汇总」这些纯逻辑 —— 不加载模型、不碰真实 state/chroma，
embedding 与真实语料那一跑留在 `python scripts/eval_real_corpus.py` 时验证。
"""
import json

import pytest

from scripts import eval_real_corpus as erc
from scripts import eval_retrieval as ev


def _write_state(state_dir, session_id, conversations):
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"{session_id}.json"
    path.write_text(json.dumps({
        "version": 2,
        "session_id": session_id,
        "conversations": conversations,
    }, ensure_ascii=False), encoding="utf-8")
    return path


# ── 从真实状态里取语料与提问 ──

def test_conversations_from_state_collects_files_and_user_questions(tmp_path):
    state_dir = tmp_path / "state"
    _write_state(state_dir, "sid-1", {
        "c1": {
            "files": [{"file_name": "a.md", "file_text": "正文"}],
            "messages": [{"role": "user", "content": "问题一"},
                         {"role": "assistant", "content": "回答"},
                         {"role": "user", "content": "  问题二  "}],
        },
    })

    got = erc.conversations_from_state(state_dir)

    assert [c["conv_id"] for c in got] == ["c1"]
    assert got[0]["files"] == [("a.md", "正文")]
    assert got[0]["questions"] == ["问题一", "问题二"], "只取用户提问，并去掉首尾空白"


def test_conversations_from_state_skips_conversations_without_files(tmp_path):
    state_dir = tmp_path / "state"
    _write_state(state_dir, "sid-2", {
        "empty": {"files": [], "messages": [{"role": "user", "content": "问题"}]},
        "ok": {"files": [{"file_name": "a.md", "file_text": "正文"}], "messages": []},
    })

    got = erc.conversations_from_state(state_dir)

    assert [c["conv_id"] for c in got] == ["ok"], "没有文件的对话没有可检索的东西"


def test_conversations_from_state_skips_broken_file(tmp_path, caplog):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "bad.json").write_text("{ 这不是合法 JSON", encoding="utf-8")
    _write_state(state_dir, "sid-ok", {
        "c9": {"files": [{"file_name": "y.md", "file_text": "正文"}], "messages": []},
    })

    got = erc.conversations_from_state(state_dir)

    assert [c["conv_id"] for c in got] == ["c9"], "一个坏文件不该让整次统计失败"


def test_conversations_from_state_missing_dir_is_empty(tmp_path):
    assert erc.conversations_from_state(tmp_path / "nope") == []


def test_conversations_from_state_ignores_blank_questions(tmp_path):
    state_dir = tmp_path / "state"
    _write_state(state_dir, "sid-3", {
        "c1": {"files": [{"file_name": "a.md", "file_text": "正文"}],
               "messages": [{"role": "user", "content": "   "}, {"role": "user"}]},
    })

    assert erc.conversations_from_state(state_dir)[0]["questions"] == []


# ── 距离带 ──

def test_band_stats_reports_band_and_over_threshold():
    got = erc.band_stats([0.30, 0.50, 0.70, 0.90], threshold=0.55)

    assert got["count"] == 4
    assert got["min"] == pytest.approx(0.30)
    assert got["median"] == pytest.approx(0.60)
    assert got["max"] == pytest.approx(0.90)
    assert got["over_threshold"] == 2, "> 0.55 的两条会被关键词兜底接走"
    assert got["over_rate"] == pytest.approx(0.5)


def test_band_stats_threshold_is_exclusive():
    """正好等于阈值不算超：线上是 `dist <= MAX_DISTANCE` 保留、`> ` 才丢。"""
    got = erc.band_stats([0.55], threshold=0.55)

    assert got["over_threshold"] == 0
    assert got["over_rate"] == pytest.approx(0.0)


def test_band_stats_is_all_none_when_empty():
    got = erc.band_stats([], threshold=0.55)

    assert got["count"] == 0
    assert got["min"] is None and got["median"] is None and got["max"] is None
    assert got["over_threshold"] == 0
    assert got["over_rate"] is None, "没有样本时不许给一个会骗人的比例"


# ── 无关提问的代理（真实语料没有 gold 负样本）──

def test_negative_questions_reads_only_negatives(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([
        {"question": "量子纠缠", "kind": "negative"},
        {"question": "净资产收益率", "kind": "positive"},
        {"question": "  ", "kind": "negative"},
    ], ensure_ascii=False), encoding="utf-8")

    assert erc.negative_questions(path) == ["量子纠缠"]


def test_annotation_file_really_has_negative_questions():
    """代理要是空的，第二行的「无关侧」就成了空话 —— 钉一下标注文件。"""
    got = erc.negative_questions(ev.DEFAULT_QUERIES)

    assert len(got) >= 3
    assert all(q.strip() for q in got)


# ── 原始查询的最佳距离 ──

class _FakeCollection:
    def __init__(self, distances=None, boom=False):
        self._distances = distances or []
        self._boom = boom

    def query(self, query_texts, n_results):
        if self._boom:
            raise RuntimeError("向量库挂了")
        return {"distances": [self._distances]}


def test_best_distance_takes_the_closest_of_the_returned_chunks():
    """原始查询取最佳距离（不经阈值）—— 阈值只决定它接下来走哪条路。"""
    coll = _FakeCollection([0.62, 0.41, 0.55])

    assert erc.best_distance(coll, "问题", top_k=3, count=3) == pytest.approx(0.41)


def test_best_distance_is_none_when_the_query_fails():
    assert erc.best_distance(_FakeCollection(boom=True), "问题", top_k=3, count=3) is None


def test_best_distance_is_none_when_nothing_came_back():
    assert erc.best_distance(_FakeCollection([]), "问题", top_k=3, count=3) is None


# ── HyDE 调用缓存与自检的用例已随助手函数搬到 tests/test_eval_retrieval.py ──
# （助手函数现在住在 scripts/eval_retrieval.py，本脚本只是复用）
