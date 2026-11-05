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


# ── 原始查询的最佳距离（双路探针）──

class _FakeCollection:
    """`per_query` 给了就按 query 文本各回一条列表，否则永远只回一条（模拟老实现）。"""

    def __init__(self, distances=None, boom=False):
        self._distances = distances or []
        self._boom = boom
        self.per_query = None

    def query(self, query_texts, n_results):
        if self._boom:
            raise RuntimeError("向量库挂了")
        if self.per_query is None:
            return {"distances": [self._distances]}
        return {"distances": [self.per_query.get(text, []) for text in query_texts]}


def test_probe_distances_returns_one_best_distance_per_probe():
    """两条探针（HyDE / 原问题）各自取最佳距离；融合闸门用的是两者中的较小值。"""
    coll = _FakeCollection([0.62, 0.41, 0.55])
    coll.per_query = {"假设文本": [0.62, 0.41, 0.55], "问题": [0.30, 0.90]}

    hyde, question = erc.probe_distances(coll, "假设文本", "问题", top_k=3, count=3)

    assert hyde == pytest.approx(0.41)
    assert question == pytest.approx(0.30)
    assert min(hyde, question) == pytest.approx(0.30)


def test_probe_distances_is_none_when_the_query_fails():
    assert erc.probe_distances(_FakeCollection(boom=True), "假设", "问题",
                               top_k=3, count=3) == (None, None)


def test_probe_distances_is_none_when_nothing_came_back():
    assert erc.probe_distances(_FakeCollection([]), "假设", "问题",
                               top_k=3, count=3) == (None, None)


def test_probe_distances_treats_a_missing_second_list_as_no_measurement():
    """老实现/替身只回一条列表时，第二路必须是 None（而不是拿第一路的值冒充）。"""
    coll = _FakeCollection([0.42])
    coll.per_query = None                     # 无论几条 query 都只回一条列表

    hyde, question = erc.probe_distances(coll, "假设", "问题", top_k=3, count=1)

    assert hyde == pytest.approx(0.42)
    assert question is None


# ── HyDE 调用缓存与自检的用例已随助手函数搬到 tests/test_eval_retrieval.py ──
# （助手函数现在住在 scripts/eval_retrieval.py，本脚本只是复用）
