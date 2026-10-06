# -*- coding: utf-8 -*-
"""重建索引脚本回归（P1）。

`reindex_all.py` 会**重写向量库**，所以除了「重建成功」，重点守三条性质：
- 不加 `--apply` 时一个 collection 都不许打开（干跑是默认）
- `--apply` 走的是 `server._rebuild_collection` 的「先加新、再删旧」，不是清空重来
- 坏状态文件跳过，不能让一个坏文件把别的对话漏掉

测试全部注入假 collection（`open_collection` 是为此留的接缝）——真实实现会加载
embedding 模型，而仓库约定测试套件不碰 torch / sentence-transformers。
"""
import json

import pytest

import src.server as server
import src.state as state
from scripts import reindex_all


class FakeCollection:
    """按 id 记账的替身，另外记下 add/delete 的调用顺序。"""

    def __init__(self, docs=None):
        self.docs = dict(docs or {})
        self.metas = {}
        self.events = []

    def count(self):
        return len(self.docs)

    def get(self):
        return {"ids": list(self.docs.keys())}

    def add(self, documents, ids, metadatas=None):
        self.events.append(("add", list(ids)))
        for i, doc in zip(ids, documents):
            self.docs[i] = doc
        for i, meta in zip(ids, metadatas or []):
            self.metas[i] = meta

    def delete(self, ids):
        self.events.append(("delete", list(ids)))
        for i in ids:
            self.docs.pop(i, None)


def _write_state(state_dir, session_id, conversations):
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"{session_id}.json"
    path.write_text(json.dumps({
        "version": 2,
        "session_id": session_id,
        "conversations": conversations,
    }, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    """一个会话、两个对话：c1 有文件，c2 没有（应被跳过）。"""
    state_dir = tmp_path / "state"
    _write_state(state_dir, "sid-1", {
        "c1": {
            "files": [{"file_name": "a.md", "file_text": "第一段。\n\n第二段。"}],
            "collection_name": "kb_conv_c1",
        },
        "c2": {"files": [], "collection_name": "kb_conv_c2"},
    })
    opened = {}

    def fake_open(name):
        coll = FakeCollection(docs={f"old_{name}_0": "旧块"})
        opened[name] = coll
        return coll

    monkeypatch.setattr(reindex_all, "open_collection", fake_open)
    monkeypatch.setattr(state, "STATE_DIR", state_dir)
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", tmp_path / "no-legacy.json")
    return state_dir, opened


def test_dry_run_touches_nothing(env, capsys):
    state_dir, opened = env

    rc = reindex_all.main(["--state-dir", str(state_dir)])

    assert rc == 0
    assert opened == {}, "干跑不许打开/重建任何 collection"
    out = capsys.readouterr().out
    assert "c1" in out and "--apply" in out
    assert "c2" not in out, "没有文件的对话不该出现在重建计划里"


def test_apply_rebuilds_and_adds_before_deleting(env):
    state_dir, opened = env

    rc = reindex_all.main(["--state-dir", str(state_dir), "--apply"])

    assert rc == 0
    assert list(opened) == ["kb_conv_c1"], "无文件的对话不该被打开"
    coll = opened["kb_conv_c1"]
    assert coll.docs, "重建后应留下新块"
    assert all(not i.startswith("old_") for i in coll.docs), "旧 id 必须被清理"

    kinds = [event[0] for event in coll.events]
    assert kinds[0] == "add", "必须先写新块"
    assert kinds[-1] == "delete", "旧块要留到新块写成功之后再删"
    assert "delete" not in kinds[:-1]


def test_conv_without_collection_name_uses_derived_name(tmp_path, monkeypatch):
    """状态里没写 collection_name 的对话，沿用服务端的派生规则。"""
    state_dir = tmp_path / "state"
    _write_state(state_dir, "sid-2", {
        "c7": {"files": [{"file_name": "x.md", "file_text": "内容"}]},
    })
    opened = {}
    monkeypatch.setattr(reindex_all, "open_collection",
                        lambda name: opened.setdefault(name, FakeCollection()))
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", tmp_path / "no-legacy.json")

    rc = reindex_all.main(["--state-dir", str(state_dir), "--apply"])

    assert rc == 0
    assert list(opened) == [server._conv_cname("c7")]


def test_broken_state_file_is_skipped_not_fatal(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "bad.json").write_text("{ 这不是合法 JSON", encoding="utf-8")
    _write_state(state_dir, "sid-ok", {
        "c9": {"files": [{"file_name": "y.md", "file_text": "内容"}],
               "collection_name": "kb_conv_c9"},
    })
    opened = {}
    monkeypatch.setattr(reindex_all, "open_collection",
                        lambda name: opened.setdefault(name, FakeCollection()))
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", tmp_path / "no-legacy.json")

    rc = reindex_all.main(["--state-dir", str(state_dir), "--apply"])

    assert rc == 0, "一个坏文件不该让整次重建失败"
    assert list(opened) == ["kb_conv_c9"], "好文件里的对话必须照常重建"


def test_rebuild_failure_is_reported_with_exit_code_1(env, monkeypatch, capsys):
    state_dir, _opened = env

    def boom(name):
        raise RuntimeError("模型挂了")

    monkeypatch.setattr(reindex_all, "open_collection", boom)

    rc = reindex_all.main(["--state-dir", str(state_dir), "--apply"])

    assert rc == 1
    assert "模型挂了" in capsys.readouterr().out


def test_missing_state_dir_aborts_unless_forced(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", tmp_path / "no-legacy.json")

    rc = reindex_all.main(["--state-dir", str(tmp_path / "nope")])

    assert rc == 2, "扫不到状态文件通常说明路径指错，必须中止"
    assert "--force" in capsys.readouterr().out

    rc = reindex_all.main(["--state-dir", str(tmp_path / "nope"), "--force"])

    assert rc == 0, "--force 越过安全闸之后应是「无事可做」"


def test_no_conversation_with_files_is_a_noop(tmp_path, monkeypatch, capsys):
    state_dir = tmp_path / "state"
    _write_state(state_dir, "sid-3", {"c0": {"files": [], "collection_name": "kb_x"}})
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", tmp_path / "no-legacy.json")
    monkeypatch.setattr(reindex_all, "open_collection",
                        lambda name: pytest.fail("没有可重建的对话时不该打开 collection"))

    rc = reindex_all.main(["--state-dir", str(state_dir), "--apply"])

    assert rc == 0
    assert "没有需要重建的对话" in capsys.readouterr().out


def test_apply_writes_source_metadata(tmp_path, monkeypatch):
    """重建时要把来源写进 metadatas（P2）—— 引用可核验的数据基础。"""
    state_dir = tmp_path / "state"
    _write_state(state_dir, "sid-meta", {
        "c1": {"files": [{"file_name": "a.md", "file_text": "第一段。\n\n第二段。"}],
               "collection_name": "kb_conv_meta"},
    })
    opened = {}
    monkeypatch.setattr(reindex_all, "open_collection",
                        lambda name: opened.setdefault(name, FakeCollection()))
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", tmp_path / "no-legacy.json")

    rc = reindex_all.main(["--state-dir", str(state_dir), "--apply"])

    assert rc == 0
    metas = list(opened["kb_conv_meta"].metas.values())
    assert metas and all(m["file_name"] == "a.md" for m in metas)
    assert [m["chunk_index"] for m in metas] == list(range(len(metas)))
    # Chroma 收不了 None，所以 chroma_meta 会把「没有页码」的 section 字段整个去掉；
    # 读的那一侧必须用 .get()，不能假设这个键一定存在。
    assert all(m.get("section") is None for m in metas), "md 没有结构标记，不该编造页码"


def _write_legacy(path, conversations):
    """旧版全局状态文件：没有 session_id（靠文件名兜底），只有 conversations。"""
    path.write_text(json.dumps({"active_conv": None, "conversations": conversations},
                               ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def dup_env(tmp_path, monkeypatch):
    """同一个对话同时存在于 `state/` 会话文件与 legacy `.json.migrated`，内容不同。

    这是真实数据的形态：迁移前的旧快照与线上会话**同 conv_id、同 collection_name**，
    但文件列表和对话轮次都是旧的。不处理的话，后扫到的旧快照会覆盖线上索引。
    """
    state_dir = tmp_path / "state"
    _write_state(state_dir, "sid-live", {
        "live1": {
            "files": [{"file_name": "live.md", "file_text": "线上内容：第一段。\n\n第二段。"}],
            "collection_name": "kb_conv_live1",
        },
    })
    legacy = tmp_path / "kb_state.json"
    _write_legacy(legacy.with_suffix(".json.migrated"), {
        "live1": {
            "files": [{"file_name": "old.md", "file_text": "旧快照：迁移前的内容。"}],
            "collection_name": "kb_conv_live1",
        },
        "only-legacy": {
            "files": [{"file_name": "only.md", "file_text": "只存在于旧文件里的对话。"}],
            "collection_name": "kb_conv_only_legacy",
        },
    })
    opened = {}
    monkeypatch.setattr(reindex_all, "open_collection",
                        lambda name: opened.setdefault(name, FakeCollection()))
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", legacy)
    return state_dir, opened


def test_duplicated_conv_id_is_rebuilt_once_from_the_live_state(dup_env):
    """state/ 与 legacy 撞同一个 conv_id 时只认 state/ 那份，并且只重建一次。

    两份的 collection_name 是同一个（由 conv_id 派生），所以「两份都重建」= 后跑的
    legacy 旧快照覆盖线上索引 —— 这是静默毁数据，必须由测试钉死。
    """
    state_dir, opened = dup_env

    rc = reindex_all.main(["--state-dir", str(state_dir), "--apply"])

    assert rc == 0
    coll = opened["kb_conv_live1"]
    kinds = [event[0] for event in coll.events]
    assert kinds.count("add") == 1, "同一个对话只许重建一次，否则后一份会覆盖前一份"
    assert any("线上内容" in doc for doc in coll.docs.values()), "必须用 state/ 里的内容"
    assert all("旧快照" not in doc for doc in coll.docs.values()), "旧快照不许进线上索引"


def test_legacy_only_conversation_is_still_rebuilt(dup_env):
    """去重只针对重复项：只在 legacy 里出现的对话照常重建，不整个丢掉 legacy。"""
    state_dir, opened = dup_env

    rc = reindex_all.main(["--state-dir", str(state_dir), "--apply"])

    assert rc == 0
    assert "kb_conv_only_legacy" in opened


def test_dry_run_lists_duplicates_separately(dup_env, capsys):
    """干跑要说清「跳过了谁」，否则用户看不懂计划里为什么少了对话。"""
    state_dir, opened = dup_env

    rc = reindex_all.main(["--state-dir", str(state_dir)])

    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("[待重建] live1") == 1, "重复的对话在计划里只该出现一次"
    assert "[待重建] only-legacy" in out, "legacy 独有的对话仍要出现在计划里"
    assert "[跳过] live1" in out, "被跳过的重复项要明确列出来"
    assert opened == {}, "干跑不许打开 collection"


def test_same_conv_id_across_sessions_rebuilds_once_and_warns(tmp_path, monkeypatch,
                                                              capsys, caplog):
    """跨会话撞 conv_id（理论上不该发生）也只重建一次，并留下告警。"""
    state_dir = tmp_path / "state"
    _write_state(state_dir, "sid-a", {
        "dup": {"files": [{"file_name": "a.md", "file_text": "会话 A 的内容。"}],
                "collection_name": "kb_conv_dup"},
    })
    _write_state(state_dir, "sid-b", {
        "dup": {"files": [{"file_name": "b.md", "file_text": "会话 B 的内容。"}],
                "collection_name": "kb_conv_dup"},
    })
    opened = {}
    monkeypatch.setattr(reindex_all, "open_collection",
                        lambda name: opened.setdefault(name, FakeCollection()))
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", tmp_path / "no-legacy.json")

    rc = reindex_all.main(["--state-dir", str(state_dir), "--apply"])

    assert rc == 0
    kinds = [event[0] for event in opened["kb_conv_dup"].events]
    assert kinds.count("add") == 1, "共用同一个 collection 的对话只能重建一次"
    assert "[跳过] dup" in capsys.readouterr().out
    assert "跳过重复对话 dup" in caplog.text, "跳过了什么要有告警，不能静默"
