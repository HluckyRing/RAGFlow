# -*- coding: utf-8 -*-
"""孤儿 collection 清理脚本回归（P2）。

这个脚本会**删数据**，所以除了判定逻辑，重点守两条安全性质：
- 不加 --apply 时一个都不许删（干跑是默认）
- 一个被引用的集合都没扫到（通常是 --state-dir 指错）时必须中止，不能照删
"""
import json
from pathlib import Path

import chromadb
import pytest

from scripts import cleanup_orphan_collections as clean


# ── 判定逻辑 ──

def test_find_orphans_splits_project_and_foreign():
    orphans, foreign = clean.find_orphans(
        ["kb_conv_a_1", "kb_conv_b_2", "knowledge_base"],
        referenced={"kb_conv_a_1"},
    )
    assert orphans == ["kb_conv_b_2"]
    assert foreign == ["knowledge_base"], "非本项目命名的集合默认只报告"


def test_find_orphans_can_include_foreign():
    orphans, foreign = clean.find_orphans(
        ["kb_conv_b_2", "knowledge_base"], referenced=set(), include_foreign=True)
    assert orphans == ["kb_conv_b_2", "knowledge_base"]
    assert foreign == []


def test_find_orphans_keeps_referenced():
    orphans, foreign = clean.find_orphans(["kb_conv_a_1"], referenced={"kb_conv_a_1"})
    assert orphans == [] and foreign == []


def test_collect_referenced_reads_state_and_legacy(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "a.json").write_text(json.dumps({
        "conversations": {
            "c1": {"collection_name": "kb_conv_a_1"},
            "c2": {"_cname": "kb_conv_b_2"},
            "c3": {"name": "没有 collection_name 的旧对话"},
        }
    }, ensure_ascii=False), encoding="utf-8")
    legacy = tmp_path / "kb_state.json"
    legacy.write_text(json.dumps({"conversations": {"c9": {"collection_name": "kb_conv_c_3"}}}),
                      encoding="utf-8")

    got = clean.collect_referenced([state_dir / "a.json", legacy])

    assert got == {"kb_conv_a_1", "kb_conv_b_2", "kb_conv_c_3"}


def test_broken_state_file_is_skipped_not_fatal(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ 这不是合法 JSON", encoding="utf-8")
    missing = tmp_path / "nope.json"

    assert clean.collect_referenced([bad, missing]) == set()


# ── 真 ChromaDB 集成（临时库，不碰真实数据）──

@pytest.fixture
def env(tmp_path):
    db = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(db))
    for name in ("kb_conv_keep_1", "kb_conv_orphan_1", "kb_conv_orphan_2", "knowledge_base"):
        client.create_collection(name=name)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "s.json").write_text(json.dumps({
        "conversations": {"c1": {"collection_name": "kb_conv_keep_1"}}
    }, ensure_ascii=False), encoding="utf-8")
    yield client, db, state_dir


def _names(client):
    return sorted(c.name for c in client.list_collections())


def _run(db, state_dir, tmp_path, *extra):
    return clean.main([
        "--db-path", str(db),
        "--state-dir", str(state_dir),
        "--legacy-file", str(tmp_path / "no-such-legacy.json"),
        *extra,
    ])


def test_dry_run_deletes_nothing(env, tmp_path, capsys):
    client, db, state_dir = env

    rc = _run(db, state_dir, tmp_path)

    assert rc == 0
    assert len(_names(client)) == 4, "干跑不许动任何集合"
    out = capsys.readouterr().out
    assert "kb_conv_orphan_1" in out and "--apply" in out


def test_apply_deletes_only_project_orphans(env, tmp_path, capsys):
    client, db, state_dir = env

    rc = _run(db, state_dir, tmp_path, "--apply")

    assert rc == 0
    assert _names(client) == ["kb_conv_keep_1", "knowledge_base"], "被引用的与外来集合都要留下"


def test_include_foreign_also_deletes_foreign(env, tmp_path):
    client, db, state_dir = env

    rc = _run(db, state_dir, tmp_path, "--apply", "--include-foreign")

    assert rc == 0
    assert _names(client) == ["kb_conv_keep_1"]


def test_apply_refuses_when_nothing_is_referenced(env, tmp_path, capsys):
    """扫不到被引用集合通常是路径指错了 —— 必须中止，而不是把库删空。"""
    client, db, state_dir = env
    (state_dir / "s.json").write_text(json.dumps({"conversations": {}}), encoding="utf-8")

    rc = _run(db, state_dir, tmp_path, "--apply")

    assert rc == 2, "应以便于识别的退出码中止"
    assert len(_names(client)) == 4, "中止时一个都不许删"
    assert "--force" in capsys.readouterr().out


def test_force_overrides_the_refusal(env, tmp_path):
    client, db, state_dir = env
    (state_dir / "s.json").write_text(json.dumps({"conversations": {}}), encoding="utf-8")

    rc = _run(db, state_dir, tmp_path, "--apply", "--force")

    assert rc == 0
    assert _names(client) == ["knowledge_base"]


def test_missing_db_path_is_a_noop(tmp_path, capsys):
    rc = clean.main([
        "--db-path", str(tmp_path / "not-created-yet"),
        "--state-dir", str(tmp_path / "state"),
        "--legacy-file", str(tmp_path / "no-such-legacy.json"),
    ])

    assert rc == 0
    assert not (tmp_path / "not-created-yet").exists(), "干跑不该顺手把库目录建出来"
