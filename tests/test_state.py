# -*- coding: utf-8 -*-
"""状态层单测：多会话隔离、持久化、并发、旧数据迁移、路径穿越、坏文件保护。"""
import json
import logging
import threading

import pytest

import src.state as state


@pytest.fixture
def store(tmp_path, monkeypatch):
    """把状态目录和旧版状态文件都指到临时目录，并清掉模块级缓存/锁。"""
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", tmp_path / "kb_state.json")
    state._cache.clear()
    state._locks.clear()
    return state


def _write_legacy(store, conversations, active_conv=None):
    store.LEGACY_STATE_FILE.write_text(
        json.dumps({"active_conv": active_conv, "conversations": conversations}, ensure_ascii=False),
        encoding="utf-8",
    )


# ── 隔离 ──

def test_sessions_are_isolated(store):
    sid_a, sid_b = "a" * 16, "b" * 16
    a = store.load(sid_a)
    a["conversations"]["convA"] = {"name": "A 的对话"}
    a["active_conv"] = "convA"
    assert store.save(sid_a, a)

    b = store.load(sid_b)
    assert b["conversations"] == {}
    assert b["active_conv"] is None

    assert store.file_for(sid_a) != store.file_for(sid_b)
    on_disk_a = json.loads(store.file_for(sid_a).read_text(encoding="utf-8"))
    on_disk_b = json.loads(store.file_for(sid_b).read_text(encoding="utf-8"))
    assert list(on_disk_a["conversations"]) == ["convA"]
    assert on_disk_b["conversations"] == {}


def test_persistence_roundtrip(store):
    sid = "c" * 16
    data = store.load(sid)
    data["conversations"]["c1"] = {"name": "对话 1", "messages": [{"role": "user", "content": "你好"}]}
    assert store.save(sid, data)

    store.drop_cache(sid)                      # 模拟进程重启
    again = store.load(sid)
    assert again["conversations"]["c1"]["messages"][0]["content"] == "你好"


def test_load_hits_disk_only_once(store, monkeypatch):
    reads = []
    real_read = store.read_file
    monkeypatch.setattr(store, "read_file", lambda sid: (reads.append(sid), real_read(sid))[1])

    sid = "d" * 16
    for _ in range(5):
        store.load(sid)
    assert len(reads) == 1, f"同一个会话只该读一次磁盘，实际 {len(reads)} 次"


# ── 并发 ──

def test_concurrent_save_never_fails_and_leaves_no_tmp(store, caplog):
    sid = "e" * 16
    store.load(sid)
    results = []

    def worker(n):
        payload = {"active_conv": "c1", "conversations": {"c1": {"name": f"并发-{n}", "messages": []}}}
        for _ in range(25):
            results.append(store.save(sid, payload))

    with caplog.at_level(logging.ERROR, logger="ai_rag.state"):
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert all(results), "出现写入失败"
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert not errors, f"写入过程产生错误日志: {errors}"
    assert not list(store.STATE_DIR.glob("*.tmp")), "有残留临时文件"

    final = json.loads(store.file_for(sid).read_text(encoding="utf-8"))
    assert final["session_id"] == sid
    assert final["conversations"]["c1"]["name"].startswith("并发-")


# ── 旧数据迁移 ──

def test_legacy_claim_is_one_shot(store):
    _write_legacy(store, {"old1": {"name": "旧对话", "messages": []}}, active_conv="old1")

    first = store.load("1" * 16, allow_legacy=True)
    assert list(first["conversations"]) == ["old1"]
    assert first["active_conv"] == "old1"

    # 原文件被原子改名，第二个会话再也认领不到
    assert not store.LEGACY_STATE_FILE.exists()
    assert store.LEGACY_STATE_FILE.with_suffix(".json.migrated").exists()

    second = store.load("2" * 16, allow_legacy=True)
    assert second["conversations"] == {}


def test_new_session_does_not_steal_legacy(store):
    _write_legacy(store, {"old1": {"name": "旧对话"}})

    fresh = store.load("3" * 16, allow_legacy=False)      # /api/session 发出的新 sid
    assert fresh["conversations"] == {}
    assert store.LEGACY_STATE_FILE.exists(), "新 sid 不应认领旧状态"

    # 之后自带旧 sid 的浏览器仍然能拿到
    owner = store.load("4" * 16, allow_legacy=True)
    assert list(owner["conversations"]) == ["old1"]


# ── 健壮性 ──

def test_new_session_logs_no_noise(store, caplog):
    """旧状态文件本来就不存在时，不该刷 WARNING。"""
    with caplog.at_level(logging.WARNING, logger="ai_rag.state"):
        store.load("7" * 16)
    assert not caplog.records, [r.getMessage() for r in caplog.records]


def test_session_id_cannot_escape_state_dir(store, tmp_path):
    for evil in ["../../evil", "..\\..\\evil", "/etc/passwd", "a/../../b", "."]:
        assert store.file_for(evil).parent == store.STATE_DIR

    store.load("../../evil")
    assert list(store.STATE_DIR.glob("*.json")), "状态文件应写在 STATE_DIR 内"
    assert not (tmp_path / "evil.json").exists()


def test_corrupt_file_is_backed_up_not_silently_reset(store):
    sid = "f" * 16
    store.STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = store.file_for(sid)
    path.write_text("{ 这不是合法 JSON", encoding="utf-8")

    data = store.load(sid)
    assert data["conversations"] == {}                        # 降级为空状态，不抛异常
    backups = list(store.STATE_DIR.glob("*.corrupt-*.json"))
    assert backups, "坏文件必须先备份，不能直接丢掉"
    assert "这不是合法 JSON" in backups[0].read_text(encoding="utf-8")
    # 备份之后补建一个新的空状态文件，保证会话还能继续用
    assert json.loads(path.read_text(encoding="utf-8"))["conversations"] == {}


def test_wrong_owner_file_is_not_overwritten(store):
    sid = "9" * 16
    store.STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = store.file_for(sid)
    path.write_text(json.dumps({"session_id": "别人的-sid", "conversations": {"x": {}}}), encoding="utf-8")

    data = store.load(sid)
    assert data["conversations"] == {}
    assert json.loads(path.read_text(encoding="utf-8"))["conversations"] == {"x": {}}, "不该覆盖归属不符的文件"
