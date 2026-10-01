# -*- coding: utf-8 -*-
"""server 层状态行为回归测试。

这三个用例对应改造前的真实缺陷：
- 每个请求都全量重载状态，且用磁盘内容覆盖内存 → 流式回答丢失
- 会话之间没有任何隔离
"""
import json

import pytest

import src.server as server
import src.state as state


@pytest.fixture
def app_state(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(state, "LEGACY_STATE_FILE", tmp_path / "kb_state.json")
    state._cache.clear()
    state._locks.clear()
    server.sessions.clear()
    # 不碰真实 ChromaDB / 不加载 embedding 模型
    monkeypatch.setattr(server, "init_vector_store", lambda name=None: (None, False))
    yield
    server.sessions.clear()
    state._cache.clear()
    state._locks.clear()


def test_session_is_loaded_from_disk_only_once(app_state, monkeypatch):
    reads = []
    real_read = state.read_file
    monkeypatch.setattr(state, "read_file", lambda sid: (reads.append(sid), real_read(sid))[1])

    sid, sess = server._ensure_session("a" * 16)
    for _ in range(5):
        _, again = server._ensure_session(sid)

    assert again is sess, "同一个会话必须复用同一份内存状态"
    assert len(reads) == 1, f"磁盘只该读一次，实际 {len(reads)} 次"


def test_in_flight_message_survives_concurrent_requests(app_state):
    sid, sess = server._ensure_session("b" * 16)
    sess["conversations"]["c1"] = {
        "id": "c1", "name": "对话", "created_at": 1,
        "files": [{"file_name": "a.txt", "file_text": "x"}], "full_text": "x",
        "collection": None, "use_vector": False, "messages": [], "_cname": "kb_c1",
    }
    server._save_state(sid)

    conv_ref = sess["conversations"]["c1"]        # 流式生成器持有的引用
    server._ensure_session(sid)                    # 模拟并发请求（改造前会在这里把对象换掉）
    conv_ref["messages"].append({"role": "assistant", "content": "这段回答必须落盘"})
    server._save_state(sid)

    saved = json.loads(state.file_for(sid).read_text(encoding="utf-8"))
    assert [m["content"] for m in saved["conversations"]["c1"]["messages"]] == ["这段回答必须落盘"]


def test_two_sessions_do_not_see_each_other(app_state):
    sid_a, sess_a = server._ensure_session("c" * 16)
    sess_a["conversations"]["convA"] = {
        "name": "A 的私有对话", "created_at": 1, "files": [], "messages": [], "full_text": "",
    }
    server._save_state(sid_a)

    sid_b, sess_b = server._ensure_session("d" * 16)
    assert sess_b["conversations"] == {}
    on_disk_b = json.loads(state.file_for(sid_b).read_text(encoding="utf-8"))
    assert on_disk_b["conversations"] == {}

    # A 的数据仍然完好
    on_disk_a = json.loads(state.file_for(sid_a).read_text(encoding="utf-8"))
    assert list(on_disk_a["conversations"]) == ["convA"]
