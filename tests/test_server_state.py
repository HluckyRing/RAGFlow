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


# ── P1-9 状态瘦身：full_text 不再重复落盘，改为从 files 派生 ──

def _conv_with_two_files():
    return {
        "id": "c1", "name": "对话", "created_at": 1,
        "files": [{"file_name": "a.txt", "file_text": "第一段"},
                  {"file_name": "b.txt", "file_text": "第二段"}],
        "full_text": "第一段\n\n第二段",
        "collection": None, "use_vector": False, "messages": [], "_cname": "kb_c1",
    }


def test_state_file_does_not_store_duplicate_full_text(app_state):
    """full_text 等于 files 的拼接，重复存等于把正文在磁盘上写两遍。"""
    sid, sess = server._ensure_session("e" * 16)
    sess["conversations"]["c1"] = _conv_with_two_files()
    server._save_state(sid)

    saved = json.loads(state.file_for(sid).read_text(encoding="utf-8"))
    conv = saved["conversations"]["c1"]
    assert "full_text" not in conv, "full_text 是派生值，不该落盘"
    assert [f["file_text"] for f in conv["files"]] == ["第一段", "第二段"], "原始文本必须保留"


def test_full_text_is_derived_from_files_on_load(app_state):
    """磁盘上只有 files（新格式）时，加载后必须能把全文拼回来给检索用。"""
    sid = "f" * 16
    state.STATE_DIR.mkdir(parents=True, exist_ok=True)
    state.file_for(sid).write_text(json.dumps({
        "version": 2, "session_id": sid, "active_conv": "c1",
        "conversations": {"c1": {
            "name": "对话", "created_at": 1,
            "files": [{"file_name": "a.txt", "file_text": "第一段"},
                      {"file_name": "b.txt", "file_text": "第二段"}],
            "messages": [],
        }},
    }, ensure_ascii=False), encoding="utf-8")
    state._cache.clear()

    _, sess = server._ensure_session(sid)
    assert sess["conversations"]["c1"]["full_text"] == "第一段\n\n第二段"


def test_legacy_state_with_full_text_still_loads(app_state):
    """旧状态文件里带 full_text，仍要能无损加载（向后兼容）。"""
    sid = "1" * 16
    state.STATE_DIR.mkdir(parents=True, exist_ok=True)
    state.file_for(sid).write_text(json.dumps({
        "version": 2, "session_id": sid, "active_conv": "c1",
        "conversations": {"c1": {
            "name": "旧对话", "created_at": 1,
            "files": [{"file_name": "a.txt", "file_text": "旧文本"}],
            "full_text": "旧文本", "messages": [],
        }},
    }, ensure_ascii=False), encoding="utf-8")
    state._cache.clear()

    _, sess = server._ensure_session(sid)
    assert sess["conversations"]["c1"]["full_text"] == "旧文本"
    assert sess["conversations"]["c1"]["files"][0]["file_text"] == "旧文本"
